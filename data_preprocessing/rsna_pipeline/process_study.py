"""Study-level integration for RSNA DICOM geometry and vertebra masks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from data_preprocessing.rsna_pipeline.classifier_plane_sampling import (
    SampledClassifierPlanes,
    load_hu_volume,
    sample_classifier_planes,
    sampling_metadata,
    write_study_classifier_outputs_atomic,
)
from data_preprocessing.rsna_pipeline.classifier_planes import (
    ClassifierPlanePlan,
    assign_bbox_centers_to_vertebrae,
    build_classifier_plane_plan,
    load_study_bbox_centers,
    load_study_fracture_levels,
)
from data_preprocessing.rsna_pipeline.dicom_geometry import (
    DicomSeriesGeometry,
    GeometryValidationError,
    load_approximate_dicom_series_from_nifti,
    load_dicom_series,
)
from data_preprocessing.rsna_pipeline.mask_processing import (
    ProcessedVertebraMask,
    load_and_process_vertebra_mask,
)
from data_preprocessing.rsna_pipeline.orientation import (
    OrientationSearchResult,
    find_best_physical_orientation,
)
from data_preprocessing.rsna_pipeline.segmentation_plane_sampling import (
    SampledSegmentationPlanes,
    sample_segmentation_planes,
    segmentation_sampling_metadata,
    write_study_segmentation_outputs_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
TRAIN_IMAGES_DIR = RSNA_DATA_DIR / "train_images"
SEGMENTATION_DIR = RSNA_DATA_DIR / "segmentations"
STUDY_METADATA_DIR = RSNA_DATA_DIR / "processing_metadata"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
BOUNDING_BOX_CSV = RSNA_DATA_DIR / "train_bounding_boxes.csv"
TRAIN_LABEL_CSV = RSNA_DATA_DIR / "train.csv"

VERTEBRA_LEVELS: Final = tuple(f"C{level}" for level in range(1, 8))
PIPELINE_VERSION: Final = "rsna-preprocessing-v2"


class StudyProcessingError(RuntimeError):
    """Raised when a study cannot produce complete processing metadata."""


@dataclass(frozen=True)
class StudyProcessingResult:
    """Processed study geometry, masks, and persisted metadata path."""

    study_id: str
    geometry: DicomSeriesGeometry
    vertebrae: dict[str, ProcessedVertebraMask]
    orientations: dict[str, OrientationSearchResult]
    classifier_planes: dict[str, ClassifierPlanePlan]
    sampled_classifier_planes: dict[str, SampledClassifierPlanes]
    sampled_segmentation_planes: dict[str, SampledSegmentationPlanes]
    metadata_path: Path
    output_directory: Path


def process_study(
    study_directory: Path,
    segmentation_directory: Path,
    metadata_path: Path,
    output_directory: Path,
    *,
    bbox_csv_path: Path | None = BOUNDING_BOX_CSV,
    excluded_levels: frozenset[str] = frozenset(),
) -> StudyProcessingResult:
    """Process one study and atomically persist sampled classifier planes."""
    study_id = study_directory.name
    if not study_id:
        raise StudyProcessingError("Study directory must have a name")
    if not segmentation_directory.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory not found: {segmentation_directory}"
        )

    mask_paths = _required_mask_paths(segmentation_directory)
    geometry, geometry_mode = _load_study_geometry(
        study_directory,
        mask_paths["C1"],
    )
    processed_masks = {
        level: load_and_process_vertebra_mask(
            mask_paths[level],
            dicom_geometry=geometry,
        )
        for level in VERTEBRA_LEVELS
    }
    orientations = {
        level: find_best_physical_orientation(
            processed_masks[level].mask,
            processed_masks[level].affine_ras,
            base_row_basis_lps=geometry.row_direction,
            base_column_basis_lps=geometry.column_direction,
        )
        for level in VERTEBRA_LEVELS
    }
    bbox_centers = (
        load_study_bbox_centers(bbox_csv_path, study_id, geometry)
        if bbox_csv_path is not None
        else ()
    )
    bbox_assignments = assign_bbox_centers_to_vertebrae(
        bbox_centers,
        processed_masks,
        orientations,
        unique_levels=load_study_fracture_levels(
            TRAIN_LABEL_CSV,
            study_id,
        ),
    )
    classifier_planes = {
        level: build_classifier_plane_plan(
            processed_masks[level],
            orientations[level],
            bbox_centers=bbox_assignments[level],
        )
        for level in VERTEBRA_LEVELS
    }
    hu_volume = load_hu_volume(geometry)
    sampled_classifier_planes = {
        level: sample_classifier_planes(
            hu_volume,
            geometry,
            processed_masks[level],
            classifier_planes[level],
        )
        for level in VERTEBRA_LEVELS
    }
    sampled_segmentation_planes = {
        level: sample_segmentation_planes(
            hu_volume,
            geometry,
            processed_masks[level],
            orientations[level],
        )
        for level in VERTEBRA_LEVELS
    }
    metadata = _study_metadata(
        study_id=study_id,
        study_directory=study_directory,
        segmentation_directory=segmentation_directory,
        geometry=geometry,
        processed_masks=processed_masks,
        orientations=orientations,
        classifier_planes=classifier_planes,
        sampled_classifier_planes=sampled_classifier_planes,
        sampled_segmentation_planes=sampled_segmentation_planes,
        bbox_interval_count=len(bbox_centers),
        bbox_csv_path=bbox_csv_path,
        output_directory=output_directory,
        geometry_mode=geometry_mode,
    )
    write_study_classifier_outputs_atomic(
        output_directory,
        {k: v for k, v in sampled_classifier_planes.items() if k not in excluded_levels},
    )
    write_study_segmentation_outputs_atomic(
        output_directory,
        {k: v for k, v in sampled_segmentation_planes.items() if k not in excluded_levels},
    )
    _write_json_atomic(metadata_path, metadata)
    return StudyProcessingResult(
        study_id=study_id,
        geometry=geometry,
        vertebrae=processed_masks,
        orientations=orientations,
        classifier_planes=classifier_planes,
        sampled_classifier_planes=sampled_classifier_planes,
        sampled_segmentation_planes=sampled_segmentation_planes,
        metadata_path=metadata_path,
        output_directory=output_directory,
    )


def _required_mask_paths(
    segmentation_directory: Path,
) -> dict[str, Path]:
    mask_paths = {
        level: segmentation_directory / f"vertebrae_{level}.nii.gz"
        for level in VERTEBRA_LEVELS
    }
    missing_levels = [level for level, path in mask_paths.items() if not path.is_file()]
    if missing_levels:
        missing_text = ", ".join(missing_levels)
        raise StudyProcessingError(f"Missing vertebra masks: {missing_text}")
    return mask_paths


def _study_metadata(
    *,
    study_id: str,
    study_directory: Path,
    segmentation_directory: Path,
    geometry: DicomSeriesGeometry,
    processed_masks: dict[str, ProcessedVertebraMask],
    orientations: dict[str, OrientationSearchResult],
    classifier_planes: dict[str, ClassifierPlanePlan],
    sampled_classifier_planes: dict[str, SampledClassifierPlanes],
    sampled_segmentation_planes: dict[str, SampledSegmentationPlanes],
    bbox_interval_count: int,
    bbox_csv_path: Path | None,
    output_directory: Path,
    geometry_mode: str,
) -> dict[str, object]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "study_id": study_id,
        "status": "complete",
        "source": {
            "dicom_directory": str(study_directory),
            "segmentation_directory": str(segmentation_directory),
            "bbox_csv": str(bbox_csv_path) if bbox_csv_path is not None else None,
        },
        "dicom_geometry": _geometry_metadata(geometry),
        "vertebrae": {
            level: _vertebra_metadata(
                segmentation_directory / f"vertebrae_{level}.nii.gz",
                processed_masks[level],
                orientations[level],
                classifier_planes[level],
                sampled_classifier_planes[level],
                sampled_segmentation_planes[level],
                output_directory / level,
            )
            for level in VERTEBRA_LEVELS
        },
        "qc": {
            "all_vertebrae_present": True,
            "all_masks_non_empty": True,
            "all_masks_inside_dicom_volume": True,
            "orientation_search_boundary_count": sum(
                orientation.at_search_boundary for orientation in orientations.values()
            ),
            "bbox_interval_count": bbox_interval_count,
            "bbox_forced_plane_count": sum(
                plane.bbox_forced
                for plan in classifier_planes.values()
                for plane in plan.planes
            ),
            "all_classifier_masks_non_empty": all(
                sampled.qc.all_masks_non_empty
                for sampled in sampled_classifier_planes.values()
            ),
            "all_classifier_masks_inside_fov": all(
                sampled.qc.all_masks_inside_fov
                for sampled in sampled_classifier_planes.values()
            ),
            "irregular_slice_spacing": geometry.has_irregular_slice_spacing(),
            "geometry_mode": geometry_mode,
        },
    }


def _load_study_geometry(
    study_directory: Path,
    reference_mask_path: Path,
) -> tuple[DicomSeriesGeometry, str]:
    try:
        return load_dicom_series(study_directory), "native_dicom"
    except GeometryValidationError as error:
        if "Inconsistent image orientation" not in str(error):
            raise
        geometry = load_approximate_dicom_series_from_nifti(
            study_directory,
            reference_mask_path,
        )
        return geometry, "repaired_nifti_affine"


def _geometry_metadata(
    geometry: DicomSeriesGeometry,
) -> dict[str, object]:
    return {
        "series_instance_uid": geometry.series_instance_uid,
        "shape_slice_row_column": list(geometry.shape),
        "row_direction_lps": list(geometry.row_direction),
        "column_direction_lps": list(geometry.column_direction),
        "slice_normal_lps": list(geometry.slice_normal),
        "pixel_spacing_row_column_mm": list(geometry.pixel_spacing),
        "median_slice_spacing_mm": geometry.slice_spacing_mm,
        "spacing_deviations_mm": list(geometry.spacing_deviations_mm),
        "slices": [
            {
                "source_file": slice_metadata.path.name,
                "sop_instance_uid": slice_metadata.sop_instance_uid,
                "instance_number": slice_metadata.instance_number,
                "image_position_lps_mm": list(slice_metadata.image_position),
                "slice_position_mm": slice_metadata.slice_position_mm,
            }
            for slice_metadata in geometry.slices
        ],
    }


def _vertebra_metadata(
    mask_path: Path,
    processed_mask: ProcessedVertebraMask,
    orientation: OrientationSearchResult,
    classifier_plane_plan: ClassifierPlanePlan,
    sampled_classifier_planes: SampledClassifierPlanes,
    sampled_segmentation_planes: SampledSegmentationPlanes,
    output_directory: Path,
) -> dict[str, object]:
    alignment = processed_mask.dicom_alignment
    if alignment is None:
        raise StudyProcessingError("DICOM alignment metadata is required")
    return {
        "mask_source": str(mask_path),
        "mask_affine_ras": processed_mask.affine_ras.tolist(),
        "voxel_count": processed_mask.voxel_count,
        "component_count": processed_mask.component_count,
        "retained_fraction": processed_mask.retained_fraction,
        "volume_mm3": processed_mask.volume_mm3,
        "centroid_lps_mm": list(processed_mask.centroid_lps_mm),
        "bbox_min_lps_mm": list(processed_mask.bbox_min_lps_mm),
        "bbox_max_lps_mm": list(processed_mask.bbox_max_lps_mm),
        "superior_inferior_range_mm": list(processed_mask.superior_inferior_range_mm),
        "dicom_alignment": asdict(alignment),
        "orientation": asdict(orientation),
        "classifier_planes": {
            "robust_range_mm": list(classifier_plane_plan.robust_range_mm),
            "full_range_mm": list(classifier_plane_plan.full_range_mm),
            "sequence_order": "head_to_tail",
            "planes": [
                {
                    **asdict(plane),
                    "sequence_index": index,
                }
                for index, plane in enumerate(classifier_plane_plan.planes)
            ],
        },
        "sampling": sampling_metadata(
            sampled_classifier_planes,
            output_directory,
        ),
        "segmentation_sampling": segmentation_sampling_metadata(
            sampled_segmentation_planes,
            output_directory,
        ),
        "qc": {
            "non_empty": processed_mask.voxel_count > 0,
            "inside_dicom_volume": alignment.outside_fraction == 0.0,
            "removed_component_fraction": 1.0 - processed_mask.retained_fraction,
            "orientation_at_search_boundary": orientation.at_search_boundary,
            "classifier_plane_count": len(classifier_plane_plan.planes),
            "bbox_forced_plane_count": sum(
                plane.bbox_forced for plane in classifier_plane_plan.planes
            ),
            "all_classifier_masks_non_empty": (
                sampled_classifier_planes.qc.all_masks_non_empty
            ),
            "all_classifier_masks_inside_fov": (
                sampled_classifier_planes.qc.all_masks_inside_fov
            ),
        },
    }


def _write_json_atomic(
    output_path: Path,
    payload: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Process one RSNA study from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study_id")
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=TRAIN_IMAGES_DIR,
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=SEGMENTATION_DIR,
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=STUDY_METADATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FRACTURE_DATASET_DIR,
    )
    parser.add_argument(
        "--bbox-csv",
        type=Path,
        default=BOUNDING_BOX_CSV,
    )
    arguments = parser.parse_args()

    result = process_study(
        arguments.train_images_dir / arguments.study_id,
        arguments.segmentation_dir / arguments.study_id,
        arguments.metadata_dir / f"{arguments.study_id}.json",
        arguments.output_dir / arguments.study_id,
        bbox_csv_path=arguments.bbox_csv,
    )
    print(result.metadata_path)


if __name__ == "__main__":
    main()
