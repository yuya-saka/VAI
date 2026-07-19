"""Build bbox-centred corrected CT views from canonical 3D bbox geometry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from data_preprocessing.rsna_pipeline.bbox_geometry import (
    BboxCenteredPlaneSelection,
    BboxGeometryError,
    BboxRun,
    UnsupportedBboxGeometryModeError,
    contours_to_json,
    load_metadata,
    load_study_bbox_runs,
    render_bbox_run,
    select_bbox_centered_planes,
)
from data_preprocessing.rsna_pipeline.classifier_plane_sampling import (
    CHANNEL_INDEX_OFFSETS,
    EXPECTED_CHANNEL_COUNT,
    WINDOW_LOW_HU,
    apply_bone_window,
    load_hu_volume,
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
from data_preprocessing.rsna_pipeline.plane_sampling import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PIXEL_SPACING_MM,
    PhysicalPlane,
    sample_nifti_physical_planes,
    sample_physical_planes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
SEGMENTATION_DIR = DATA_DIR / "segmentations"
METADATA_DIR = DATA_DIR / "processing_metadata"
BBOX_CSV = DATA_DIR / "train_bounding_boxes.csv"
OUTPUT_DIR = DATA_DIR / "bbox_centered_dataset"

PIPELINE_VERSION: Final = "bbox-centered-v1"
EXPECTED_PLANE_COUNT: Final = 15

Uint8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class SampledBboxCenteredView:
    """CT, vertebra mask, and corrected bbox occupancy for one bbox run."""

    ct: Uint8Array
    vertebra_mask: Uint8Array
    bbox_occupancy: Uint8Array
    bbox_contours: list[dict[str, object]]
    channel_offsets_mm: tuple[float, ...]
    mask_pixel_counts: tuple[int, ...]
    bbox_area_by_plane: tuple[float, ...]


def sample_bbox_centered_view(
    hu_volume: npt.NDArray[np.float32],
    geometry: DicomSeriesGeometry,
    processed_mask: ProcessedVertebraMask,
    run: BboxRun,
    selection: BboxCenteredPlaneSelection,
) -> SampledBboxCenteredView:
    """Sample every bbox-centred modality from one shared physical-plane list."""
    if len(selection.planes) != EXPECTED_PLANE_COUNT:
        raise BboxGeometryError(f"Expected {EXPECTED_PLANE_COUNT} bbox-centred planes")
    channel_offsets = tuple(
        float(offset) * geometry.slice_spacing_mm for offset in CHANNEL_INDEX_OFFSETS
    )
    channel_planes = tuple(
        _shifted_plane(plane, offset)
        for plane in selection.planes
        for offset in channel_offsets
    )
    sampled_hu = sample_physical_planes(
        hu_volume,
        geometry,
        channel_planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=1,
        cval=WINDOW_LOW_HU,
    )
    sampled_mask = sample_nifti_physical_planes(
        processed_mask.mask,
        processed_mask.affine_ras,
        selection.planes,
        output_size=DEFAULT_OUTPUT_SIZE,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        interpolation_order=0,
        cval=0.0,
    )
    ct = apply_bone_window(sampled_hu).reshape(
        EXPECTED_PLANE_COUNT,
        EXPECTED_CHANNEL_COUNT,
        *DEFAULT_OUTPUT_SIZE,
    )
    vertebra_mask = np.asarray(sampled_mask > 0, dtype=np.uint8)
    bbox_occupancy, contours = render_bbox_run(run, selection.planes)
    if bbox_occupancy.shape != (EXPECTED_PLANE_COUNT, *DEFAULT_OUTPUT_SIZE):
        raise BboxGeometryError("Corrected bbox occupancy has an invalid shape")
    if int(bbox_occupancy[EXPECTED_PLANE_COUNT // 2].sum()) <= 0:
        raise BboxGeometryError("Centre plane does not contain corrected bbox geometry")
    return SampledBboxCenteredView(
        ct=ct,
        vertebra_mask=vertebra_mask,
        bbox_occupancy=bbox_occupancy,
        bbox_contours=contours_to_json(contours),
        channel_offsets_mm=channel_offsets,
        mask_pixel_counts=tuple(
            int(value) for value in vertebra_mask.sum(axis=(1, 2)).tolist()
        ),
        bbox_area_by_plane=tuple(
            float(value)
            for value in (bbox_occupancy.astype(np.float64) / 255.0)
            .sum(axis=(1, 2))
            .tolist()
        ),
    )


def build_study(
    study_id: str,
    *,
    train_images_dir: Path,
    segmentation_dir: Path,
    metadata_dir: Path,
    bbox_csv_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    allow_repaired_geometry: bool = False,
) -> list[Path]:
    """Build all bbox-run-centred views for one study."""
    study_directory = train_images_dir / study_id
    study_segmentation_directory = segmentation_dir / study_id
    metadata_path = metadata_dir / f"{study_id}.json"
    if not study_directory.is_dir():
        raise FileNotFoundError(f"DICOM study not found: {study_directory}")
    if not study_segmentation_directory.is_dir():
        raise FileNotFoundError(
            f"Segmentation study not found: {study_segmentation_directory}"
        )
    metadata = load_metadata(metadata_path)
    geometry_mode = str(metadata["qc"]["geometry_mode"])
    if geometry_mode != "native_dicom" and not allow_repaired_geometry:
        raise UnsupportedBboxGeometryModeError(
            f"{study_id}: exact bbox overlay requires native_dicom geometry; "
            f"got {geometry_mode}"
        )
    runs_by_level = load_study_bbox_runs(
        study_id,
        bbox_csv_path,
        study_directory,
        metadata,
    )
    if not runs_by_level:
        return []
    reference_mask_path = study_segmentation_directory / "vertebrae_C1.nii.gz"
    geometry = _load_sampling_geometry(study_directory, reference_mask_path)
    hu_volume = load_hu_volume(geometry)
    written: list[Path] = []
    for level, runs in sorted(runs_by_level.items()):
        mask_path = study_segmentation_directory / f"vertebrae_{level}.nii.gz"
        processed_mask = load_and_process_vertebra_mask(
            mask_path,
            dicom_geometry=geometry,
        )
        vertebra_metadata = metadata["vertebrae"][level]
        reference_plane, reference_position = _reference_plane(vertebra_metadata)
        robust_range = tuple(
            float(value)
            for value in vertebra_metadata["classifier_planes"]["robust_range_mm"]
        )
        for run_index, run in enumerate(runs):
            destination = output_dir / study_id / level / f"run_{run_index:02d}"
            if destination.is_dir() and not overwrite:
                written.append(destination)
                continue
            selection = select_bbox_centered_planes(
                run,
                reference_plane,
                reference_position,
                (robust_range[0], robust_range[1]),
            )
            sampled = sample_bbox_centered_view(
                hu_volume,
                geometry,
                processed_mask,
                run,
                selection,
            )
            payload = _view_metadata(
                study_id=study_id,
                level=level,
                run_index=run_index,
                run=run,
                selection=selection,
                sampled=sampled,
                source_metadata_path=metadata_path,
                geometry_mode=geometry_mode,
            )
            _write_view_atomic(destination, sampled, payload, overwrite=overwrite)
            written.append(destination)
    return written


def _load_sampling_geometry(
    study_directory: Path,
    reference_mask_path: Path,
) -> DicomSeriesGeometry:
    try:
        return load_dicom_series(study_directory)
    except GeometryValidationError as error:
        if "Inconsistent image orientation" not in str(error):
            raise
        return load_approximate_dicom_series_from_nifti(
            study_directory,
            reference_mask_path,
        )


def _reference_plane(vertebra_metadata: dict) -> tuple[PhysicalPlane, float]:
    planes = vertebra_metadata["classifier_planes"]["planes"]
    reference = next(
        (plane for plane in planes if bool(plane.get("max_area_forced"))),
        planes[len(planes) // 2],
    )
    return (
        PhysicalPlane(
            center=tuple(float(value) for value in reference["center_lps_mm"]),
            row_basis=tuple(float(value) for value in reference["row_basis_lps"]),
            column_basis=tuple(float(value) for value in reference["column_basis_lps"]),
        ),
        float(reference["position_mm"]),
    )


def _shifted_plane(plane: PhysicalPlane, offset_mm: float) -> PhysicalPlane:
    row_basis = np.asarray(plane.row_basis, dtype=np.float64)
    column_basis = np.asarray(plane.column_basis, dtype=np.float64)
    normal = np.cross(row_basis, column_basis)
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(plane.center, dtype=np.float64) + offset_mm * normal
    return PhysicalPlane(
        center=tuple(float(value) for value in center),
        row_basis=plane.row_basis,
        column_basis=plane.column_basis,
    )


def _view_metadata(
    *,
    study_id: str,
    level: str,
    run_index: int,
    run: BboxRun,
    selection: BboxCenteredPlaneSelection,
    sampled: SampledBboxCenteredView,
    source_metadata_path: Path,
    geometry_mode: str,
) -> dict[str, object]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "sampling_mode": "bbox_centered",
        "study_id": study_id,
        "level": level,
        "bbox_run_id": run_index,
        "source_bbox_slice_numbers": list(run.slice_numbers),
        "source_processing_metadata": str(source_metadata_path),
        "geometry_mode": geometry_mode,
        "bbox_support_range_mm": list(selection.bbox_support_range_mm),
        "bbox_center_position_mm": selection.bbox_center_position_mm,
        "plane_spacing_mm": selection.plane_spacing_mm,
        "center_plane_index": EXPECTED_PLANE_COUNT // 2,
        "channel_offsets_mm": list(sampled.channel_offsets_mm),
        "bbox_occupancy_area_by_plane": list(sampled.bbox_area_by_plane),
        "vertebra_mask_pixel_counts": list(sampled.mask_pixel_counts),
        "planes": [
            {
                "sequence_index": index,
                "position_mm": selection.positions_mm[index],
                "center_lps_mm": list(plane.center),
                "row_basis_lps": list(plane.row_basis),
                "column_basis_lps": list(plane.column_basis),
                "normal_lps": list(
                    np.cross(
                        np.asarray(plane.row_basis, dtype=np.float64),
                        np.asarray(plane.column_basis, dtype=np.float64),
                    )
                ),
            }
            for index, plane in enumerate(selection.planes)
        ],
    }


def _write_view_atomic(
    destination: Path,
    sampled: SampledBboxCenteredView,
    metadata: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    backup = destination.with_name(f".{destination.name}.backup")
    try:
        np.save(staging / "ct.npy", sampled.ct)
        np.save(staging / "vertebra_mask.npy", sampled.vertebra_mask)
        np.save(staging / "bbox_corrected_occupancy.npy", sampled.bbox_occupancy)
        (staging / "bbox_corrected_contours.json").write_text(
            json.dumps(sampled.bbox_contours, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {destination}")
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _study_ids(bbox_csv_path: Path) -> list[str]:
    with bbox_csv_path.open(newline="", encoding="utf-8") as file:
        return sorted({row["StudyInstanceUID"] for row in csv.DictReader(file)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bbox-centred corrected CT views"
    )
    parser.add_argument("--study-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-images-dir", type=Path, default=TRAIN_IMAGES_DIR)
    parser.add_argument("--segmentation-dir", type=Path, default=SEGMENTATION_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--bbox-csv", type=Path, default=BBOX_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-repaired-geometry", action="store_true")
    arguments = parser.parse_args()
    study_ids = arguments.study_id or _study_ids(arguments.bbox_csv)
    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("Limit must be positive")
        study_ids = study_ids[: arguments.limit]
    total = 0
    skipped = 0
    failed = 0
    for index, study_id in enumerate(study_ids, start=1):
        try:
            written = build_study(
                study_id,
                train_images_dir=arguments.train_images_dir,
                segmentation_dir=arguments.segmentation_dir,
                metadata_dir=arguments.metadata_dir,
                bbox_csv_path=arguments.bbox_csv,
                output_dir=arguments.output_dir,
                overwrite=arguments.overwrite,
                allow_repaired_geometry=arguments.allow_repaired_geometry,
            )
        except UnsupportedBboxGeometryModeError as error:
            skipped += 1
            print(f"[{index}/{len(study_ids)}] [SKIP] {error}")
            continue
        except Exception:
            failed += 1
            print(
                f"[{index}/{len(study_ids)}] [ERROR] {study_id}:\n"
                f"{traceback.format_exc()}"
            )
            continue
        total += len(written)
        print(f"[{index}/{len(study_ids)}] {study_id}: {len(written)} views")
    print(
        f"bbox-centred views: {total}; skipped studies: {skipped}; "
        f"failed studies: {failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
