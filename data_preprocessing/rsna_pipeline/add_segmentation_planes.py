"""Add seg_ct.npy and seg_vertebra_mask.npy using existing classifier outputs.

CT channels are extracted directly from the existing ct.npy (no DICOM loading).
Vertebra masks are sampled from NIfTI files at the same plane positions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Final

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from data_preprocessing.rsna_pipeline.plane_sampling import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PIXEL_SPACING_MM,
    PhysicalPlane,
    sample_nifti_physical_planes,
)
from data_preprocessing.rsna_pipeline.process_dataset import (
    DatasetStudyResult,
    load_excluded_levels_by_study,
    load_excluded_studies,
)
from data_preprocessing.rsna_pipeline.process_study import (
    FRACTURE_DATASET_DIR,
    PIPELINE_VERSION,
    RSNA_DATA_DIR,
    SEGMENTATION_DIR,
    STUDY_METADATA_DIR,
    VERTEBRA_LEVELS,
)
from data_preprocessing.rsna_pipeline.segmentation_plane_sampling import (
    SEG_SLICE_INDEX_OFFSETS,
    write_npy_atomic,
)

EXCLUDED_STUDIES_CSV: Final = RSNA_DATA_DIR / "excluded_studies.csv"
EXCLUDED_LEVELS_CSV: Final = RSNA_DATA_DIR / "excluded_levels.csv"
EXPECTED_SEG_SHAPE: Final = (5, 224, 224)
DEFAULT_WORKERS: Final = 8


def is_complete_seg_output(
    study_id: str,
    metadata_dir: Path,
    output_dir: Path,
    excluded_levels: frozenset[str] = frozenset(),
) -> bool:
    """Return True if all non-excluded levels already have valid seg files."""
    metadata_path = metadata_dir / f"{study_id}.json"
    if not metadata_path.is_file():
        return False
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if meta.get("status") != "complete":
        return False
    for level in VERTEBRA_LEVELS:
        if level in excluded_levels:
            continue
        level_dir = output_dir / study_id / level
        if not _valid_seg_array(level_dir / "seg_ct.npy"):
            return False
        if not _valid_seg_array(level_dir / "seg_vertebra_mask.npy"):
            return False
    return True


def add_segmentation_planes_dataset(
    *,
    metadata_dir: Path,
    output_dir: Path,
    segmentation_dir: Path,
    workers: int,
    log_path: Path,
    excluded_studies_csv: Path = EXCLUDED_STUDIES_CSV,
    excluded_levels_csv: Path = EXCLUDED_LEVELS_CSV,
    overwrite: bool = False,
) -> tuple[DatasetStudyResult, ...]:
    """Add seg files to all complete studies in parallel."""
    if workers <= 0:
        raise ValueError("Workers must be positive")

    excluded_studies = load_excluded_studies(excluded_studies_csv)
    excluded_levels_by_study = load_excluded_levels_by_study(excluded_levels_csv)

    candidates = sorted(
        p.name
        for p in output_dir.iterdir()
        if p.is_dir() and p.name not in excluded_studies
    )
    pending = tuple(
        study_id
        for study_id in candidates
        if overwrite
        or not is_complete_seg_output(
            study_id,
            metadata_dir,
            output_dir,
            excluded_levels_by_study.get(study_id, frozenset()),
        )
    )
    skipped_count = len(candidates) - len(pending)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_log(log_path, {
        "event": "start",
        "candidates": len(candidates),
        "pending": len(pending),
        "skipped": skipped_count,
        "workers": workers,
    })
    if not pending:
        return ()

    results: list[DatasetStudyResult] = []
    started_at = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures: dict[Future[DatasetStudyResult], str] = {
            executor.submit(
                _add_one_study,
                study_id,
                metadata_dir,
                output_dir,
                segmentation_dir,
                excluded_levels_by_study.get(study_id, frozenset()),
            ): study_id
            for study_id in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            elapsed = time.monotonic() - started_at
            rate = completed_count / elapsed if elapsed > 0.0 else 0.0
            remaining = len(pending) - completed_count
            eta_seconds = remaining / rate if rate > 0.0 else None
            _append_log(log_path, {
                "event": "study_complete",
                **asdict(result),
                "completed": completed_count,
                "total": len(pending),
                "eta_seconds": eta_seconds,
            })
            print(
                f"[{completed_count}/{len(pending)}] {result.study_id} "
                f"{result.status} {result.elapsed_seconds:.1f}s "
                f"ETA {_format_duration(eta_seconds)}",
                flush=True,
            )
    return tuple(results)


def _add_one_study(
    study_id: str,
    metadata_dir: Path,
    output_dir: Path,
    segmentation_dir: Path,
    excluded_levels: frozenset[str],
) -> DatasetStudyResult:
    started_at = time.monotonic()
    try:
        _add_seg_planes_study(
            study_id,
            metadata_dir,
            output_dir,
            segmentation_dir,
            excluded_levels,
        )
    except Exception as error:
        return DatasetStudyResult(
            study_id=study_id,
            status="failed",
            elapsed_seconds=time.monotonic() - started_at,
            error=f"{type(error).__name__}: {error}",
            traceback_text=traceback.format_exc(),
        )
    return DatasetStudyResult(
        study_id=study_id,
        status="complete",
        elapsed_seconds=time.monotonic() - started_at,
    )


def _add_seg_planes_study(
    study_id: str,
    metadata_dir: Path,
    output_dir: Path,
    segmentation_dir: Path,
    excluded_levels: frozenset[str],
) -> None:
    meta = json.loads((metadata_dir / f"{study_id}.json").read_text(encoding="utf-8"))
    slice_spacing_mm: float = meta["dicom_geometry"]["median_slice_spacing_mm"]

    for level in VERTEBRA_LEVELS:
        if level in excluded_levels:
            continue

        level_dir = output_dir / study_id / level
        seg_ct_path = level_dir / "seg_ct.npy"
        seg_mask_path = level_dir / "seg_vertebra_mask.npy"
        if _valid_seg_array(seg_ct_path) and _valid_seg_array(seg_mask_path):
            continue

        # CT: extract channels of the max_area_forced classifier plane
        ct = np.load(level_dir / "ct.npy")  # (15, 5, 224, 224)
        planes_meta = meta["vertebrae"][level]["classifier_planes"]["planes"]
        max_area_idx = next(
            i for i, p in enumerate(planes_meta) if p["max_area_forced"]
        )
        seg_ct = ct[max_area_idx]  # (5, 224, 224)

        # Mask: sample 5 NIfTI planes at the same center as the classifier plane
        plane_meta = planes_meta[max_area_idx]
        center = np.asarray(plane_meta["center_lps_mm"], dtype=np.float64)
        normal = np.asarray(plane_meta["normal_lps"], dtype=np.float64)
        row_basis = tuple(float(v) for v in plane_meta["row_basis_lps"])
        col_basis = tuple(float(v) for v in plane_meta["column_basis_lps"])

        offsets_mm = tuple(
            float(idx) * slice_spacing_mm for idx in SEG_SLICE_INDEX_OFFSETS
        )
        seg_planes = tuple(
            PhysicalPlane(
                center=_vec3(center + offset_mm * normal),
                row_basis=row_basis,
                column_basis=col_basis,
            )
            for offset_mm in offsets_mm
        )

        mask_path = segmentation_dir / study_id / f"vertebrae_{level}.nii.gz"
        nifti_img = nib.load(mask_path)
        mask_vol = np.asarray(np.asarray(nifti_img.dataobj) > 0, dtype=np.uint8)
        affine_ras = np.asarray(nifti_img.affine, dtype=np.float64)

        sampled_mask = sample_nifti_physical_planes(
            mask_vol,
            affine_ras,
            seg_planes,
            output_size=DEFAULT_OUTPUT_SIZE,
            pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
            interpolation_order=0,
            cval=0.0,
        )
        seg_mask = np.asarray(sampled_mask > 0, dtype=np.uint8)

        write_npy_atomic(seg_ct_path, seg_ct)
        write_npy_atomic(seg_mask_path, seg_mask)


def _valid_seg_array(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return bool(array.shape == EXPECTED_SEG_SHAPE and array.dtype == np.uint8)


def _vec3(values: npt.ArrayLike) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    return (float(v[0]), float(v[1]), float(v[2]))


def _append_log(log_path: Path, payload: dict[str, object]) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add seg_ct.npy/seg_vertebra_mask.npy to existing study outputs."
    )
    parser.add_argument("--metadata-dir", type=Path, default=STUDY_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument("--segmentation-dir", type=Path, default=SEGMENTATION_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(".tmp/add_segmentation_planes.jsonl"),
    )
    parser.add_argument(
        "--excluded-studies-csv", type=Path, default=EXCLUDED_STUDIES_CSV
    )
    parser.add_argument(
        "--excluded-levels-csv", type=Path, default=EXCLUDED_LEVELS_CSV
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = add_segmentation_planes_dataset(
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        segmentation_dir=args.segmentation_dir,
        workers=args.workers,
        log_path=args.log_path,
        excluded_studies_csv=args.excluded_studies_csv,
        excluded_levels_csv=args.excluded_levels_csv,
        overwrite=args.overwrite,
    )
    failed = [r for r in results if r.status == "failed"]
    print(
        f"Finished: complete={len(results) - len(failed)}, failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
