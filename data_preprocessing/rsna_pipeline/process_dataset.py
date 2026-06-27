"""Parallel, resumable dataset runner for RSNA classifier-plane preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from data_preprocessing.rsna_pipeline.process_study import (
    BOUNDING_BOX_CSV,
    FRACTURE_DATASET_DIR,
    PIPELINE_VERSION,
    RSNA_DATA_DIR,
    SEGMENTATION_DIR,
    STUDY_METADATA_DIR,
    TRAIN_IMAGES_DIR,
    process_study,
)

DEFAULT_WORKERS: Final = 8
EXPECTED_CT_SHAPE: Final = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE: Final = (15, 224, 224)
EXPECTED_SEG_CT_SHAPE: Final = (5, 224, 224)
EXPECTED_SEG_MASK_SHAPE: Final = (5, 224, 224)
VERTEBRA_LEVELS: Final = tuple(f"C{level}" for level in range(1, 8))
EXCLUDED_STUDIES_CSV: Final = RSNA_DATA_DIR / "excluded_studies.csv"
EXCLUDED_LEVELS_CSV: Final = RSNA_DATA_DIR / "excluded_levels.csv"


@dataclass(frozen=True)
class DatasetStudyResult:
    """Serializable result for one study."""

    study_id: str
    status: str
    elapsed_seconds: float
    error: str | None = None
    traceback_text: str | None = None


def load_excluded_studies(csv_path: Path) -> frozenset[str]:
    """Return study UIDs listed in the excluded-studies CSV."""
    if not csv_path.is_file():
        return frozenset()
    with csv_path.open(newline="", encoding="utf-8") as file:
        return frozenset(row["study_uid"] for row in csv.DictReader(file))


def load_excluded_levels_by_study(csv_path: Path) -> dict[str, frozenset[str]]:
    """Return {study_uid: {vertebra, ...}} from the excluded-levels CSV."""
    if not csv_path.is_file():
        return {}
    result: dict[str, list[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            result.setdefault(row["study_uid"], []).append(row["vertebra"])
    return {uid: frozenset(levels) for uid, levels in result.items()}


def discover_eligible_studies(
    train_images_dir: Path,
    segmentation_dir: Path,
) -> tuple[str, ...]:
    """Return sorted studies having both DICOM input and complete segmentation."""
    if not train_images_dir.is_dir():
        raise FileNotFoundError(f"Train image directory not found: {train_images_dir}")
    if not segmentation_dir.is_dir():
        raise FileNotFoundError(f"Segmentation directory not found: {segmentation_dir}")

    study_ids: list[str] = []
    for study_directory in train_images_dir.iterdir():
        if not study_directory.is_dir():
            continue
        mask_directory = segmentation_dir / study_directory.name
        if _has_complete_segmentation(mask_directory):
            study_ids.append(study_directory.name)
    return tuple(sorted(study_ids))


def is_complete_study_output(
    study_id: str,
    metadata_dir: Path,
    output_dir: Path,
    excluded_levels: frozenset[str] = frozenset(),
) -> bool:
    """Return whether one study has a complete current-version output."""
    metadata_path = metadata_dir / f"{study_id}.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if metadata.get("status") != "complete":
        return False

    study_output = output_dir / study_id
    for level in VERTEBRA_LEVELS:
        if level in excluded_levels:
            continue
        level_directory = study_output / level
        if not _valid_array(level_directory / "ct.npy", EXPECTED_CT_SHAPE):
            return False
        if not _valid_array(
            level_directory / "vertebra_mask.npy",
            EXPECTED_MASK_SHAPE,
        ):
            return False
        if not (level_directory / "sampling_qc.json").is_file():
            return False
        if not _valid_array(level_directory / "seg_ct.npy", EXPECTED_SEG_CT_SHAPE):
            return False
        if not _valid_array(
            level_directory / "seg_vertebra_mask.npy",
            EXPECTED_SEG_MASK_SHAPE,
        ):
            return False
    return True


def process_dataset(
    *,
    train_images_dir: Path,
    segmentation_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    bbox_csv_path: Path | None,
    workers: int,
    log_path: Path,
    excluded_studies_csv: Path = EXCLUDED_STUDIES_CSV,
    excluded_levels_csv: Path = EXCLUDED_LEVELS_CSV,
    limit: int | None = None,
    overwrite: bool = False,
) -> tuple[DatasetStudyResult, ...]:
    """Process all eligible studies with bounded parallelism and resume support."""
    if workers <= 0:
        raise ValueError("Workers must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("Limit must be positive")

    excluded_studies = load_excluded_studies(excluded_studies_csv)
    excluded_levels_by_study = load_excluded_levels_by_study(excluded_levels_csv)

    eligible = discover_eligible_studies(train_images_dir, segmentation_dir)
    eligible = tuple(s for s in eligible if s not in excluded_studies)
    selected = eligible[:limit] if limit is not None else eligible
    pending = tuple(
        study_id
        for study_id in selected
        if overwrite
        or not is_complete_study_output(
            study_id,
            metadata_dir,
            output_dir,
            excluded_levels_by_study.get(study_id, frozenset()),
        )
    )
    skipped_count = len(selected) - len(pending)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_log(
        log_path,
        {
            "event": "start",
            "pipeline_version": PIPELINE_VERSION,
            "eligible": len(eligible),
            "selected": len(selected),
            "pending": len(pending),
            "skipped": skipped_count,
            "workers": workers,
        },
    )
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
                _process_one_study,
                study_id,
                train_images_dir,
                segmentation_dir,
                metadata_dir,
                output_dir,
                bbox_csv_path,
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
            _append_log(
                log_path,
                {
                    "event": "study_complete",
                    **asdict(result),
                    "completed": completed_count,
                    "total": len(pending),
                    "eta_seconds": eta_seconds,
                },
            )
            print(
                f"[{completed_count}/{len(pending)}] {result.study_id} "
                f"{result.status} {result.elapsed_seconds:.1f}s "
                f"ETA {_format_duration(eta_seconds)}",
                flush=True,
            )
    return tuple(results)


def _process_one_study(
    study_id: str,
    train_images_dir: Path,
    segmentation_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    bbox_csv_path: Path | None,
    excluded_levels: frozenset[str] = frozenset(),
) -> DatasetStudyResult:
    started_at = time.monotonic()
    try:
        process_study(
            train_images_dir / study_id,
            segmentation_dir / study_id,
            metadata_dir / f"{study_id}.json",
            output_dir / study_id,
            bbox_csv_path=bbox_csv_path,
            excluded_levels=excluded_levels,
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


def _has_complete_segmentation(segmentation_directory: Path) -> bool:
    return segmentation_directory.is_dir() and all(
        (segmentation_directory / f"vertebrae_{level}.nii.gz").is_file()
        for level in VERTEBRA_LEVELS
    )


def _valid_array(path: Path, expected_shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return bool(array.shape == expected_shape and array.dtype == np.uint8)


def _append_log(log_path: Path, payload: dict[str, object]) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"


def _set_thread_limits(thread_count: int) -> None:
    value = str(thread_count)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = value


def main() -> None:
    """Run the complete RSNA classifier-plane preprocessing dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images-dir", type=Path, default=TRAIN_IMAGES_DIR)
    parser.add_argument("--segmentation-dir", type=Path, default=SEGMENTATION_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=STUDY_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=FRACTURE_DATASET_DIR)
    parser.add_argument("--bbox-csv", type=Path, default=BOUNDING_BOX_CSV)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(".tmp/rsna_process_dataset.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--excluded-studies-csv",
        type=Path,
        default=EXCLUDED_STUDIES_CSV,
    )
    parser.add_argument(
        "--excluded-levels-csv",
        type=Path,
        default=EXCLUDED_LEVELS_CSV,
    )
    arguments = parser.parse_args()
    if arguments.threads_per_worker <= 0:
        raise ValueError("Threads per worker must be positive")
    _set_thread_limits(arguments.threads_per_worker)

    results = process_dataset(
        train_images_dir=arguments.train_images_dir,
        segmentation_dir=arguments.segmentation_dir,
        metadata_dir=arguments.metadata_dir,
        output_dir=arguments.output_dir,
        bbox_csv_path=arguments.bbox_csv,
        workers=arguments.workers,
        log_path=arguments.log_path,
        excluded_studies_csv=arguments.excluded_studies_csv,
        excluded_levels_csv=arguments.excluded_levels_csv,
        limit=arguments.limit,
        overwrite=arguments.overwrite,
    )
    failed = [result for result in results if result.status == "failed"]
    print(
        f"Finished: complete={len(results) - len(failed)}, failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
