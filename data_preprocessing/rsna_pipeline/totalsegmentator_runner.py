"""Resource-bounded TotalSegmentator runner for RSNA DICOM studies."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
TRAIN_IMAGES_DIR = RSNA_DATA_DIR / "train_images"
SEGMENTATION_OUTPUT_DIR = RSNA_DATA_DIR / "segmentations"

VERTEBRAE_ROI: Final = tuple(f"vertebrae_C{level}" for level in range(1, 8))
EXPECTED_MASK_NAMES: Final = tuple(f"{roi}.nii.gz" for roi in VERTEBRAE_ROI)
DEFAULT_WORKERS_PER_GPU: Final = 1
DEFAULT_CPU_THREAD_BUDGET: Final = 8
DEFAULT_CONVERSION_TIMEOUT_SECONDS: Final = 300

Segmenter = Callable[[Path, Path], None]
Converter = Callable[[Path, Path], None]
InputMode = Literal["auto", "dicom", "nifti"]


class SegmentationError(RuntimeError):
    """Raised when a study does not produce all expected vertebra masks."""


@dataclass(frozen=True)
class TotalSegmentatorSettings:
    """Runtime settings passed to one TotalSegmentator invocation."""

    threads: int = 1
    saving_threads: int = 1
    fast: bool = False
    quiet: bool = True


@dataclass(frozen=True)
class StudySegmentationResult:
    """Result of one study segmentation attempt."""

    study_id: str
    status: str
    input_mode: str | None
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class WorkerLayout:
    """GPU assignment and CPU-thread allocation."""

    gpu_assignments: tuple[int, ...]
    threads_per_worker: int


def is_complete_segmentation(output_directory: Path) -> bool:
    """Return whether all C1-C7 mask files exist and are non-empty."""
    return all(
        (output_directory / name).is_file()
        and (output_directory / name).stat().st_size > 0
        for name in EXPECTED_MASK_NAMES
    )


def calculate_worker_layout(
    *,
    pending_studies: int,
    gpu_ids: Sequence[int],
    workers_per_gpu: int,
    max_workers: int | None,
    cpu_thread_budget: int,
) -> WorkerLayout:
    """Calculate a bounded worker layout without oversubscribing CPU threads."""
    if pending_studies < 0:
        raise ValueError("Pending study count must not be negative")
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    if workers_per_gpu <= 0:
        raise ValueError("Workers per GPU must be positive")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("Max workers must be positive")
    if cpu_thread_budget <= 0:
        raise ValueError("CPU thread budget must be positive")
    if pending_studies == 0:
        return WorkerLayout(gpu_assignments=(), threads_per_worker=1)

    assignments = tuple(
        gpu_id
        for gpu_id in gpu_ids
        for _ in range(workers_per_gpu)
    )
    worker_limit = len(assignments)
    if max_workers is not None:
        worker_limit = min(worker_limit, max_workers)
    worker_count = min(pending_studies, worker_limit)
    selected_assignments = assignments[:worker_count]
    threads_per_worker = max(1, cpu_thread_budget // worker_count)
    return WorkerLayout(
        gpu_assignments=selected_assignments,
        threads_per_worker=threads_per_worker,
    )


def run_study_segmentation(
    study_directory: Path,
    output_directory: Path,
    *,
    scratch_root: Path,
    settings: TotalSegmentatorSettings | None = None,
    segmenter: Segmenter | None = None,
    converter: Converter | None = None,
    input_mode: InputMode = "auto",
) -> StudySegmentationResult:
    """Run DICOM-first segmentation with an explicit temporary-NIfTI fallback."""
    if not study_directory.is_dir():
        raise FileNotFoundError(f"Study directory not found: {study_directory}")
    if is_complete_segmentation(output_directory):
        return StudySegmentationResult(
            study_id=study_directory.name,
            status="skip",
            input_mode=None,
            elapsed_seconds=0.0,
        )

    runtime_settings = settings or TotalSegmentatorSettings()
    active_segmenter = segmenter or _segmenter_from_settings(runtime_settings)
    active_converter = converter or convert_dicom_to_temporary_nifti
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{study_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    direct_error: Exception | None = None
    try:
        if input_mode in {"auto", "dicom"}:
            try:
                active_segmenter(study_directory, staging_directory)
                _validate_segmentation(staging_directory)
                _commit_staging_directory(staging_directory, output_directory)
                return StudySegmentationResult(
                    study_id=study_directory.name,
                    status="ok",
                    input_mode="dicom",
                    elapsed_seconds=time.monotonic() - start_time,
                )
            except Exception as error:  # noqa: BLE001
                direct_error = error
                _clear_directory(staging_directory)
                if input_mode == "dicom":
                    raise

        with tempfile.TemporaryDirectory(
            prefix=f"rsna_{study_directory.name}_",
            dir=scratch_root,
        ) as temporary_directory:
            nifti_path = Path(temporary_directory) / "ct.nii.gz"
            active_converter(study_directory, nifti_path)
            if not nifti_path.is_file() or nifti_path.stat().st_size == 0:
                raise SegmentationError("Fallback conversion produced no NIfTI file")
            active_segmenter(nifti_path, staging_directory)
            _validate_segmentation(staging_directory)

        _commit_staging_directory(staging_directory, output_directory)
        return StudySegmentationResult(
            study_id=study_directory.name,
            status="ok",
            input_mode=(
                "nifti_fallback"
                if input_mode == "auto"
                else "nifti"
            ),
            elapsed_seconds=time.monotonic() - start_time,
            error=str(direct_error),
        )
    except Exception as fallback_error:
        direct_message = (
            f"Direct DICOM input failed: {direct_error}. "
            if direct_error is not None
            else ""
        )
        raise SegmentationError(
            f"{direct_message}Fallback failed: {fallback_error}"
        ) from fallback_error
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


def convert_dicom_to_temporary_nifti(
    study_directory: Path,
    output_path: Path,
) -> None:
    """Convert one validated DICOM series to a temporary NIfTI with dcm2niix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = _dcm2niix_command(study_directory, output_path)
    _run_dcm2niix(command, output_path)

    if output_path.is_file():
        return

    candidates = _conversion_candidates(output_path)
    if len(candidates) > 1:
        for candidate in candidates:
            candidate.unlink()
        merge_command = _dcm2niix_command(
            study_directory,
            output_path,
            force_merge=True,
        )
        _run_dcm2niix(merge_command, output_path)
        if output_path.is_file():
            _repair_missing_slice_axis(output_path, study_directory)
            return
        candidates = _conversion_candidates(output_path)

    if len(candidates) != 1:
        raise SegmentationError(
            f"dcm2niix produced {len(candidates)} candidate NIfTI files"
        )
    candidate = candidates[0]
    if candidate.suffix == ".gz":
        candidate.replace(output_path)
        return
    with candidate.open("rb") as source, gzip.open(output_path, "wb") as destination:
        shutil.copyfileobj(source, destination)
    candidate.unlink()


def _dcm2niix_command(
    study_directory: Path,
    output_path: Path,
    *,
    force_merge: bool = False,
) -> list[str]:
    command = [
        "dcm2niix",
        "-z",
        "y",
        "-f",
        output_path.name.removesuffix(".nii.gz"),
        "-b",
        "n",
        "-o",
        str(output_path.parent),
        str(study_directory),
    ]
    if force_merge:
        command[1:1] = ["-m", "y"]
    return command


def _run_dcm2niix(command: list[str], output_path: Path) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=DEFAULT_CONVERSION_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0 and not _conversion_candidates(output_path):
        error_message = result.stderr.strip() or result.stdout.strip()
        raise SegmentationError(
            f"dcm2niix failed with code {result.returncode}: "
            f"{error_message[-500:]}"
        )


def _conversion_candidates(output_path: Path) -> list[Path]:
    output_prefix = output_path.name.removesuffix(".nii.gz")
    candidates = sorted(output_path.parent.glob(f"{output_prefix}*.nii.gz"))
    candidates.extend(sorted(output_path.parent.glob(f"{output_prefix}*.nii")))
    return candidates


def _repair_missing_slice_axis(
    nifti_path: Path,
    study_directory: Path,
) -> None:
    import nibabel as nib
    import numpy as np
    import pydicom

    image: Any = nib.load(nifti_path)
    affine = np.asarray(image.affine, dtype=np.float64)
    if np.linalg.norm(affine[:3, 2]) > 1e-6:
        return

    positions = []
    for dicom_path in study_directory.iterdir():
        if not dicom_path.is_file():
            continue
        dataset = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient"],
        )
        positions.append(
            np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        )
    if len(positions) != image.shape[2]:
        raise SegmentationError(
            "Cannot repair slice axis: DICOM and NIfTI slice counts differ"
        )

    position_array = np.stack(positions)
    centered = position_array - position_array.mean(axis=0)
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    principal_axis = principal_axes[0]
    projections = position_array @ principal_axis
    endpoint_indices = (int(np.argmin(projections)), int(np.argmax(projections)))
    endpoints = position_array[list(endpoint_indices)]
    endpoint_ras = endpoints * np.array([-1.0, -1.0, 1.0])
    start_index = int(
        np.argmin(np.linalg.norm(endpoint_ras - affine[:3, 3], axis=1))
    )
    end_index = 1 - start_index
    slice_vector = (
        endpoint_ras[end_index] - endpoint_ras[start_index]
    ) / (image.shape[2] - 1)
    if np.linalg.norm(slice_vector) <= 1e-6:
        raise SegmentationError("Cannot repair zero-length slice axis")

    repaired_affine = affine.copy()
    repaired_affine[:3, 2] = slice_vector
    repaired_image = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        np.asanyarray(image.dataobj),
        repaired_affine,
        image.header,
    )
    repaired_image.set_qform(repaired_affine, code=1)  # type: ignore[no-untyped-call]
    repaired_image.set_sform(repaired_affine, code=1)  # type: ignore[no-untyped-call]
    nib.save(repaired_image, nifti_path)


def run_totalsegmentator(
    input_path: Path,
    output_directory: Path,
    settings: TotalSegmentatorSettings,
) -> None:
    """Invoke TotalSegmentator after worker resource limits are configured."""
    from totalsegmentator.python_api import (  # type: ignore[import-untyped]
        totalsegmentator,
    )

    totalsegmentator(
        input_path,
        output_directory,
        roi_subset=list(VERTEBRAE_ROI),
        fast=settings.fast,
        quiet=settings.quiet,
        nr_thr_resamp=settings.threads,
        nr_thr_saving=settings.saving_threads,
        device="gpu",
    )


def _segmenter_from_settings(settings: TotalSegmentatorSettings) -> Segmenter:
    def segmenter(input_path: Path, output_directory: Path) -> None:
        run_totalsegmentator(input_path, output_directory, settings)

    return segmenter


def _validate_segmentation(output_directory: Path) -> None:
    missing = [
        name
        for name in EXPECTED_MASK_NAMES
        if not (output_directory / name).is_file()
        or (output_directory / name).stat().st_size == 0
    ]
    if missing:
        raise SegmentationError(f"Missing masks: {', '.join(missing)}")


def _clear_directory(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _commit_staging_directory(
    staging_directory: Path,
    output_directory: Path,
) -> None:
    if output_directory.exists():
        shutil.rmtree(output_directory)
    os.replace(staging_directory, output_directory)


def _configure_worker_environment(
    *,
    gpu_id: int,
    threads: int,
    scratch_root: Path,
) -> None:
    scratch_root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TMPDIR"] = str(scratch_root)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    tempfile.tempdir = str(scratch_root)


def _worker_loop(
    worker_id: int,
    gpu_id: int,
    study_directories: Sequence[Path],
    output_root: Path,
    scratch_root: Path,
    settings: TotalSegmentatorSettings,
    result_path: Path,
    input_mode: InputMode,
) -> None:
    _configure_worker_environment(
        gpu_id=gpu_id,
        threads=settings.threads,
        scratch_root=scratch_root,
    )

    import torch

    torch.set_num_threads(settings.threads)
    results: list[dict[str, object]] = []
    for index, study_directory in enumerate(study_directories, start=1):
        try:
            result = run_study_segmentation(
                study_directory,
                output_root / study_directory.name,
                scratch_root=scratch_root,
                settings=settings,
                input_mode=input_mode,
            )
            results.append(asdict(result))
            print(
                f"[w{worker_id}] ({index}/{len(study_directories)}) "
                f"{study_directory.name} {result.status} "
                f"{result.input_mode or '-'} {result.elapsed_seconds:.1f}s",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            results.append(
                asdict(
                    StudySegmentationResult(
                        study_id=study_directory.name,
                        status="fail",
                        input_mode=None,
                        elapsed_seconds=0.0,
                        error=str(error),
                    )
                )
            )
            print(
                f"[w{worker_id}] ({index}/{len(study_directories)}) "
                f"{study_directory.name} ERROR: {error}",
                flush=True,
            )
            if os.environ.get("DEBUG") == "1":
                traceback.print_exc()

    result_path.write_text(json.dumps(results), encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSNA DICOMをTotalSegmentatorでC1-C7 segmentation",
    )
    parser.add_argument("--input-dir", type=Path, default=TRAIN_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=SEGMENTATION_OUTPUT_DIR)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--cpu-thread-budget",
        type=int,
        default=min(DEFAULT_CPU_THREAD_BUDGET, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "RSNA_SCRATCH_DIR",
                "/tmp/rsna_totalsegmentator",
            )
        ),
    )
    parser.add_argument(
        "--study-id",
        action="append",
        default=None,
        help="処理するStudyInstanceUID。複数回指定可能",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--input-mode",
        choices=("nifti", "auto", "dicom"),
        default="nifti",
        help="nifti: dcm2niix一時変換、auto: DICOM失敗時fallback",
    )
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study_directories = sorted(
        path for path in args.input_dir.iterdir() if path.is_dir()
    )
    if args.study_id:
        requested_ids = set(args.study_id)
        study_directories = [
            path for path in study_directories if path.name in requested_ids
        ]
        found_ids = {path.name for path in study_directories}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            missing_text = ", ".join(sorted(missing_ids))
            raise FileNotFoundError(f"Study directory not found: {missing_text}")
    if args.limit is not None:
        study_directories = study_directories[: args.limit]

    pending = [
        path
        for path in study_directories
        if not is_complete_segmentation(args.output_dir / path.name)
    ]
    skipped = len(study_directories) - len(pending)
    layout = calculate_worker_layout(
        pending_studies=len(pending),
        gpu_ids=tuple(args.gpus),
        workers_per_gpu=args.workers_per_gpu,
        max_workers=args.max_workers,
        cpu_thread_budget=args.cpu_thread_budget,
    )

    print(f"対象: {len(study_directories)}, skip: {skipped}, pending: {len(pending)}")
    print(
        f"GPU割当: {layout.gpu_assignments}, "
        f"threads/worker: {layout.threads_per_worker}, "
        f"scratch: {args.scratch_dir}"
    )
    if not pending:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    worker_count = len(layout.gpu_assignments)
    shards: list[list[Path]] = [[] for _ in range(worker_count)]
    for index, study_directory in enumerate(pending):
        shards[index % worker_count].append(study_directory)

    settings = TotalSegmentatorSettings(
        threads=layout.threads_per_worker,
        saving_threads=min(2, layout.threads_per_worker),
        fast=args.fast,
    )
    start_time = time.monotonic()
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="rsna_seg_results_",
        dir=args.scratch_dir,
    ) as result_directory_string:
        result_directory = Path(result_directory_string)
        result_paths = [
            result_directory / f"worker_{worker_id}.json"
            for worker_id in range(worker_count)
        ]
        processes = [
            context.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    layout.gpu_assignments[worker_id],
                    shards[worker_id],
                    args.output_dir,
                    args.scratch_dir,
                    settings,
                    result_paths[worker_id],
                    args.input_mode,
                ),
                daemon=False,
            )
            for worker_id in range(worker_count)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        results: list[dict[str, object]] = []
        for result_path in result_paths:
            if result_path.is_file():
                results.extend(json.loads(result_path.read_text(encoding="utf-8")))
            else:
                print(f"[WARN] worker result missing: {result_path.name}")

    success_count = sum(result["status"] == "ok" for result in results)
    failure_count = sum(result["status"] == "fail" for result in results)
    elapsed_minutes = (time.monotonic() - start_time) / 60.0
    print(
        f"完了: success={success_count}, skip={skipped}, "
        f"fail={failure_count}, elapsed={elapsed_minutes:.1f}分"
    )
    for result in results:
        if result["status"] == "fail":
            print(f"  {result['study_id']}: {result['error']}")


if __name__ == "__main__":
    main()
