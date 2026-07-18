"""Stage the Stage3 dataset onto fast local storage for the duration of a run."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STAGE_FILE_NAMES = ("ct.npy", "vertebra_mask.npy", "region_4class.npy")
_STAGE_DIR_PREFIX = "vai-stage3-data"
_CAPACITY_SAFETY_FACTOR = 1.2
_PROGRESS_LOG_INTERVAL_SECONDS = 10.0


def _stage_dir_name(pid: int | None = None) -> str:
    """Return a stage directory name unique to this user and process."""
    return (
        f"{_STAGE_DIR_PREFIX}-{os.getuid()}-{pid if pid is not None else os.getpid()}"
    )


def _pid_alive(pid: int) -> bool:
    """Return whether a process with the given PID currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_stages(stage_root: Path) -> None:
    """Remove stage directories left behind by processes that no longer exist.

    Guards against leaked local copies after a SIGKILL or power loss, which
    ``cleanup_stage``'s try/finally cannot catch.
    """
    if not stage_root.exists():
        return
    for entry in stage_root.glob(f"{_STAGE_DIR_PREFIX}-{os.getuid()}-*"):
        if not entry.is_dir():
            continue
        try:
            pid = int(entry.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if not _pid_alive(pid):
            shutil.rmtree(entry, ignore_errors=True)


def _iter_relative_files(source_dir: Path) -> list[Path]:
    """Return dataset-relative paths for the three files each sample needs."""
    return [
        path.relative_to(source_dir)
        for path in source_dir.glob("*/*/*")
        if path.name in STAGE_FILE_NAMES
    ]


def _raise_if_insufficient_capacity(required_bytes: int, stage_root: Path) -> None:
    """Raise if `stage_root` lacks room for `required_bytes` plus a safety margin."""
    available = shutil.disk_usage(stage_root).free
    needed = int(required_bytes * _CAPACITY_SAFETY_FACTOR)
    if available < needed:
        raise RuntimeError(
            f"insufficient space at {stage_root}: need "
            f"~{needed / 1e9:.1f}GB (dataset {required_bytes / 1e9:.1f}GB + "
            f"{int((_CAPACITY_SAFETY_FACTOR - 1) * 100)}% margin), "
            f"only {available / 1e9:.1f}GB free"
        )


def stage_dataset(
    source_dir: Path,
    stage_root: Path,
    max_workers: int = 32,
) -> Path:
    """Copy the dataset's required files onto local storage and return the new root."""
    relative_files = _iter_relative_files(source_dir)
    sizes = {
        relative: (source_dir / relative).stat().st_size for relative in relative_files
    }
    required_bytes = sum(sizes.values())
    stage_root.mkdir(parents=True, exist_ok=True)
    _raise_if_insufficient_capacity(required_bytes, stage_root)

    stage_dir = stage_root / _stage_dir_name()
    stage_dir.mkdir(mode=0o700, parents=True, exist_ok=False)

    def _copy(relative: Path) -> None:
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / relative, destination)

    started = time.perf_counter()
    completed_files = 0
    completed_bytes = 0
    last_log = started
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_relative = {
            executor.submit(_copy, relative): relative for relative in relative_files
        }
        for future in as_completed(future_to_relative):
            future.result()
            completed_files += 1
            completed_bytes += sizes[future_to_relative[future]]
            now = time.perf_counter()
            is_last = completed_files == len(relative_files)
            if now - last_log >= _PROGRESS_LOG_INTERVAL_SECONDS or is_last:
                elapsed = now - started
                rate_mb_s = completed_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
                print(
                    f"[INFO] staging progress: {completed_files}/{len(relative_files)} "
                    f"files ({completed_bytes / 1e9:.1f}/{required_bytes / 1e9:.1f}GB) "
                    f"{rate_mb_s:.0f}MB/s elapsed={elapsed:.0f}s",
                    flush=True,
                )
                last_log = now
    elapsed = time.perf_counter() - started

    print(
        f"[INFO] staged {len(relative_files)} files "
        f"({required_bytes / 1e9:.1f}GB) to {stage_dir} in {elapsed:.1f}s",
        flush=True,
    )
    return stage_dir


def cleanup_stage(stage_dir: Path | None) -> None:
    """Remove a staged dataset directory; safe to call with ``None`` or twice."""
    if stage_dir is None:
        return
    shutil.rmtree(stage_dir, ignore_errors=True)
