"""full設定用のマニフェスト単位共有ステージングキャッシュ。"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from fracture_detection.baseline0.data.constants import DATASET_DIR

STAGE_ROOT = Path("/dev/shm/vai-fracture-dataset")
STAGE_MARKER = "READY.json"
STAGE_FILES = ("ct.npy", "vertebra_mask.npy", "region_4class.npy")
RESERVED_FREE_BYTES = 1 << 30


@dataclass(frozen=True)
class StageFile:
    """One source file and its destination-relative staging metadata."""

    source_path: Path
    relative_path: Path
    size_bytes: int


def manifest_sha256(manifest_path: Path) -> str:
    """マニフェスト内容のSHA256を返す。"""
    import hashlib

    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def stage_path(stage_root: Path, source_manifest_sha256: str) -> Path:
    """マニフェストのハッシュ値に対応する共有キャッシュパスを返す。"""
    return stage_root / source_manifest_sha256


def _iter_bag_files(
    manifest: pd.DataFrame, source_dir: Path
) -> Iterator[tuple[Path, Path]]:
    """元ファイルとステージ内相対パスを順に返す。"""
    for row in manifest.itertuples(index=False):
        study_id = str(row.study_id)
        level = str(row.level)
        for filename in STAGE_FILES:
            source_path = source_dir / study_id / level / filename
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"staging対象ファイルがありません: {source_path}"
                )
            yield source_path, Path(study_id) / level / filename


def required_stage_bytes(manifest: pd.DataFrame, source_dir: Path) -> int:
    """マニフェストが参照するステージ対象ファイルの総バイト数を返す。"""
    return sum(
        source_path.stat().st_size
        for source_path, _ in _iter_bag_files(manifest, source_dir)
    )


def _build_stage_inventory(manifest: pd.DataFrame, source_dir: Path) -> list[StageFile]:
    """Build the complete file inventory while reporting source-scan progress."""
    total_files = len(manifest) * len(STAGE_FILES)
    print(
        "[staging] source走査開始: "
        f"bags={len(manifest):,}, files={total_files:,}, source={source_dir}",
        flush=True,
    )
    inventory: list[StageFile] = []
    with tqdm(
        _iter_bag_files(manifest, source_dir),
        total=total_files,
        desc="[staging] source走査",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress:
        for source_path, relative_path in progress:
            inventory.append(
                StageFile(
                    source_path=source_path,
                    relative_path=relative_path,
                    size_bytes=source_path.stat().st_size,
                )
            )
            progress.set_postfix_str(relative_path.as_posix(), refresh=False)
    required_bytes = sum(item.size_bytes for item in inventory)
    print(
        "[staging] source走査完了: "
        f"files={len(inventory):,}, size={_format_bytes(required_bytes)}",
        flush=True,
    )
    return inventory


def _marker_payload(
    source_manifest_sha256: str, manifest: pd.DataFrame, required_bytes: int
) -> dict[str, int | str]:
    """完了マーカーの内容を構築する。"""
    return {
        "manifest_sha256": source_manifest_sha256,
        "bags": int(len(manifest)),
        "files": int(len(manifest) * len(STAGE_FILES)),
        "bytes": required_bytes,
    }


def _is_ready(stage_dir: Path, expected: dict[str, int | str]) -> bool:
    """マーカーと必須ファイル数からキャッシュの完成を確認する。"""
    print(f"[staging] cache検証開始: {stage_dir}", flush=True)
    marker_path = stage_dir / STAGE_MARKER
    if not marker_path.is_file():
        print("[staging] cache検証: READY markerなし", flush=True)
        return False
    try:
        actual = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("[staging] cache検証: READY markerが不正", flush=True)
        return False
    if actual != expected:
        print("[staging] cache検証: READY markerが期待値と不一致", flush=True)
        return False
    expected_files = int(expected["files"])
    file_count = 0
    with tqdm(
        stage_dir.rglob("*.npy"),
        total=expected_files,
        desc="[staging] cache検証",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress:
        for path in progress:
            if path.is_file():
                file_count += 1
            progress.set_postfix_str(
                path.relative_to(stage_dir).as_posix(), refresh=False
            )
    ready = file_count == expected_files
    print(
        "[staging] cache検証完了: "
        f"files={file_count:,}/{expected_files:,}, ready={ready}",
        flush=True,
    )
    return ready


@contextmanager
def _stage_lock(source_manifest_sha256: str) -> Iterator[None]:
    """同じマニフェストを複数プロセスが同時コピーしないようにする。"""
    lock_path = Path("/tmp") / f"vai-fracture-dataset-{source_manifest_sha256}.lock"
    with lock_path.open("a+") as lock_file:
        started_at = time.monotonic()
        print(f"[staging] lock待機: {lock_path}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        waited_seconds = time.monotonic() - started_at
        print(
            f"[staging] lock取得: wait={waited_seconds:.1f}s, path={lock_path}",
            flush=True,
        )
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[staging] lock解放: {lock_path}", flush=True)


def _copy_stage_file(item: StageFile, temporary_path: Path) -> StageFile:
    """Copy one inventory item without preserving unnecessary NFS metadata."""
    destination = temporary_path / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(item.source_path, destination)
    return item


def _copy_inventory(
    inventory: list[StageFile], temporary_path: Path, copy_workers: int
) -> None:
    """Copy all inventoried files while reporting byte and file progress."""
    if copy_workers < 1:
        raise ValueError("copy_workers must be at least 1")
    total_bytes = sum(item.size_bytes for item in inventory)
    started_at = time.monotonic()
    print(
        "[staging] copy開始: "
        f"files={len(inventory):,}, size={_format_bytes(total_bytes)}, "
        f"workers={copy_workers}, destination={temporary_path}",
        flush=True,
    )
    futures: dict[Future[StageFile], StageFile] = {}
    try:
        with ThreadPoolExecutor(
            max_workers=copy_workers, thread_name_prefix="fracture-stage"
        ) as executor:
            futures = {
                executor.submit(_copy_stage_file, item, temporary_path): item
                for item in inventory
            }
            with tqdm(
                total=total_bytes,
                desc="[staging] copy",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                file=sys.stdout,
            ) as progress:
                for file_index, future in enumerate(as_completed(futures), start=1):
                    item = future.result()
                    progress.update(item.size_bytes)
                    progress.set_postfix(
                        files=f"{file_index:,}/{len(inventory):,}",
                        current=item.relative_path.as_posix(),
                        refresh=False,
                    )
    except Exception:
        for future in futures:
            future.cancel()
        print(
            f"[staging] copy失敗: temporary cacheを確認してください: {temporary_path}",
            flush=True,
        )
        raise
    elapsed_seconds = time.monotonic() - started_at
    bytes_per_second = total_bytes / max(elapsed_seconds, 1e-9)
    print(
        "[staging] copy完了: "
        f"files={len(inventory):,}, size={_format_bytes(total_bytes)}, "
        f"elapsed={elapsed_seconds:.1f}s, speed={_format_bytes(bytes_per_second)}/s",
        flush=True,
    )


def _format_bytes(byte_count: float) -> str:
    """Format a byte count for human-readable progress output."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _remove_stale_temporary_paths(
    stage_root: Path, source_manifest_sha256: str
) -> None:
    """Remove abandoned temporary directories while holding the manifest lock."""
    pattern = f".{source_manifest_sha256}.tmp-*"
    stale_paths = sorted(stage_root.glob(pattern))
    print(
        f"[staging] 未完了tmp確認: count={len(stale_paths):,}, pattern={pattern}",
        flush=True,
    )
    for stale_path in stale_paths:
        if not stale_path.is_dir() or stale_path.is_symlink():
            raise RuntimeError(f"予期しないstaging temporary pathです: {stale_path}")
        print(f"[staging] 未完了tmp削除: {stale_path}", flush=True)
        shutil.rmtree(stale_path)


def stage_dataset(
    manifest: pd.DataFrame,
    source_manifest_sha256: str,
    source_dir: Path = DATASET_DIR,
    stage_root: Path = STAGE_ROOT,
    copy_workers: int = 8,
) -> Path:
    """データセットを共有キャッシュへ1回だけコピーし、完成パスを返す。"""
    required_columns = {"study_id", "level"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    if manifest.duplicated(["study_id", "level"]).any():
        raise ValueError("staging manifestに重複したstudy_id・levelがあります")

    stage_root.mkdir(parents=True, exist_ok=True)
    final_path = stage_path(stage_root, source_manifest_sha256)
    print(
        "[staging] 準備開始: "
        f"manifest={source_manifest_sha256}, destination={final_path}",
        flush=True,
    )

    with _stage_lock(source_manifest_sha256):
        _remove_stale_temporary_paths(stage_root, source_manifest_sha256)
        inventory = _build_stage_inventory(manifest, source_dir)
        required_bytes = sum(item.size_bytes for item in inventory)
        expected = _marker_payload(source_manifest_sha256, manifest, required_bytes)
        if _is_ready(final_path, expected):
            print(f"[staging] 既存cacheを再利用: {final_path}", flush=True)
            return final_path
        if final_path.exists():
            raise RuntimeError(
                f"未完成または不整合なstaging cacheがあります: {final_path}。手動で確認してください"
            )

        available_bytes = shutil.disk_usage(stage_root).free
        print(
            "[staging] 容量確認: "
            f"data={_format_bytes(required_bytes)}, "
            f"reserve={_format_bytes(RESERVED_FREE_BYTES)}, "
            f"required={_format_bytes(required_bytes + RESERVED_FREE_BYTES)}, "
            f"available={_format_bytes(available_bytes)}",
            flush=True,
        )
        if available_bytes < required_bytes + RESERVED_FREE_BYTES:
            raise RuntimeError(
                "staging先の空き容量が不足しています: "
                f"必要={required_bytes + RESERVED_FREE_BYTES}, 空き={available_bytes}"
            )

        temporary_path = stage_root / f".{source_manifest_sha256}.tmp-{os.getpid()}"
        if temporary_path.exists():
            raise RuntimeError(
                f"未完了の一時staging directoryがあります: {temporary_path}"
            )
        temporary_path.mkdir()
        _copy_inventory(inventory, temporary_path, copy_workers)

        print(
            f"[staging] READY marker作成: {temporary_path / STAGE_MARKER}", flush=True
        )
        (temporary_path / STAGE_MARKER).write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[staging] cache確定: {temporary_path} -> {final_path}", flush=True)
        temporary_path.replace(final_path)
        print(f"[staging] 準備完了: {final_path}", flush=True)
    return final_path
