"""full設定用のマニフェスト単位共有ステージングキャッシュ。"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from fracture_detection.common.constants import DATASET_DIR

STAGE_ROOT = Path("/dev/shm/vai-baseline1")
STAGE_MARKER = "READY.json"
STAGE_FILES = ("ct.npy", "vertebra_mask.npy", "region_4class.npy")
RESERVED_FREE_BYTES = 1 << 30


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
    marker_path = stage_dir / STAGE_MARKER
    if not marker_path.is_file():
        return False
    try:
        actual = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if actual != expected:
        return False
    file_count = sum(1 for path in stage_dir.rglob("*.npy") if path.is_file())
    return file_count == expected["files"]


@contextmanager
def _stage_lock(source_manifest_sha256: str) -> Iterator[None]:
    """同じマニフェストを複数プロセスが同時コピーしないようにする。"""
    lock_path = Path("/tmp") / f"vai-baseline1-{source_manifest_sha256}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def stage_dataset(
    manifest: pd.DataFrame,
    source_manifest_sha256: str,
    source_dir: Path = DATASET_DIR,
    stage_root: Path = STAGE_ROOT,
) -> Path:
    """データセットを共有キャッシュへ1回だけコピーし、完成パスを返す。"""
    required_columns = {"study_id", "level"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    if manifest.duplicated(["study_id", "level"]).any():
        raise ValueError("staging manifestに重複したstudy_id・levelがあります")

    stage_root.mkdir(parents=True, exist_ok=True)
    required_bytes = required_stage_bytes(manifest, source_dir)
    expected = _marker_payload(source_manifest_sha256, manifest, required_bytes)
    final_path = stage_path(stage_root, source_manifest_sha256)

    with _stage_lock(source_manifest_sha256):
        if _is_ready(final_path, expected):
            return final_path
        if final_path.exists():
            raise RuntimeError(
                f"未完成または不整合なstaging cacheがあります: {final_path}。手動で確認してください"
            )

        available_bytes = shutil.disk_usage(stage_root).free
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
        for source_path, relative_path in _iter_bag_files(manifest, source_dir):
            destination = temporary_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        (temporary_path / STAGE_MARKER).write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(final_path)
    return final_path
