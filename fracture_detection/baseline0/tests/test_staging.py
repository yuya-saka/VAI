from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fracture_detection.baseline0.data.staging import (
    STAGE_MARKER,
    required_stage_bytes,
    stage_dataset,
)


def _write_source(source_dir: Path) -> pd.DataFrame:
    bag_dir = source_dir / "study-a" / "C1"
    bag_dir.mkdir(parents=True)
    for filename in ("ct.npy", "vertebra_mask.npy", "region_4class.npy"):
        (bag_dir / filename).write_bytes(filename.encode("utf-8"))
    return pd.DataFrame(
        [
            {
                "study_id": "study-a",
                "level": "C1",
                "ct_bytes": 1,
                "vertebra_mask_bytes": 1,
                "region_4class_bytes": 1,
            }
        ]
    )


def test_stage_dataset_reuses_completed_manifest_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "source"
    stage_root = tmp_path / "stage"
    manifest = _write_source(source_dir)
    stale_temporary_path = stage_root / ".manifest-sha.tmp-999999"
    stale_temporary_path.mkdir(parents=True)
    (stale_temporary_path / "partial.npy").write_bytes(b"partial")

    staged = stage_dataset(
        manifest, "manifest-sha", source_dir, stage_root, copy_workers=2
    )
    first_output = capsys.readouterr().out
    reused = stage_dataset(
        manifest, "manifest-sha", source_dir, stage_root, copy_workers=2
    )
    reused_output = capsys.readouterr().out

    assert staged == reused
    assert (staged / STAGE_MARKER).is_file()
    assert (staged / "study-a" / "C1" / "ct.npy").read_bytes() == b"ct.npy"
    assert required_stage_bytes(manifest, source_dir) > 0
    assert "lock待機" in first_output
    assert "source走査" in first_output
    assert "容量確認" in first_output
    assert "copy開始" in first_output
    assert "workers=2" in first_output
    assert "copy完了" in first_output
    assert "cache確定" in first_output
    assert "未完了tmp削除" in first_output
    assert not stale_temporary_path.exists()
    assert "cache検証" in reused_output
    assert "既存cacheを再利用" in reused_output
