"""中心付き5スライスDatasetのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from Unet.line_2p5d.src.dataset import CenteredLineDataset, build_centered_records


def _lines(offset: float = 0.0) -> dict[str, list[list[float]]]:
    """4本の水平線fixtureを返す。"""
    return {
        "line_1": [[2.0, 3.0 + offset], [13.0, 3.0 + offset]],
        "line_2": [[2.0, 6.0 + offset], [13.0, 6.0 + offset]],
        "line_3": [[2.0, 9.0 + offset], [13.0, 9.0 + offset]],
        "line_4": [[2.0, 12.0 + offset], [13.0, 12.0 + offset]],
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    """密スライスと手動中心線の最小fixtureを作る。"""
    dense_root = root / "dense"
    annotation_root = root / "annotation"
    dense_dir = dense_root / "sample1" / "C1"
    annotation_dir = annotation_root / "sample1" / "C1"
    (dense_dir / "images").mkdir(parents=True)
    (dense_dir / "masks").mkdir()
    annotation_dir.mkdir(parents=True)
    for slice_index in range(7):
        image = np.full((16, 16), slice_index * 30, dtype=np.uint8)
        mask = np.full((16, 16), 255, dtype=np.uint8)
        Image.fromarray(image).save(
            dense_dir / "images" / f"slice_{slice_index:03d}.png"
        )
        Image.fromarray(mask).save(dense_dir / "masks" / f"slice_{slice_index:03d}.png")
    (annotation_dir / "lines.json").write_text(
        json.dumps({"1": _lines(), "3": _lines(0.2)}),
        encoding="utf-8",
    )
    return dense_root, annotation_root


def test_records_use_one_sample_per_manual_center(tmp_path: Path) -> None:
    """教師画像を重複させず、中心±2の入力を作る。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_centered_records(
        dense_root,
        annotation_root,
        ["sample1"],
        "C1",
        (-2, -1, 0, 1, 2),
    )
    assert [record.center_slice_index for record in records] == [1, 3]
    assert records[0].context_slice_indices == (0, 0, 1, 2, 3)
    assert records[0].context_valid == (False, True, True, True, True)
    assert records[1].context_slice_indices == (1, 2, 3, 4, 5)


def test_dataset_returns_slice_explicit_contract(tmp_path: Path) -> None:
    """入力・中心教師・文脈maskのshapeを固定する。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_centered_records(
        dense_root,
        annotation_root,
        ["sample1"],
        "C1",
        (-2, -1, 0, 1, 2),
    )
    item = CenteredLineDataset(records, image_size=16, sigma=1.5)[1]
    assert item["image"].shape == (5, 2, 16, 16)
    assert item["heatmaps"].shape == (4, 16, 16)
    assert item["line_params_gt"].shape == (4, 2)
    assert json.loads(item["lines_json"])["line_1"] == _lines(0.2)["line_1"]
    assert item["context_valid"].shape == (5,)
    assert torch.equal(item["context_slice_indices"], torch.tensor([1, 2, 3, 4, 5]))
    center_intensity = float(item["image"][2, 0].mean())
    assert center_intensity == pytest.approx(90.0 / 255.0)
