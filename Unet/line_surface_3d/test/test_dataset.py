"""スラブDatasetと共有augmentationのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from line_surface_3d.src.dataset import (
    SlabLineDataset,
    build_slab_records,
    get_transforms,
)
from PIL import Image


def _lines(offset: float = 0.0) -> dict[str, list[list[float]]]:
    """4本の単純な線を作る。"""
    return {
        "line_1": [[3.0 + offset, 3.0], [12.0 + offset, 3.0]],
        "line_2": [[3.0 + offset, 6.0], [12.0 + offset, 6.0]],
        "line_3": [[3.0 + offset, 9.0], [12.0 + offset, 9.0]],
        "line_4": [[3.0 + offset, 12.0], [12.0 + offset, 12.0]],
    }


def _write_fixture(root: Path, slice_count: int = 7) -> tuple[Path, Path]:
    """最小の密画像・手動教師fixtureを作る。"""
    dense_root = root / "dense"
    annotation_root = root / "annotation"
    dense_dir = dense_root / "sample1" / "C1"
    annotation_dir = annotation_root / "sample1" / "C1"
    (dense_dir / "images").mkdir(parents=True)
    (dense_dir / "masks").mkdir()
    annotation_dir.mkdir(parents=True)
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 180
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:14, 2:14] = 255
    for slice_index in range(slice_count):
        Image.fromarray(image).save(
            dense_dir / "images" / f"slice_{slice_index:03d}.png"
        )
        Image.fromarray(mask).save(dense_dir / "masks" / f"slice_{slice_index:03d}.png")
    labels = {str(index): _lines(float(index) * 0.1) for index in (1, 2, 3, 4)}
    (annotation_dir / "lines.json").write_text(
        json.dumps(labels),
        encoding="utf-8",
    )
    (dense_dir / "lines.json").write_text(
        json.dumps({"0": {"poison": True}}),
        encoding="utf-8",
    )
    return dense_root, annotation_root


def test_build_slab_records_uses_dense_images_and_manual_labels(
    tmp_path: Path,
) -> None:
    """擬似線JSONを使わず、連続窓だけを作る。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_slab_records(
        dense_root=dense_root,
        annotation_root=annotation_root,
        sample_names=["sample1"],
        group="C1",
        slab_size=3,
        stride=1,
        min_labeled_slices=1,
        require_labels=True,
    )
    assert [record.slice_indices for record in records] == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
    ]
    assert set(records[0].labels) == {1, 2}
    assert all("poison" not in lines for lines in records[0].labels.values())


def test_inference_records_cover_last_slice(tmp_path: Path) -> None:
    """strideが大きくても末尾窓を追加する。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_slab_records(
        dense_root=dense_root,
        annotation_root=annotation_root,
        sample_names=["sample1"],
        group="C1",
        slab_size=3,
        stride=3,
        min_labeled_slices=1,
        require_labels=False,
    )
    assert [record.slice_indices for record in records] == [
        (0, 1, 2),
        (3, 4, 5),
        (4, 5, 6),
    ]


def test_dataset_returns_slice_major_contract(tmp_path: Path) -> None:
    """入力・教師・label maskのshapeを固定する。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_slab_records(
        dense_root,
        annotation_root,
        ["sample1"],
        "C1",
        slab_size=3,
        stride=1,
        min_labeled_slices=1,
        require_labels=True,
    )
    item = SlabLineDataset(records, image_size=16, sigma=1.5)[0]
    assert item["image"].shape == (6, 16, 16)
    assert item["heatmaps"].shape == (3, 4, 16, 16)
    assert item["label_mask"].shape == (3, 4)
    assert item["line_params_gt"].shape == (3, 4, 2)
    assert item["label_mask"].sum().item() == 8
    assert torch.isnan(item["line_params_gt"][0]).all()
    assert torch.isfinite(item["line_params_gt"][item["label_mask"]]).all()
    assert torch.equal(item["slice_indices"], torch.tensor([0, 1, 2]))


def test_shared_replay_keeps_identical_slices_identical(tmp_path: Path) -> None:
    """同一画像へ同じaugmentation replayが適用される。"""
    dense_root, annotation_root = _write_fixture(tmp_path)
    records = build_slab_records(
        dense_root,
        annotation_root,
        ["sample1"],
        "C1",
        slab_size=3,
        stride=1,
        min_labeled_slices=1,
        require_labels=True,
    )
    transform = get_transforms(
        "train",
        {
            "rotation": True,
            "rotation_limit": 20,
            "brightness_contrast": True,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
        },
    )
    item = SlabLineDataset(
        records,
        image_size=16,
        sigma=1.5,
        transform=transform,
    )[1]
    ct_slices = item["image"][0::2]
    mask_slices = item["image"][1::2]
    assert torch.allclose(ct_slices[0], ct_slices[1])
    assert torch.allclose(ct_slices[1], ct_slices[2])
    assert torch.allclose(mask_slices[0], mask_slices[1])
