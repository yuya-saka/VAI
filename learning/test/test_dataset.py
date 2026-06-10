"""
dataset.py のユニットテスト。

2.5D スタック形状・端複製・mask 抑制を検証する。
実ファイル（dataset_zprop）を 1 bag 分だけ読み込んで smoke test も行う。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from learning.src.dataset import FractureDataset, _build_2d5_stack, collect_bags


class TestBuild2d5Stack:
    def test_shape(self):
        """出力形状が [H, W, 3] になる。"""
        grays = [np.zeros((224, 224), dtype=np.float32) for _ in range(5)]
        stack = _build_2d5_stack(grays, idx=2)
        assert stack.shape == (224, 224, 3)

    def test_edge_replication_start(self):
        """idx=0 では [0, 0, 1] をスタックする（前端複製）。"""
        grays = [np.full((4, 4), float(i), dtype=np.float32) for i in range(5)]
        stack = _build_2d5_stack(grays, idx=0)
        # ch0 と ch1 は同じ（idx=0 の複製）
        np.testing.assert_array_equal(stack[:, :, 0], stack[:, :, 1])
        np.testing.assert_array_equal(stack[:, :, 1], grays[0])
        np.testing.assert_array_equal(stack[:, :, 2], grays[1])

    def test_edge_replication_end(self):
        """idx=末尾 では [t-2, t-1, t-1] をスタックする（後端複製）。"""
        grays = [np.full((4, 4), float(i), dtype=np.float32) for i in range(5)]
        last = len(grays) - 1
        stack = _build_2d5_stack(grays, idx=last)
        np.testing.assert_array_equal(stack[:, :, 1], stack[:, :, 2])
        np.testing.assert_array_equal(stack[:, :, 1], grays[last])

    def test_middle_index(self):
        """中間 idx では [z-1, z, z+1] を正しくスタックする。"""
        grays = [np.full((4, 4), float(i), dtype=np.float32) for i in range(5)]
        stack = _build_2d5_stack(grays, idx=2)
        np.testing.assert_array_equal(stack[:, :, 0], grays[1])
        np.testing.assert_array_equal(stack[:, :, 1], grays[2])
        np.testing.assert_array_equal(stack[:, :, 2], grays[3])


class TestFractureDatasetMock:
    """モックデータで Dataset の動作を検証する。"""

    def _make_fake_bag(self, n_slices: int = 10, label: int = 0) -> dict:
        """一時ディレクトリ不要のフェイク bag メタデータ（パスはモック用）。"""
        return {
            "sample": "sample_test",
            "vertebra": "C3",
            "label": label,
            "patient_id": "sample_test",
            "slice_paths": [Path(f"img_{i:03d}.png") for i in range(n_slices)],
            "mask_paths": [Path(f"mask_{i:03d}.png") for i in range(n_slices)],
        }

    def _patch_loaders(self, n_slices: int):
        """_load_png_gray と _load_mask をランダム配列で差し替えるパッチ。"""
        gray = np.random.rand(224, 224).astype(np.float32)
        mask = (np.random.rand(224, 224) > 0.3).astype(np.float32)
        p_gray = patch("learning.src.dataset._load_png_gray", return_value=gray)
        p_mask = patch("learning.src.dataset._load_mask", return_value=mask)
        return p_gray, p_mask

    def test_output_shape(self):
        """__getitem__ が [t, 3, 224, 224] の tensor を返す。"""
        n = 15
        bag = self._make_fake_bag(n_slices=n, label=1)
        p_gray, p_mask = self._patch_loaders(n)
        with p_gray, p_mask:
            ds = FractureDataset([bag], training=False)
            stacks, label, sample, vertebra = ds[0]

        assert stacks.shape == (n, 3, 224, 224)
        assert label == 1
        assert sample == "sample_test"
        assert vertebra == "C3"

    def test_mask_suppression(self):
        """mask が 0 の領域は ImageNet 正規化後も中立値 0 になる。"""
        n = 5
        bag = self._make_fake_bag(n_slices=n)
        all_zero_mask = np.zeros((224, 224), dtype=np.float32)
        all_one_mask = np.ones((224, 224), dtype=np.float32)
        gray = np.ones((224, 224), dtype=np.float32) * 0.5

        with (
            patch("learning.src.dataset._load_png_gray", return_value=gray),
            patch("learning.src.dataset._load_mask", return_value=all_zero_mask),
        ):
            ds = FractureDataset([bag], training=False)
            stacks_zero, *_ = ds[0]

        with (
            patch("learning.src.dataset._load_png_gray", return_value=gray),
            patch("learning.src.dataset._load_mask", return_value=all_one_mask),
        ):
            ds = FractureDataset([bag], training=False)
            stacks_one, *_ = ds[0]

        assert torch.count_nonzero(stacks_zero) == 0
        assert not torch.allclose(stacks_zero, stacks_one)

    def test_label_0(self):
        """陰性 bag (label=0) が正しく返る。"""
        bag = self._make_fake_bag(n_slices=8, label=0)
        gray = np.zeros((224, 224), dtype=np.float32)
        mask = np.ones((224, 224), dtype=np.float32)
        with (
            patch("learning.src.dataset._load_png_gray", return_value=gray),
            patch("learning.src.dataset._load_mask", return_value=mask),
        ):
            ds = FractureDataset([bag], training=False)
            _, label, _, _ = ds[0]
        assert label == 0


class TestCollectBagsSmoke:
    """実 dataset_zprop を使ったスモークテスト（存在確認のみ）。"""

    def test_returns_nonempty_list(self):
        """dataset_zprop が存在すれば bag が収集できる。"""
        root = Path(__file__).resolve().parents[2]
        zprop_dir = root / "dataset_zprop"
        if not zprop_dir.exists():
            pytest.skip("dataset_zprop が存在しない")

        bags = collect_bags(zprop_dir)
        assert len(bags) > 0

    def test_bag_fields(self):
        """各 bag が必須フィールドを持つ。"""
        root = Path(__file__).resolve().parents[2]
        zprop_dir = root / "dataset_zprop"
        if not zprop_dir.exists():
            pytest.skip("dataset_zprop が存在しない")

        bags = collect_bags(zprop_dir)
        for bag in bags[:5]:
            assert "sample" in bag
            assert "vertebra" in bag
            assert "label" in bag
            assert "patient_id" in bag
            assert len(bag["slice_paths"]) > 0
            assert bag["label"] in (0, 1)

    def test_label_counts(self):
        """陽性 bag が存在する。"""
        root = Path(__file__).resolve().parents[2]
        zprop_dir = root / "dataset_zprop"
        if not zprop_dir.exists():
            pytest.skip("dataset_zprop が存在しない")

        bags = collect_bags(zprop_dir)
        n_pos = sum(b["label"] for b in bags)
        assert n_pos > 0
