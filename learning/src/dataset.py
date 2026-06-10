"""
椎体骨折分類用 Dataset。

bag = 1 椎体の全 z-slice。
各 slice を 2.5D スタック [z-1, z, z+1] に変換して返す。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from learning.utils.augment import augment_bag_slice

# ImageNet 正規化パラメータ
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NORMALIZE = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

# 椎体骨折ラベル（add_fracture_labels.py の FRACTURE_VERTEBRAE と同期）
FRACTURE_VERTEBRAE: dict[str, list[str]] = {
    "sample1":    ["C2"],
    "sample2":    ["C1", "C2"],
    "sample3":    ["C2", "C3", "C4"],
    "sample4":    ["C2"],
    "sample5":    ["C2"],
    "sample6":    [],
    "sample7":    ["C2", "C7"],
    "sample9":    ["C2"],
    "sample10":   ["C6", "C7"],
    "sample12":   ["C7"],
    "sample13":   ["C2"],
    "sample15":   ["C2"],
    "sample15.2": ["C6"],
    "sample17":   ["C2"],
    "sample19":   ["C2"],
    "sample21":   ["C6"],
    "sample22":   ["C2"],
    "sample23":   ["C2"],
    "sample24":   ["C1"],
    "sample25":   ["C7"],
    "sample27":   ["C6", "C7"],
    "sample28":   ["C1", "C2"],
    "sample29":   ["C6", "C7"],
    "sample31":   ["C2"],
    "sample32":   ["C2", "C3", "C4"],
    "sample33":   ["C2", "C5", "C7"],
    "sample34":   ["C5", "C6"],
    "sample35":   ["C6"],
    "sample36":   ["C2", "C7"],
    "sample37":   ["C7"],
    "sample38":   ["C7"],
    "sample41":   ["C6"],
    "sample42":   ["C7"],
    "sample43":   ["C2"],
    "sample44":   ["C2"],
    "sample47":   ["C2", "C3", "C4", "C5"],
    "sample48":   ["C2", "C7"],
    "sample50":   ["C7"],
    "sample51":   ["C2"],
    "sample52":   ["C1"],
    "sample53":   ["C2"],
    "sample54":   ["C3", "C4"],
    "sample55":   ["C5", "C6"],
    "sample56":   ["C5", "C6"],
    "sample57":   ["C2"],
    "sample59":   ["C1", "C2"],
    "sample60":   ["C1", "C2", "C5", "C6"],
    "sample61":   ["C3", "C4"],
    "sample66":   ["C1", "C2"],
    "sample67":   ["C1"],
    "sample68":   ["C2"],
}

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
VERTEBRA_TO_INDEX = {vertebra: index for index, vertebra in enumerate(VERTEBRAE)}


def collect_bags(
    dataset_dir: Path,
) -> list[dict]:
    """
    dataset_zprop/ から全 bag（椎体）のメタデータを収集する。

    各 bag は {sample, vertebra, label, patient_id, slice_paths, mask_paths} を持つ。
    dataset_zprop に存在する (sample, vertebra) のみを対象とする。

    Args:
        dataset_dir: dataset_zprop/ のパス

    Returns:
        bag メタデータのリスト
    """
    bags = []
    for sample_dir in sorted(dataset_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        fractured = set(FRACTURE_VERTEBRAE.get(sample_id, []))

        for vertebra in VERTEBRAE:
            vert_dir = sample_dir / vertebra
            img_dir = vert_dir / "images"
            mask_dir = vert_dir / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                continue

            slice_paths = sorted(img_dir.glob("slice_*.png"))
            if len(slice_paths) == 0:
                continue

            mask_paths = [mask_dir / p.name for p in slice_paths]
            if not all(m.exists() for m in mask_paths):
                continue

            bags.append({
                "sample": sample_id,
                "vertebra": vertebra,
                "label": 1 if vertebra in fractured else 0,
                "patient_id": sample_id,
                "slice_paths": slice_paths,
                "mask_paths": mask_paths,
            })

    return bags


def _load_png_gray(path: Path) -> np.ndarray:
    """PNG を [0,1] の float32 グレースケール配列として読み込む。"""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    """マスク PNG を [0,1] の float32 配列として読み込む。"""
    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float32)
    return (arr > 0).astype(np.float32)


def _build_2d5_stack(
    slice_grays: list[np.ndarray],
    idx: int,
) -> np.ndarray:
    """
    z-1, z, z+1 の 3 枚をスタックして [H, W, 3] を返す。

    端はそのスライス自身を複製する（エッジパディング）。
    """
    t = len(slice_grays)
    prev_idx = max(0, idx - 1)
    next_idx = min(t - 1, idx + 1)
    stack = np.stack(
        [slice_grays[prev_idx], slice_grays[idx], slice_grays[next_idx]],
        axis=-1,
    )  # [H, W, 3]
    return stack


class FractureDataset(Dataset):
    """
    椎体骨折分類用 Dataset（bag = 1 椎体）。

    prefetch_to_ram=True のとき、初期化時に全 PNG を RAM に読み込む。
    NFS 環境では IO ボトルネックを排除でき、2epoch 目以降が大幅に高速化する。

    Returns (per __getitem__):
        stacks:    Tensor [t, 3, H, W]  ImageNet 正規化済み
        label:     int (0/1)
        sample:    str
        vertebra:  str
    """

    def __init__(
        self,
        bags: list[dict],
        training: bool = True,
        aug_cfg: dict | None = None,
        prefetch_to_ram: bool = True,
    ) -> None:
        self.bags = bags
        self.training = training
        self.aug_cfg = aug_cfg

        # RAM キャッシュ: {bag_idx: (grays, masks)}
        self._cache: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        if prefetch_to_ram:
            self._prefetch()

    def _prefetch(self) -> None:
        """全 bag の PNG を RAM に読み込む（初回のみ）。"""
        print(f"[Dataset] RAM プリロード中 ({len(self.bags)} bags)...")
        for i, bag in enumerate(self.bags):
            grays = [_load_png_gray(p) for p in bag["slice_paths"]]
            masks = [_load_mask(p) for p in bag["mask_paths"]]
            self._cache[i] = (grays, masks)
        print("[Dataset] プリロード完了")

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str, str]:
        bag = self.bags[idx]
        label: int = bag["label"]

        # キャッシュがあれば使う、なければファイルから読む
        if idx in self._cache:
            grays, masks = self._cache[idx]
        else:
            grays = [_load_png_gray(p) for p in bag["slice_paths"]]
            masks = [_load_mask(p) for p in bag["mask_paths"]]

        stacks_list: list[Tensor] = []
        for i in range(len(grays)):
            # 2.5D スタック [H, W, 3]
            stack_np = _build_2d5_stack(grays, i)
            mask_np = masks[i]

            # mask 抑制: 背景（他椎体/軟部）をゼロにする
            stack_np = stack_np * mask_np[:, :, np.newaxis]

            # augment（内部で mask 再適用を含む）
            img_t, mask_t = augment_bag_slice(
                stack_np,
                mask_np,
                training=self.training,
                aug_cfg=self.aug_cfg,
            )

            # ImageNet 正規化
            img_t = _NORMALIZE(img_t)
            # 正規化前の背景 0 は大きな負値になるため、中立値 0 に戻す
            img_t = img_t * (mask_t > 0.5).float()

            stacks_list.append(img_t)

        stacks = torch.stack(stacks_list, dim=0)  # [t, 3, H, W]
        return stacks, label, bag["sample"], bag["vertebra"]
