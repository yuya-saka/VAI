#!/usr/bin/env python
"""
fold1 での augmentation 比較実験（修正版）
ShiftScaleRotate（旧）vs Affine（新）の影響を確認する

重要:
- 実際に学習で使われるのは line_only.src.data_utils 側で import 済みの
  get_transforms なので、dataset だけでなく data_utils 側も patch する
"""

import sys
from pathlib import Path

import albumentations as A
import cv2

_here = Path(__file__).resolve().parent       # test/
_unet = _here.parent.parent                   # Unet/
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))

import line_only.src.dataset as dataset_module
import line_only.src.data_utils as data_utils_module
from line_only.src.data_utils import load_config, set_seed
from line_only.src.trainer import train_one_fold


# -------------------------
# 旧 augmentation (ShiftScaleRotate)
# -------------------------
def _get_transforms_ssr(phase="train", cfg_aug=None):
    """旧来の ShiftScaleRotate ベースの augmentation"""
    if phase != "train":
        return None

    cfg_aug = cfg_aug or {}
    ts = []

    if cfg_aug.get("rotation", False):
        ts.append(A.Rotate(limit=float(cfg_aug.get("rotation_limit", 20)), p=0.5))

    if cfg_aug.get("scale", False):
        ts.append(
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=float(cfg_aug.get("scale_limit", 0.1)),
                rotate_limit=0,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            )
        )

    # flip を使っていないなら入らない
    if cfg_aug.get("horizontal_flip", False):
        ts.append(
            A.HorizontalFlip(
                p=float(cfg_aug.get("horizontal_flip_prob", 0.1))
            )
        )

    if cfg_aug.get("brightness_contrast", False):
        ts.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(cfg_aug.get("brightness_limit", 0.2)),
                contrast_limit=float(cfg_aug.get("contrast_limit", 0.2)),
                p=0.5,
            )
        )

    if cfg_aug.get("gaussian_noise", False):
        var_lim = cfg_aug.get("noise_var_limit", [10, 50])
        ts.append(A.GaussNoise(var_limit=tuple(var_lim), p=0.3))

    additional_targets = {
        "mask": "mask",
        "hm1": "image",
        "hm2": "image",
        "hm3": "image",
        "hm4": "image",
    }

    # flip 適用の有無を記録したい場合のみ ReplayCompose
    if cfg_aug.get("horizontal_flip", False):
        return A.ReplayCompose(ts, additional_targets=additional_targets)
    return A.Compose(ts, additional_targets=additional_targets)


def _patch_get_transforms(patch_fn):
    """
    実際に学習で参照される get_transforms を patch する。
    data_utils.py は dataset.py から get_transforms を import 済みなので、
    dataset だけでなく data_utils 側も差し替える必要がある。
    """
    originals = {
        "dataset": dataset_module.get_transforms,
        "data_utils": data_utils_module.get_transforms,
    }

    if patch_fn is not None:
        dataset_module.get_transforms = patch_fn
        data_utils_module.get_transforms = patch_fn

    return originals


def _restore_get_transforms(originals):
    dataset_module.get_transforms = originals["dataset"]
    data_utils_module.get_transforms = originals["data_utils"]


def run_fold1(aug_name: str, patch_fn=None) -> dict:
    """
    fold1 を学習して結果を返す

    引数:
        aug_name: 実験名（チェックポイントディレクトリ名に使用）
        patch_fn: get_transforms を差し替える関数
                  None = 現行 Affine 実装をそのまま使う
    """
    cfg = load_config("config/config.yaml")

    # fold1 固定
    cfg["data"]["test_fold"] = 1

    # 実験ごとに別ディレクトリに保存
    cfg["training"]["checkpoint_dir"] = f"outputs/checkpoints_aug_cmp_{aug_name}"
    cfg["evaluation"]["visualization_dir"] = f"outputs/vis_aug_cmp_{aug_name}"

    # wandb は無効（比較実験なので不要）
    cfg["wandb"]["enabled"] = False

    seed = int(cfg.get("data", {}).get("random_seed", 42))
    set_seed(seed)

    originals = _patch_get_transforms(patch_fn)

    print(f"\n{'=' * 60}")
    print(f"[EXP] aug={aug_name}  fold=1")
    print(f"{'=' * 60}")

    try:
        results = train_one_fold(cfg)
    finally:
        _restore_get_transforms(originals)

    return results


def main():
    print("\n" + "=" * 60)
    print("[比較実験] ShiftScaleRotate vs Affine (fold1) [修正版]")
    print("=" * 60)

    # 実験1: 旧 SSR
    results_ssr = run_fold1("ShiftScaleRotate_fixed", patch_fn=_get_transforms_ssr)

    # 実験2: 現行 Affine
    results_affine = run_fold1("Affine_fixed", patch_fn=None)

    angle_ssr = results_ssr.get("line_angle_error_deg_mean")
    angle_affine = results_affine.get("line_angle_error_deg_mean")

    print("\n" + "=" * 60)
    print("[結果まとめ]")
    print("=" * 60)
    print(f"  新src + ShiftScaleRotate (修正版) : {angle_ssr:.2f}°")
    print(f"  新src + Affine (修正版)           : {angle_affine:.2f}°")
    print("=" * 60)

    if angle_ssr is not None and angle_affine is not None:
        diff = angle_affine - angle_ssr
        print(f"\n差分 (Affine - ShiftScaleRotate): {diff:+.2f}°")
        if abs(diff) > 2.0:
            print("→ augmentation の変更が精度に影響している可能性が高い")
        else:
            print("→ augmentation の変更は主因ではない可能性が高い")


if __name__ == "__main__":
    main()