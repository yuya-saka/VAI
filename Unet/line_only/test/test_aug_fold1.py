#!/usr/bin/env python
"""
fold1 での augmentation 比較実験
ShiftScaleRotate（旧）vs Affine（新）の影響を確認する

使い方:
    uv run python Unet/line_only/test/test_aug_fold1.py
"""

import copy
import sys
from pathlib import Path

import albumentations as A
import cv2

_here = Path(__file__).resolve().parent       # test/
_unet = _here.parent.parent                   # Unet/
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))

import line_only.src.dataset as dataset_module
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

    if cfg_aug.get("horizontal_flip", False):
        return A.ReplayCompose(ts, additional_targets=additional_targets)
    return A.Compose(ts, additional_targets=additional_targets)


def run_fold1(aug_name: str, patch_fn=None) -> dict:
    """
    fold1 を学習して結果を返す

    引数:
        aug_name: 実験名（チェックポイントディレクトリ名に使用）
        patch_fn: get_transforms を差し替える関数（None = 現行 Affine を使用）

    戻り値:
        train_one_fold の結果辞書
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

    # augmentation をパッチ
    original = dataset_module.get_transforms
    if patch_fn is not None:
        dataset_module.get_transforms = patch_fn

    print(f"\n{'=' * 60}")
    print(f"[EXP] aug={aug_name}  fold=1")
    print(f"{'=' * 60}")

    try:
        results = train_one_fold(cfg)
    finally:
        dataset_module.get_transforms = original

    return results


def main():
    print("\n" + "=" * 60)
    print("[比較実験] ShiftScaleRotate vs Affine (fold1)")
    print("=" * 60)
    print("既知の結果:")
    print("  baseline (旧shim + ShiftScaleRotate) : 8.60°")
    print("  sig2.0_base (新src + Affine)         : 17.77°")
    print("=" * 60)

    # --- 実験1: ShiftScaleRotate（旧）---
    results_ssr = run_fold1("ShiftScaleRotate", patch_fn=_get_transforms_ssr)

    # --- 実験2: Affine（新、現行コード）---
    results_affine = run_fold1("Affine", patch_fn=None)

    # --- 結果比較 ---
    angle_ssr = results_ssr.get("line_angle_error_deg_mean")
    angle_affine = results_affine.get("line_angle_error_deg_mean")

    print("\n" + "=" * 60)
    print("[結果まとめ]")
    print("=" * 60)
    print(f"  baseline (参考)                      :  8.60°")
    print(f"  新src + ShiftScaleRotate (今回)       : {angle_ssr:.2f}°")
    print(f"  新src + Affine (今回)                 : {angle_affine:.2f}°")
    print(f"  sig2.0_base (参考)                   : 17.77°")
    print("=" * 60)

    if angle_ssr is not None and angle_affine is not None:
        diff = angle_affine - angle_ssr
        print(f"\n差分 (Affine - ShiftScaleRotate): {diff:+.2f}°")
        if abs(diff) > 2.0:
            print("→ augmentation の変更が fold1 の精度に大きく影響している")
        else:
            print("→ augmentation の変更は fold1 に大きく影響していない（別の原因）")


if __name__ == "__main__":
    main()
