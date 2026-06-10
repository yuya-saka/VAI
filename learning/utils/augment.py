"""
データ拡張ユーティリティ。

両クラス共通のポリシーで、3-slice スタックとマスクに同一パラメータを適用する。
flip/elastic/cutout/mixup は不使用（左右非対称性を将来利用するため）。
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def _affine_grid(
    angle_deg: float,
    tx: float,
    ty: float,
    scale: float,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """アフィン変換グリッドを生成する（H×W 画像用）。"""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # 回転・スケール・並進を組み合わせた 2×3 行列
    mat = torch.tensor(
        [
            [cos_t / scale, -sin_t / scale, tx],
            [sin_t / scale,  cos_t / scale, ty],
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # [1, 2, 3]
    grid_size = torch.Size([1, 1, size[0], size[1]])
    return F.affine_grid(mat, grid_size, align_corners=False)


def apply_spatial_aug(
    stack: torch.Tensor,
    mask: torch.Tensor,
    p: float = 0.7,
    rot_max: float = 7.0,
    trans_max: float = 0.04,
    scale_min: float = 0.95,
    scale_max: float = 1.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Affine 空間変換を stack（C,H,W）と mask（1,H,W）に同一パラメータで適用する。

    Args:
        stack: shape [C, H, W]（3ch ImageNet 正規化済み）
        mask:  shape [1, H, W]（0/1 float）
        p: 適用確率
        rot_max: 回転角の最大値（度）
        trans_max: 並進量の最大値（画像サイズに対する比率）
        scale_min/scale_max: スケールの範囲

    Returns:
        (変換後 stack, 変換後 mask)
    """
    if random.random() > p:
        return stack, mask

    device = stack.device
    H, W = stack.shape[-2], stack.shape[-1]

    angle = random.uniform(-rot_max, rot_max)
    tx = random.uniform(-trans_max, trans_max)
    ty = random.uniform(-trans_max, trans_max)
    scale = random.uniform(scale_min, scale_max)

    grid = _affine_grid(angle, tx, ty, scale, (H, W), device)

    # 画像: bilinear 補間
    stack_out = F.grid_sample(
        stack.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)

    # マスク: nearest 補間
    mask_out = F.grid_sample(
        mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False
    ).squeeze(0)

    # マスク再適用（背景ゼロ保証）
    stack_out = stack_out * (mask_out > 0.5).float()

    return stack_out, mask_out


def apply_intensity_aug(
    stack: torch.Tensor,
    brightness: float = 0.08,
    contrast: float = 0.12,
    gamma_min: float = 0.9,
    gamma_max: float = 1.1,
    noise_sigma_max: float = 0.02,
    noise_p: float = 0.25,
    blur_sigma_min: float = 0.3,
    blur_sigma_max: float = 0.7,
    blur_p: float = 0.1,
) -> torch.Tensor:
    """
    強度変換を stack 全体に同一パラメータで適用する。

    ImageNet 正規化 *前* の [0,1] float tensor を想定している。
    呼び出し元が正規化順序を管理すること（正規化後は brightness/contrast の
    スケールが変わるため、本関数は正規化前に呼ぶことを推奨）。

    Args:
        stack: shape [C, H, W]（[0,1] float）

    Returns:
        変換後 stack（同 shape）
    """
    # brightness
    b = random.uniform(-brightness, brightness)
    stack = (stack + b).clamp(0.0, 1.0)

    # contrast
    c = random.uniform(1 - contrast, 1 + contrast)
    mean = stack.mean()
    stack = ((stack - mean) * c + mean).clamp(0.0, 1.0)

    # gamma
    g = random.uniform(gamma_min, gamma_max)
    stack = stack.pow(g)

    # Gaussian noise
    if random.random() < noise_p:
        sigma = random.uniform(0.0, noise_sigma_max)
        stack = (stack + torch.randn_like(stack) * sigma).clamp(0.0, 1.0)

    # Gaussian blur（近似: 固定カーネル）
    if random.random() < blur_p:
        sigma = random.uniform(blur_sigma_min, blur_sigma_max)
        # 3×3 ガウスカーネルで近似
        k = 3
        ax = torch.arange(k, dtype=torch.float32, device=stack.device) - k // 2
        kernel_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.outer(kernel_1d).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        C = stack.shape[0]
        kernel_2d = kernel_2d.expand(C, 1, k, k)
        stack = F.conv2d(
            stack.unsqueeze(0), kernel_2d, padding=k // 2, groups=C
        ).squeeze(0).clamp(0.0, 1.0)

    return stack


def augment_bag_slice(
    img: NDArray,
    mask: NDArray,
    training: bool = True,
    aug_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1 slice 分（3ch スタック）の拡張を行い tensor を返す。

    Args:
        img:     numpy [H, W, 3] float32 [0,1]（2.5D スタック）
        mask:    numpy [H, W] float32（0/255 or 0/1）
        training: False なら拡張をスキップ
        aug_cfg: config["augmentation"] dict。None のときはデフォルト値を使用

    Returns:
        (img_tensor [3,H,W], mask_tensor [1,H,W])  float [0,1]
    """
    cfg = aug_cfg or {}

    # HWC → CHW
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

    if training:
        img_t = apply_intensity_aug(
            img_t,
            brightness=cfg.get("brightness", 0.08),
            contrast=cfg.get("contrast", 0.12),
            gamma_min=cfg.get("gamma_min", 0.9),
            gamma_max=cfg.get("gamma_max", 1.1),
            noise_sigma_max=cfg.get("noise_sigma_max", 0.02),
            noise_p=cfg.get("noise_p", 0.25),
            blur_sigma_min=cfg.get("blur_sigma_min", 0.3),
            blur_sigma_max=cfg.get("blur_sigma_max", 0.7),
            blur_p=cfg.get("blur_p", 0.1),
        )
        img_t, mask_t = apply_spatial_aug(
            img_t,
            mask_t,
            p=cfg.get("spatial_p", 0.7),
            rot_max=cfg.get("rot_max", 7.0),
            trans_max=cfg.get("trans_max", 0.04),
            scale_min=cfg.get("scale_min", 0.95),
            scale_max=cfg.get("scale_max", 1.05),
        )

    return img_t, mask_t
