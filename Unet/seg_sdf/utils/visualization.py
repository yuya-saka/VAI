"""可視化関数: SDF オーバーレイ・境界表示・セグメンテーション"""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# SDF 可視化
# -------------------------
def save_sdf_overlay(ct01: np.ndarray, sdf4: np.ndarray, save_path, channel: int = 0) -> None:
    """
    CT画像に SDF チャンネルをオーバーレイして保存する

    符号付きカラーマップ（青=負/外側, 白=境界, 赤=正/内側）を使用

    引数:
        ct01: (H, W) float32 CT画像 [0, 1]
        sdf4: (4, H, W) float32 SDF フィールド [-1, 1]
        save_path: 保存先パス
        channel: 表示するチャンネル（0=upper, 1=lower, 2=left, 3=right）
    """
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    sdf_ch = np.clip(sdf4[channel], -1.0, 1.0)
    # [-1, 1] → [0, 255] にマッピング（0 → 128 = 白）
    sdf_u8 = ((sdf_ch + 1.0) / 2.0 * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(sdf_u8, cv2.COLORMAP_RdBu_r if hasattr(cv2, 'COLORMAP_RdBu_r') else cv2.COLORMAP_JET)

    out = cv2.addWeighted(base, 0.5, heat_color, 0.5, 0)

    ch_names = ["upper", "lower", "left", "right"]
    cv2.putText(out, f"SDF {ch_names[channel]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)


def save_sdf_grid(ct01: np.ndarray, sdf4: np.ndarray, save_path) -> None:
    """
    4チャンネル SDF を 2x2 グリッドで保存する

    引数:
        ct01: (H, W) float32 CT画像 [0, 1]
        sdf4: (4, H, W) float32 SDF フィールド [-1, 1]
        save_path: 保存先パス
    """
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    ch_names = ["upper", "lower", "left", "right"]
    tiles = []
    for c in range(4):
        sdf_ch = np.clip(sdf4[c], -1.0, 1.0)
        sdf_u8 = ((sdf_ch + 1.0) / 2.0 * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(sdf_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base, 0.5, heat_color, 0.5, 0)
        cv2.putText(
            out,
            ch_names[c],
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(out)

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), grid)


# -------------------------
# セグメンテーション可視化
# -------------------------
SEG_COLORS = [
    (50, 50, 50),       # bg: 暗灰色
    (255, 100, 100),    # body: 赤
    (100, 255, 100),    # right: 緑
    (100, 100, 255),    # left: 青
    (255, 255, 100),    # posterior: 黄
]


def save_seg_overlay(
    ct01: np.ndarray,
    seg_pred: np.ndarray,
    save_path,
    alpha: float = 0.45,
) -> None:
    """
    セグメンテーション予測を CT 画像に重ねて保存する

    引数:
        ct01: (H, W) float32 CT画像 [0, 1]
        seg_pred: (H, W) int32/uint8 セグメンテーション予測（クラスインデックス）
        save_path: 保存先パス
        alpha: オーバーレイの透明度
    """
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    seg_color = np.zeros((H, W, 3), dtype=np.uint8)
    for c, color in enumerate(SEG_COLORS):
        mask = seg_pred == c
        seg_color[mask] = color

    out = cv2.addWeighted(base, 1.0 - alpha, seg_color, alpha, 0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)
