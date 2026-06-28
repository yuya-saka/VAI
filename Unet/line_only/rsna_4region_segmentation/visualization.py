"""4 領域分割の可視化ユーティリティ。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .constants import (
    LINE_COLORS_BGR,
    LINE_KEYS,
    REGION_COLORS_BGR,
)
from .inference import PredictedLine


def make_region_overlay(
    ct_u8: np.ndarray,
    region_labels: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """CT グレー画像に 4 領域カラーオーバーレイを重ねた BGR 画像を返す。

    引数:
        ct_u8: (H, W) uint8 CT 画像
        region_labels: (H, W) uint8 ラベルマップ (0-4)
        alpha: 前景領域のブレンド係数
    """
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_layer = np.zeros_like(base)
    for lbl, bgr in enumerate(REGION_COLORS_BGR):
        color_layer[region_labels == lbl] = bgr
    fg = (region_labels > 0).astype(np.float32)[..., None]
    blended = base * (1.0 - fg * alpha) + color_layer * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_lines_on_image(
    img: np.ndarray,
    lines: dict[str, PredictedLine | None] | dict[str, Any],
    highlight_line: str | None = None,
    draw_centroids: bool = False,
) -> np.ndarray:
    """BGR 画像上に 4 本線を描画する（インプレース変更）。"""
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None:
            continue
        if isinstance(info, PredictedLine):
            endpoints = info.endpoints
            centroid = info.centroid
        else:
            endpoints = info.get("endpoints")
            centroid = info.get("centroid")
            if endpoints is None:
                continue

        color = LINE_COLORS_BGR[key]
        thickness = 3 if key == highlight_line else 2
        (x1, y1), (x2, y2) = endpoints
        cv2.line(
            img,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            thickness,
        )
        if draw_centroids and centroid is not None:
            cx, cy = centroid
            cv2.circle(img, (int(round(cx)), int(round(cy))), 3, color, -1)
    return img


def lines_to_polylines(
    lines: dict[str, PredictedLine | None] | dict[str, Any],
) -> dict[str, list[list[float]]] | None:
    """各線の endpoints を polyline 形式 [[x1,y1],[x2,y2]] に変換する。

    4 本全て検出できていない場合は None を返す。
    """
    polylines: dict[str, list[list[float]]] = {}
    for key in LINE_KEYS:
        info = lines.get(key)
        if info is None:
            return None
        if isinstance(info, PredictedLine):
            endpoints = info.endpoints
        else:
            endpoints = info.get("endpoints")
            if endpoints is None:
                return None
        polylines[key] = [list(endpoints[0]), list(endpoints[1])]
    return polylines


def make_heatmap_grid(
    ct_f32: np.ndarray,
    heatmaps: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """4 チャンネルヒートマップを 2×2 グリッドで表示した (H, W, 3) 画像を返す。"""
    H, W = ct_f32.shape
    ct_u8 = (np.clip(ct_f32, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
    tiles = []
    for c in range(4):
        hm = np.clip(heatmaps[c], 0, 1)
        heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tile = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
        color = LINE_COLORS_BGR[LINE_KEYS[c]]
        cv2.putText(
            tile,
            LINE_KEYS[c],
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    return cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)


def concat_with_separator(
    panels: list[np.ndarray],
    axis: int = 1,
    sep_width: int = 2,
    sep_value: int = 60,
) -> np.ndarray:
    """パネルリストをセパレータ付きで結合する。

    axis=1 で横並び、axis=0 で縦並び。
    """
    if not panels:
        raise ValueError("panels must not be empty")
    if axis == 1:
        sep = np.full((panels[0].shape[0], sep_width, 3), sep_value, dtype=np.uint8)
    else:
        sep = np.full((sep_width, panels[0].shape[1], 3), sep_value, dtype=np.uint8)
    result = panels[0]
    for p in panels[1:]:
        result = np.concatenate([result, sep, p], axis=axis)
    return result


def add_title_bar(
    canvas: np.ndarray,
    text: str,
    height: int = 28,
    font_scale: float = 0.55,
) -> np.ndarray:
    """画像上部にタイトルバーを追加する。"""
    title = np.zeros((height, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        text,
        (6, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([title, canvas], axis=0)


def ct_to_bgr(ct_f32: np.ndarray) -> np.ndarray:
    """float32 [0,1] CT スライスを uint8 BGR 画像に変換する。"""
    ct_u8 = (np.clip(ct_f32, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)
