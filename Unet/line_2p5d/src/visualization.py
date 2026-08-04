"""line_onlyと同形式のheatmap・直線可視化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

LINE_COLORS = {
    "line_1": (0, 255, 0),
    "line_2": (0, 0, 255),
    "line_3": (255, 0, 0),
    "line_4": (0, 255, 255),
}


def _base_image(ct_image: np.ndarray) -> np.ndarray:
    """0-1 CTをBGR画像へ変換する。"""
    ct_uint8 = (np.clip(ct_image, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(ct_uint8, cv2.COLOR_GRAY2BGR)


def _write_image(path: Path, image: np.ndarray) -> None:
    """親directoryを作成して画像を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"画像を保存できませんでした: {path}")


def save_heatmap_overlay(
    ct_image: np.ndarray,
    heatmaps: np.ndarray,
    save_path: Path,
    alpha: float = 0.55,
) -> None:
    """4channel最大heatmapをCTへ重ねて保存する。"""
    base = _base_image(ct_image)
    merged = np.clip(np.max(heatmaps, axis=0), 0, 1)
    heat_color = cv2.applyColorMap((merged * 255).astype(np.uint8), cv2.COLORMAP_JET)
    _write_image(save_path, cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0))


def save_heatmap_grid(
    ct_image: np.ndarray,
    heatmaps: np.ndarray,
    save_path: Path,
    alpha: float = 0.55,
) -> None:
    """4channel heatmapを2x2 gridで保存する。"""
    base = _base_image(ct_image)
    tiles: list[np.ndarray] = []
    for channel in range(4):
        heatmap = np.clip(heatmaps[channel], 0, 1)
        heat_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        tile = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)
        cv2.putText(
            tile,
            f"CH{channel + 1}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    grid = np.concatenate(
        [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)],
        axis=0,
    )
    _write_image(save_path, grid)


def _draw_gt_lines(image: np.ndarray, gt_lines: dict[str, Any]) -> None:
    """GTポリラインを画像へ描画する。"""
    for line_name, points in gt_lines.items():
        if points is None or len(points) < 2:
            continue
        vertices = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            image,
            [vertices],
            isClosed=False,
            color=LINE_COLORS.get(line_name, (255, 255, 255)),
            thickness=2,
        )


def _draw_pred_lines(image: np.ndarray, pred_lines: dict[str, Any]) -> None:
    """予測直線と重心を画像へ描画する。"""
    for line_name, line in pred_lines.items():
        if line is None or len(line.get("endpoints", [])) != 2:
            continue
        color = LINE_COLORS.get(line_name, (255, 255, 255))
        endpoint_1, endpoint_2 = line["endpoints"]
        cv2.line(
            image,
            tuple(int(round(value)) for value in endpoint_1),
            tuple(int(round(value)) for value in endpoint_2),
            color,
            2,
        )
        centroid = line.get("centroid")
        if centroid is not None:
            cv2.circle(
                image,
                tuple(int(round(value)) for value in centroid),
                3,
                color,
                -1,
            )


def draw_line_comparison(
    ct_image: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
) -> None:
    """GT線と予測線を横並びで保存する。"""
    gt_image = _base_image(ct_image)
    pred_image = _base_image(ct_image)
    _draw_gt_lines(gt_image, gt_lines)
    _draw_pred_lines(pred_image, pred_lines)
    cv2.putText(
        gt_image, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        pred_image, "Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    _write_image(save_path, np.concatenate([gt_image, pred_image], axis=1))


def draw_heatmap_with_lines(
    ct_image: np.ndarray,
    heatmaps: np.ndarray,
    pred_lines: dict[str, Any],
    gt_lines: dict[str, Any],
    save_path: Path,
    alpha: float = 0.6,
) -> None:
    """heatmap・予測線・GT線を3列で保存する。"""
    base = _base_image(ct_image)
    merged = np.clip(np.max(heatmaps, axis=0), 0, 1)
    heat_color = cv2.applyColorMap((merged * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_image = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)
    pred_image = base.copy()
    gt_image = base.copy()
    _draw_pred_lines(pred_image, pred_lines)
    _draw_gt_lines(gt_image, gt_lines)
    for image, label in (
        (heatmap_image, "Heatmap"),
        (pred_image, "Pred Lines"),
        (gt_image, "GT Lines"),
    ):
        cv2.putText(
            image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    _write_image(
        save_path,
        np.concatenate([heatmap_image, pred_image, gt_image], axis=1),
    )
