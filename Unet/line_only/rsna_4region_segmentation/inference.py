"""アンサンブル推論ユーティリティ。

全スクリプトで重複していた単一スライス / 複数スライス推論を統一する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from line_only.src.model import VERTEBRA_TO_IDX, TinyUNet
from line_only.utils.detection import detect_line_moments, moments_to_phi_rho

from .constants import (
    DEFAULT_HEATMAP_THRESHOLD,
    FALLBACK_LINE_LENGTH_PX,
    IMAGE_SIZE,
    LINE_KEYS,
    N_SEG_PLANES,
)


@dataclass(frozen=True)
class PredictedLine:
    """1 本の予測線の SDF パラメータとヒートマップ重心。"""

    phi: float
    rho: float
    centroid: tuple[float, float]
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    angle_deg: float
    line_length_px: float


def _ensemble_heatmaps(
    models: list[TinyUNet],
    model_input: torch.Tensor,
    vertebra_idx: torch.Tensor,
) -> np.ndarray:
    """アンサンブル平均ヒートマップを返す。shape は (batch, 4, H, W)。"""
    heatmap_sum: torch.Tensor | None = None
    for model in models:
        output = torch.sigmoid(model(model_input, vertebra_idx))
        heatmap_sum = output if heatmap_sum is None else heatmap_sum + output
    return (heatmap_sum / len(models)).cpu().numpy()  # type: ignore[union-attr]


def _detect_one_channel(
    heatmap_2d: np.ndarray,
    line_key: str,
    avg_lengths: dict[str, float],
    threshold: dict[str, object] | None = None,
) -> PredictedLine | None:
    """1 チャンネルのヒートマップから線を検出する。"""
    if threshold is None:
        threshold = dict(DEFAULT_HEATMAP_THRESHOLD)
    length_px = avg_lengths.get(line_key, FALLBACK_LINE_LENGTH_PX)
    det = detect_line_moments(
        heatmap_2d,
        length_px=length_px,
        extend_ratio=1.0,
        clip=False,
        threshold=threshold,
    )
    if det is None:
        return None
    phi, rho = moments_to_phi_rho(det, IMAGE_SIZE)
    return PredictedLine(
        phi=phi,
        rho=rho,
        centroid=(float(det["centroid"][0]), float(det["centroid"][1])),
        endpoints=tuple(det["endpoints"]),  # type: ignore[arg-type]
        angle_deg=float(det["angle_deg"]),
        line_length_px=length_px,
    )


@torch.no_grad()
def predict_single_slice(
    models: list[TinyUNet],
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    vertebra: str,
    device: torch.device,
    avg_lengths: dict[str, float],
) -> tuple[np.ndarray, dict[str, PredictedLine | None]]:
    """1 枚の CT+マスクから 4 本線を推論する。

    戻り値:
        (heatmaps (4, H, W), {line_key: PredictedLine | None})
    """
    x = (
        torch.from_numpy(np.stack([ct_slice, mask_slice], axis=0))
        .unsqueeze(0)
        .to(device)
    )
    vidx = torch.tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0)],
        device=device,
        dtype=torch.long,
    )
    heatmaps = _ensemble_heatmaps(models, x, vidx)[0]  # (4, H, W)

    lines: dict[str, PredictedLine | None] = {}
    for ch, line_key in enumerate(LINE_KEYS):
        lines[line_key] = _detect_one_channel(heatmaps[ch], line_key, avg_lengths)
    return heatmaps, lines


@torch.no_grad()
def predict_5planes(
    models: list[TinyUNet],
    seg_ct: np.ndarray,
    seg_mask: np.ndarray,
    vertebra: str,
    device: torch.device,
    avg_lengths: dict[str, float],
) -> list[dict[str, PredictedLine | None]]:
    """5 枚の seg_ct からバッチ推論で各線のパラメータを返す。

    戻り値:
        5 要素リスト。各要素は {line_key: PredictedLine | None}。
    """
    vertebra_index = VERTEBRA_TO_IDX.get(vertebra, 0)
    vidx = torch.full(
        (N_SEG_PLANES,),
        vertebra_index,
        device=device,
        dtype=torch.long,
    )
    ct_float = seg_ct.astype(np.float32) / 255.0
    mask_float = seg_mask.astype(np.float32)
    model_input = torch.from_numpy(np.stack([ct_float, mask_float], axis=1)).to(device)

    heatmaps = _ensemble_heatmaps(models, model_input, vidx)  # (5, 4, H, W)

    results: list[dict[str, PredictedLine | None]] = []
    for plane_index in range(N_SEG_PLANES):
        plane_result: dict[str, PredictedLine | None] = {}
        for ch, line_key in enumerate(LINE_KEYS):
            plane_result[line_key] = _detect_one_channel(
                heatmaps[plane_index, ch],
                line_key,
                avg_lengths,
            )
        results.append(plane_result)
    return results
