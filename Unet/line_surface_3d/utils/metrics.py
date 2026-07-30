"""リボン予測の角度・重心・平滑性指標。"""

from __future__ import annotations

import math

import numpy as np
import torch


def collect_angle_errors(
    predicted_params: torch.Tensor,
    target_params: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    """line_onlyと同じ `(phi, rho)` 角度誤差を収集する。"""
    predicted_phi = predicted_params[..., 0]
    target_phi = target_params[..., 0]
    dot = torch.cos(predicted_phi) * torch.cos(target_phi) + torch.sin(
        predicted_phi
    ) * torch.sin(target_phi)
    errors = torch.rad2deg(torch.acos(torch.abs(dot).clamp(0.0, 1.0)))
    return [float(value) for value in errors[mask].detach().cpu().tolist()]


def collect_rho_errors(
    predicted_params: torch.Tensor,
    target_params: torch.Tensor,
    image_size: int,
    mask: torch.Tensor,
) -> list[float]:
    """line_onlyと同じ符号不変rho誤差をpx単位で収集する。"""
    predicted_rho = predicted_params[..., 1]
    target_rho = target_params[..., 1]
    normalized_error = torch.minimum(
        torch.abs(predicted_rho - target_rho),
        torch.abs(predicted_rho + target_rho),
    )
    errors = normalized_error * math.sqrt(image_size**2 + image_size**2)
    return [float(value) for value in errors[mask].detach().cpu().tolist()]


def collect_blob_ious(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.1,
) -> list[float]:
    """line_onlyと同じ二値Blob IoUを有効線だけ収集する。"""
    predicted_mask = prediction >= threshold
    target_mask = target >= threshold
    values: list[float] = []
    for batch_index, channel_index in np.argwhere(mask):
        intersection = int(
            (
                predicted_mask[batch_index, channel_index]
                & target_mask[batch_index, channel_index]
            ).sum()
        )
        union = int(
            (
                predicted_mask[batch_index, channel_index]
                | target_mask[batch_index, channel_index]
            ).sum()
        )
        values.append(intersection / union if union > 0 else 1.0)
    return values


def collect_angle_errors_deg(
    predicted_doubled: torch.Tensor,
    target_doubled: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    """doubled-angleから直線角度誤差を度単位で収集する。"""
    dot = (predicted_doubled * target_doubled).sum(dim=-1).clamp(-1.0, 1.0)
    errors = 0.5 * torch.rad2deg(torch.acos(dot))
    return [float(value) for value in errors[mask].detach().cpu().tolist()]


def collect_centroid_errors(
    predicted_centroid: torch.Tensor,
    target_centroid: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    """重心のEuclidean誤差を収集する。"""
    errors = torch.linalg.vector_norm(
        predicted_centroid - target_centroid,
        dim=-1,
    )
    return [float(value) for value in errors[mask].detach().cpu().tolist()]


def summarize_errors(values: list[float]) -> dict[str, float]:
    """誤差列の平均・median・p90を返す。"""
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def angle_difference_deg(first: float, second: float) -> float:
    """180度周期の角度差を返す。"""
    difference = abs(first - second) % 180.0
    return float(min(difference, 180.0 - difference))


def smoothness_metrics(
    centroids: np.ndarray,
    angles_deg: np.ndarray,
) -> dict[str, float]:
    """z方向の1階・2階差分を集計する。"""
    if len(centroids) < 2:
        return {
            "max_adjacent_centroid_px": float("nan"),
            "max_adjacent_angle_deg": float("nan"),
            "mean_second_centroid_px": float("nan"),
            "mean_second_angle_deg": float("nan"),
        }
    centroid_first = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    angle_first = np.asarray(
        [
            angle_difference_deg(float(angles_deg[index]), float(angles_deg[index + 1]))
            for index in range(len(angles_deg) - 1)
        ]
    )
    centroid_second = (
        np.linalg.norm(np.diff(centroids, n=2, axis=0), axis=1)
        if len(centroids) >= 3
        else np.asarray([])
    )
    unwrapped = np.unwrap(np.deg2rad(angles_deg) * 2.0) / 2.0
    angle_second = (
        np.abs(np.diff(np.rad2deg(unwrapped), n=2))
        if len(angles_deg) >= 3
        else np.asarray([])
    )
    return {
        "max_adjacent_centroid_px": float(np.max(centroid_first)),
        "max_adjacent_angle_deg": float(np.max(angle_first)),
        "mean_second_centroid_px": (
            float(np.mean(centroid_second)) if centroid_second.size else float("nan")
        ),
        "mean_second_angle_deg": (
            float(np.mean(angle_second)) if angle_second.size else float("nan")
        ),
    }


def doubled_to_angle_deg(cosine: float, sine: float) -> float:
    """doubled-angleを0以上180度未満へ変換する。"""
    return float(math.degrees(0.5 * math.atan2(sine, cosine)) % 180.0)
