"""厳密平面の3パラメータに対する評価指標。

評価単位は椎体×面（＝平面1枚）である。重なり窓は先に集約する。
窓ごとの観測をそのまま数えると、同じスライスが10〜15回重複カウントされる。

`peak_dist` は削除した。教師がリッジ形状なので線上でargmaxが任意になり、
線の位置精度を測っていなかった。代わりに平面のオフセット誤差を使う。
"""

from __future__ import annotations

import math

import numpy as np


def angle_error_deg(
    predicted_doubled: np.ndarray,
    target_doubled: np.ndarray,
) -> np.ndarray:
    """doubled-angleから直線角度誤差を度単位で返す。"""
    dot = np.clip((predicted_doubled * target_doubled).sum(axis=-1), -1.0, 1.0)
    return np.asarray(0.5 * np.degrees(np.arccos(dot)))


def aligned_rho_error_px(
    predicted_normal: np.ndarray,
    predicted_rho: np.ndarray,
    target_normal: np.ndarray,
    target_rho: np.ndarray,
) -> np.ndarray:
    """法線の符号を揃えたうえでオフセット誤差を返す。

    符号不変な `min(|d|, |s|)` は使わない。予測が原点の反対側へ出た失敗を
    誤差0として報告してしまうため。
    """
    alignment = np.sign((predicted_normal * target_normal).sum(axis=-1))
    alignment = np.where(alignment == 0, 1.0, alignment)
    return np.asarray(np.abs(predicted_rho - alignment * target_rho))


def tilt_vector_error(
    predicted_tilt: np.ndarray,
    target_tilt: np.ndarray,
) -> np.ndarray:
    """傾きベクトル `v = k * n` の誤差を px/slice で返す。"""
    return np.asarray(np.linalg.norm(predicted_tilt - target_tilt, axis=-1))


def tilt_sign_correct(
    predicted_tilt: np.ndarray,
    target_tilt: np.ndarray,
) -> np.ndarray:
    """傾き方向が一致しているかを返す。ゼロ予測は不正解とする。"""
    return np.asarray((predicted_tilt * target_tilt).sum(axis=-1) > 0.0)


def plane_normal_error_deg(
    predicted_normal: np.ndarray,
    predicted_slope: np.ndarray,
    target_normal: np.ndarray,
    target_slope: np.ndarray,
) -> np.ndarray:
    """3D平面法線 `N = (n_x, n_y, -k) / |.|` の角度差を返す。"""
    predicted = np.concatenate([predicted_normal, -predicted_slope[..., None]], axis=-1)
    target = np.concatenate([target_normal, -target_slope[..., None]], axis=-1)
    predicted = predicted / np.linalg.norm(predicted, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True)
    dot = np.clip(np.abs((predicted * target).sum(axis=-1)), 0.0, 1.0)
    return np.asarray(np.degrees(np.arccos(dot)))


def collect_blob_ious(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> list[float]:
    """有効線だけで二値Blob IoUを収集する。閾値は呼び出し側が明示する。"""
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


def summarize_errors(values: list[float] | np.ndarray) -> dict[str, float]:
    """誤差列の平均・median・p90を返す。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def doubled_to_angle_deg(cosine: float, sine: float) -> float:
    """doubled-angleを0以上180度未満へ変換する。"""
    return float(math.degrees(0.5 * math.atan2(sine, cosine)) % 180.0)


def angle_difference_deg(first: float, second: float) -> float:
    """180度周期の角度差を返す。"""
    difference = abs(first - second) % 180.0
    return float(min(difference, 180.0 - difference))


def smoothness_metrics(
    centroids: np.ndarray,
    angles_deg: np.ndarray,
) -> dict[str, float]:
    """z方向の1階・2階差分を集計する（全高推論の健全性確認用）。"""
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
