"""
予測線からの領域分割生成・3D Dice 評価・スライス可視化

概要:
  1. VolumetricDiceAccumulator  : スライスをまたぐ 3D Dice 計算
  2. build_zprop                : 非外れ値予測スライスからスプライン伝播を構築
  3. predict_label_for_slice    : 指定スライスの予測ラベル画像を生成
  4. evaluate_vertebra          : 1 椎骨を全スライス評価し 3D Dice + per-slice を返す
  5. save_all_slice_overlays    : 全スライスの GT vs 予測オーバーレイ PNG を保存

外部依存:
  - data_preprocessing.learning_dataset.propagate_lines_z  (z 軸伝播)
  - data_preprocessing.segmentation_dataset.generate_region_mask (領域マスク)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing.learning_dataset.propagate_lines_z import (
    MaskGeometry,
    SliceState,
    build_smooth_trajectory,
    compute_mask_geometry,
    evaluate_trajectory,
    extract_slice_state,
    reconstruct_lines_from_state,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

# ─── 定数 ─────────────────────────────────────────────────────────────────
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
TARGET_SIZE = 224
MIN_MASK_AREA = 50
OUTLIER_THRESH_DEG = 10.0

# 領域ラベルの BGR 色（class 0-4）
_REGION_COLORS_BGR = (
    (0, 0, 0),       # 0: 背景
    (0, 200, 0),     # 1: 椎体 body
    (0, 0, 200),     # 2: 右椎間孔 right foramen
    (200, 0, 0),     # 3: 左椎間孔 left foramen
    (0, 200, 200),   # 4: 後方要素 posterior
)
_OVERLAY_ALPHA = 0.45


# ─── 3D Dice アキュムレータ ────────────────────────────────────────────────

@dataclass
class VolumetricDiceAccumulator:
    """
    椎骨ボリューム単位の 3D Dice を計算するアキュムレータ。

    複数スライスの予測/GT ピクセル数を積算し、
    finalize() で class ごとの volumetric Dice を返す。

    例::
        acc = VolumetricDiceAccumulator()
        for pred_lbl, gt_lbl in slices:
            acc.update(pred_lbl, gt_lbl)
        dice = acc.finalize()  # {"mean": ..., "class_1": ..., ...}
    """

    _inter: dict[int, int] = field(default_factory=lambda: {c: 0 for c in range(1, 5)})
    _pred:  dict[int, int] = field(default_factory=lambda: {c: 0 for c in range(1, 5)})
    _gt:    dict[int, int] = field(default_factory=lambda: {c: 0 for c in range(1, 5)})

    def update(self, pred_lbl: np.ndarray, gt_lbl: np.ndarray) -> None:
        """1 スライス分のピクセル数を加算する。"""
        for c in range(1, 5):
            pm = pred_lbl == c
            tm = gt_lbl == c
            self._inter[c] += int(np.logical_and(pm, tm).sum())
            self._pred[c]  += int(pm.sum())
            self._gt[c]    += int(tm.sum())

    def finalize(self) -> dict[str, float]:
        """
        累積カウントから 3D Dice を計算する。

        GT も予測も空のクラスは NaN（評価対象外）とする。
        mean は有効クラスのみで平均する。
        """
        scores: dict[str, float] = {}
        valid: list[float] = []
        for c in range(1, 5):
            denom = self._pred[c] + self._gt[c]
            if denom == 0:
                scores[f"class_{c}"] = float("nan")
            else:
                d = 2.0 * self._inter[c] / denom
                scores[f"class_{c}"] = d
                valid.append(d)
        scores["mean"] = float(np.mean(valid)) if valid else float("nan")
        return scores


# ─── 予測データのパース ────────────────────────────────────────────────────

def is_valid_pred(pred_data: dict) -> bool:
    """全 4 線の endpoints が有効か確認する。"""
    pred_lines = pred_data.get("pred_lines", {})
    for k in LINE_KEYS:
        line = pred_lines.get(k)
        if not isinstance(line, dict):
            return False
        eps = line.get("endpoints")
        if not isinstance(eps, list) or len(eps) < 2:
            return False
        if eps[0] is None or eps[1] is None:
            return False
    return True


def max_angle_error(pred_data: dict) -> float:
    """スライスの全線の最大角度誤差を返す。metrics がなければ inf。"""
    metrics = pred_data.get("metrics", {})
    errors = [
        metrics[k]["angle_error_deg"]
        for k in LINE_KEYS
        if k in metrics
        and "angle_error_deg" in metrics[k]
        and metrics[k]["angle_error_deg"] is not None
    ]
    return max(errors) if errors else float("inf")


def pred_to_polylines(pred_data: dict) -> dict[str, list[list[float]]]:
    """PRED_lines.json の endpoints を 2 点折れ線形式に変換する。"""
    pred_lines = pred_data["pred_lines"]
    return {
        k: [
            list(pred_lines[k]["endpoints"][0]),
            list(pred_lines[k]["endpoints"][1]),
        ]
        for k in LINE_KEYS
    }


# ─── z 伝播の構築 ────────────────────────────────────────────────────────

@dataclass
class ZPropResult:
    """build_zprop の結果を保持するデータクラス。"""
    spline: Any
    extrap_params: dict
    geometries: dict[int, MaskGeometry]
    bin_masks: dict[int, np.ndarray]
    mask_areas: dict[int, float]
    anchor_slices: dict[int, dict]
    area_lo: float
    area_hi: float
    all_z: list[int]
    fallback_used: bool = False


def build_zprop(
    bin_masks: dict[int, np.ndarray],
    slice_preds: dict[int, dict],
    outlier_thresh_deg: float = OUTLIER_THRESH_DEG,
) -> ZPropResult | None:
    """
    非外れ値予測スライスをアンカーとして z 伝播スプラインを構築する。

    引数:
        bin_masks : {slice_idx: binary_mask (H,W)}
        slice_preds: {slice_idx: PRED_lines.json の dict}
        outlier_thresh_deg: 外れ値判定の角度誤差閾値（度）

    戻り値:
        ZPropResult、または構築不可能な場合 None
    """
    # 形状情報を計算
    geometries: dict[int, MaskGeometry] = {}
    mask_areas: dict[int, float] = {}
    for z, bm in bin_masks.items():
        if np.sum(bm > 0) >= MIN_MASK_AREA:
            geometries[z] = compute_mask_geometry(bm)
            mask_areas[z] = float(np.sum(bm > 0))

    if not geometries:
        return None

    # 非外れ値スライスをアンカーに選択
    anchor_slices = {
        z: p for z, p in slice_preds.items()
        if is_valid_pred(p) and max_angle_error(p) < outlier_thresh_deg and z in geometries
    }
    fallback = False
    if len(anchor_slices) < 2:
        # フォールバック: 外れ値も含めて全有効予測を使用
        anchor_slices = {
            z: p for z, p in slice_preds.items()
            if is_valid_pred(p) and z in geometries
        }
        fallback = True

    if len(anchor_slices) < 2:
        return None

    # アンカーから SliceState を抽出
    anchor_states: dict[int, SliceState] = {}
    for z, pred in anchor_slices.items():
        try:
            anchor_states[z] = extract_slice_state(pred_to_polylines(pred), geometries[z])
        except Exception:
            pass

    if len(anchor_states) < 2:
        return None

    try:
        spline, extrap_params = build_smooth_trajectory(anchor_states)
    except Exception:
        return None

    area_lo = mask_areas.get(int(extrap_params["z_lo"]), 1.0)
    area_hi = mask_areas.get(int(extrap_params["z_hi"]), 1.0)
    all_z = sorted(bin_masks.keys())

    return ZPropResult(
        spline=spline,
        extrap_params=extrap_params,
        geometries=geometries,
        bin_masks=bin_masks,
        mask_areas=mask_areas,
        anchor_slices=anchor_slices,
        area_lo=area_lo,
        area_hi=area_hi,
        all_z=all_z,
        fallback_used=fallback,
    )


# ─── スライス単位の予測生成 ───────────────────────────────────────────────

def predict_label_for_slice(
    prop: ZPropResult,
    z: int,
) -> tuple[np.ndarray | None, str]:
    """
    z スライスの予測領域ラベル画像（0-4）と provenance を返す。

    引数:
        prop : build_zprop の結果
        z    : スライスインデックス

    戻り値:
        (label_image (H,W) uint8, provenance str)
        label_image が None のとき生成失敗。
    """
    if z not in prop.bin_masks:
        return None, "no_mask"

    if z in prop.anchor_slices:
        polylines = pred_to_polylines(prop.anchor_slices[z])
        provenance = "anchor"
    else:
        try:
            state_vec, prov, _ = evaluate_trajectory(
                prop.spline, prop.extrap_params, z,
                prop.mask_areas.get(z, 1.0), prop.area_lo, prop.area_hi,
            )
            polylines = reconstruct_lines_from_state(
                state_vec, prop.geometries[z], TARGET_SIZE
            )
            provenance = prov
        except Exception:
            return None, "traj_failed"

    try:
        seg, _ = generate_region_mask(
            line_1=polylines["line_1"],
            line_2=polylines["line_2"],
            line_3=polylines["line_3"],
            line_4=polylines["line_4"],
            vertebra_mask=prop.bin_masks[z],
        )
        return np.argmax(seg, axis=0).astype(np.uint8), provenance
    except Exception:
        return None, provenance


# ─── 椎骨単位の評価 ───────────────────────────────────────────────────────

@dataclass
class VertebralEvalResult:
    """evaluate_vertebra の評価結果。"""
    sample: str
    vertebra: str
    volumetric_dice: dict[str, float]       # 3D Dice（全由来）
    volumetric_dice_extrap: dict[str, float]  # 3D Dice（外挿のみ）
    per_slice: list[dict]                   # スライス単位の詳細
    anchor_count: int
    outlier_count: int
    fallback_used: bool
    error: str | None = None


def evaluate_vertebra(
    sample: str,
    vertebra: str,
    bin_masks: dict[int, np.ndarray],
    gt_masks: dict[int, np.ndarray],
    slice_preds: dict[int, dict],
    outlier_thresh_deg: float = OUTLIER_THRESH_DEG,
) -> VertebralEvalResult:
    """
    1 椎骨の全 z スライスを評価し 3D Dice と per-slice 結果を返す。

    引数:
        sample, vertebra : 識別用
        bin_masks  : {z: binary vertebra mask}
        gt_masks   : {z: GT 領域ラベル画像 (0-4)}
        slice_preds: {z: PRED_lines.json dict}
        outlier_thresh_deg: 外れ値閾値

    戻り値:
        VertebralEvalResult
    """
    # 外れ値カウント
    outlier_count = sum(
        1 for z, p in slice_preds.items()
        if not is_valid_pred(p) or max_angle_error(p) >= outlier_thresh_deg
    )

    prop = build_zprop(bin_masks, slice_preds, outlier_thresh_deg)
    if prop is None:
        return VertebralEvalResult(
            sample=sample, vertebra=vertebra,
            volumetric_dice={}, volumetric_dice_extrap={},
            per_slice=[], anchor_count=0,
            outlier_count=outlier_count, fallback_used=False,
            error="propagation build failed",
        )

    acc_all = VolumetricDiceAccumulator()
    acc_ext = VolumetricDiceAccumulator()
    per_slice: list[dict] = []

    for z in prop.all_z:
        gt = gt_masks.get(z)
        if gt is None:
            continue
        pred_lbl, prov = predict_label_for_slice(prop, z)
        if pred_lbl is None:
            per_slice.append({"slice": z, "provenance": prov, "error": "predict_failed"})
            continue

        acc_all.update(pred_lbl, gt)
        if prov == "extrapolated":
            acc_ext.update(pred_lbl, gt)

        # per-slice 2D Dice（参考値）
        def _dice2d(c: int) -> float:
            pm, tm = pred_lbl == c, gt == c
            d = int(pm.sum() + tm.sum())
            return 2.0 * int(np.logical_and(pm, tm).sum()) / d if d > 0 else 1.0

        per_slice.append({
            "slice": z,
            "provenance": prov,
            "dice_2d_mean": float(np.mean([_dice2d(c) for c in range(1, 5)])),
            "dice_2d_class_1": _dice2d(1),
            "dice_2d_class_2": _dice2d(2),
            "dice_2d_class_3": _dice2d(3),
            "dice_2d_class_4": _dice2d(4),
        })

    return VertebralEvalResult(
        sample=sample,
        vertebra=vertebra,
        volumetric_dice=acc_all.finalize(),
        volumetric_dice_extrap=acc_ext.finalize(),
        per_slice=per_slice,
        anchor_count=len(prop.anchor_slices),
        outlier_count=outlier_count,
        fallback_used=prop.fallback_used,
    )


# ─── 可視化 ───────────────────────────────────────────────────────────────

def _make_region_overlay(ct_gray: np.ndarray, label_img: np.ndarray) -> np.ndarray:
    """CT グレー画像に領域ラベルのカラーオーバーレイを重ねた BGR 画像を返す。"""
    ct_bgr = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_layer = np.zeros_like(ct_bgr)
    for lbl, bgr in enumerate(_REGION_COLORS_BGR):
        mask = label_img == lbl
        if mask.any():
            color_layer[mask] = bgr
    fg = (label_img > 0).astype(np.float32)[..., None]
    blended = ct_bgr * (1.0 - fg * _OVERLAY_ALPHA) + color_layer * _OVERLAY_ALPHA
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_all_slice_overlays(
    sample: str,
    vertebra: str,
    bin_masks: dict[int, np.ndarray],
    gt_masks: dict[int, np.ndarray],
    ct_images: dict[int, np.ndarray],
    slice_preds: dict[int, dict],
    out_dir: Path,
    outlier_thresh_deg: float = OUTLIER_THRESH_DEG,
) -> list[Path]:
    """
    全スライスの「GT / 予測」比較オーバーレイ PNG を保存する。

    出力: out_dir/slice_XXX.png
      左 : GT 領域マスクを CT に重ねた画像
      右 : 予測領域マスクを CT に重ねた画像

    引数:
        ct_images : {z: grayscale (H,W) uint8}

    戻り値:
        保存した PNG パスのリスト
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prop = build_zprop(bin_masks, slice_preds, outlier_thresh_deg)
    if prop is None:
        return []

    # アンカースライス集合（ラベル表示用）
    anchor_z = set(prop.anchor_slices.keys())
    pred_z = set(slice_preds.keys())

    saved: list[Path] = []
    for z in prop.all_z:
        ct = ct_images.get(z)
        gt = gt_masks.get(z)
        if ct is None or gt is None:
            continue

        pred_lbl, prov = predict_label_for_slice(prop, z)

        gt_overlay = _make_region_overlay(ct, gt)
        if pred_lbl is not None:
            pred_overlay = _make_region_overlay(ct, pred_lbl)
        else:
            pred_overlay = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)

        # 左右に並べる
        h, w = gt_overlay.shape[:2]
        canvas = np.zeros((h + 32, w * 2, 3), dtype=np.uint8)
        canvas[32:, :w] = gt_overlay
        canvas[32:, w:] = pred_overlay if pred_lbl is not None else cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)

        # ラベルテキスト
        is_anchor = z in anchor_z
        is_pred = z in pred_z
        prov_label = "anchor" if is_anchor else prov
        if is_pred and not is_anchor:
            prov_label = f"{prov}(外れ値)"

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        cv2.putText(canvas, f"GT  z={z}", (4, 20), font, scale, (220, 220, 220), thick, cv2.LINE_AA)
        cv2.putText(canvas, f"PRED [{prov_label}]", (w + 4, 20), font, scale, (220, 220, 220), thick, cv2.LINE_AA)

        out_path = out_dir / f"slice_{z:03d}.png"
        cv2.imwrite(str(out_path), canvas)
        saved.append(out_path)

    return saved
