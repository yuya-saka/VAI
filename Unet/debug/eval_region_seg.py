"""
予測線から領域分割を生成し、GT真値マスクと精度評価する。

処理フロー:
  1. sig4.0_ALL(CC適用)/vis/fold*/test_lines_reeval/ から予測線を収集
  2. 外れ値スライスを判定 (max angle_error_deg >= OUTLIER_THRESH)
  3. 非外れ値スライスをアンカーとして z 軸伝播
  4. 全 z スライスの領域マスクを生成
  5. dataset_zprop の GT マスクと Dice スコアで評価
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
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
    validate_region_mask,
)

# ─── 定数 ─────────────────────────────────────────────────────────────────
EXPERIMENT = "sig4.0_ALL(CC適用)"
FOLDS = [0, 1, 2, 3, 4]
LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")

# 外れ値判定閾値（line 単位の最大誤差がこれ以上のスライスを外れ値とする）
OUTLIER_THRESH_DEG = 10.0

TARGET_SIZE = 224
MIN_MASK_AREA = 50


# ─── パス定義 ──────────────────────────────────────────────────────────────

def get_output_base() -> Path:
    return ROOT_DIR / "Unet" / "outputs" / "line_20260616" / EXPERIMENT


def get_pred_dir(fold: int) -> Path:
    return get_output_base() / "vis" / f"fold{fold}" / "test_lines_reeval"


def get_zprop_dir(sample: str, vertebra: str) -> Path:
    return ROOT_DIR / "data" / "dataset_zprop" / sample / vertebra


# ─── 予測データの収集 ─────────────────────────────────────────────────────

def load_all_predictions(fold: int) -> dict[tuple[str, str], dict[int, dict]]:
    """fold の test_lines_reeval を読み込み、(sample, vertebra) -> {slice_idx: data} で返す。"""
    pred_dir = get_pred_dir(fold)
    groups: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)

    for json_path in sorted(pred_dir.glob("*_PRED_lines.json")):
        m = re.match(r"(sample[\w.]+)_(C\d+)_slice(\d+)_PRED_lines", json_path.stem)
        if not m:
            continue
        sample, vertebra, slice_str = m.group(1), m.group(2), m.group(3)
        slice_idx = int(slice_str)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        groups[(sample, vertebra)][slice_idx] = data

    return dict(groups)


def is_valid_prediction(pred_data: dict) -> bool:
    """全 4 線が検出されていて endpoints が有効か確認する。"""
    pred_lines = pred_data.get("pred_lines", {})
    for k in LINE_KEYS:
        line = pred_lines.get(k)
        if not isinstance(line, dict):
            return False
        endpoints = line.get("endpoints")
        if not isinstance(endpoints, list) or len(endpoints) < 2:
            return False
    return True


def get_max_angle_error(pred_data: dict) -> float:
    """スライスの全線の最大角度誤差を返す。metrics がなければ inf を返す。"""
    metrics = pred_data.get("metrics", {})
    errors = [
        metrics[k]["angle_error_deg"]
        for k in LINE_KEYS
        if k in metrics
        and "angle_error_deg" in metrics[k]
        and metrics[k]["angle_error_deg"] is not None
    ]
    return max(errors) if errors else float("inf")


def pred_lines_to_polylines(pred_data: dict) -> dict[str, list[list[float]]]:
    """PRED_lines.json の pred_lines から endpoints を 2 点折れ線に変換する。"""
    pred_lines = pred_data["pred_lines"]
    return {
        line_key: [
            list(pred_lines[line_key]["endpoints"][0]),
            list(pred_lines[line_key]["endpoints"][1]),
        ]
        for line_key in LINE_KEYS
    }


# ─── マスクの読み込み ─────────────────────────────────────────────────────

def load_binary_mask(sample: str, vertebra: str, slice_idx: int) -> np.ndarray | None:
    """dataset_zprop から二値椎骨マスクを読み込む。なければ None を返す。"""
    path = get_zprop_dir(sample, vertebra) / "masks" / f"slice_{slice_idx:03d}.png"
    if not path.exists():
        return None
    return np.array(Image.open(path))


def load_gt_mask(sample: str, vertebra: str, slice_idx: int) -> np.ndarray | None:
    """dataset_zprop から GT 領域マスク（0-4 のラベル画像）を読み込む。なければ None を返す。"""
    path = get_zprop_dir(sample, vertebra) / "gt_masks" / f"slice_{slice_idx:03d}.png"
    if not path.exists():
        return None
    return np.array(Image.open(path))


def get_valid_z_range(sample: str, vertebra: str) -> tuple[int, int] | None:
    """generation_report.json から valid_z_range を読み込む。"""
    report_path = get_zprop_dir(sample, vertebra) / "generation_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rng = report.get("valid_z_range")
    if not rng or len(rng) != 2:
        return None
    return int(rng[0]), int(rng[1])


# ─── Dice スコア計算 ───────────────────────────────────────────────────────

def class_dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """指定クラスの Dice スコアを計算する。"""
    pm = pred == class_id
    tm = target == class_id
    denom = int(pm.sum() + tm.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pm, tm).sum()) / denom


def compute_dice_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """クラス 1-4 の Dice と平均を返す。"""
    scores = {f"class_{c}": class_dice(pred_mask, gt_mask, c) for c in range(1, 5)}
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def init_volume_counts() -> dict[int, dict[str, int]]:
    """クラスごとの 3D Dice 用カウンタ（交差・予測・GT のピクセル数）を初期化する。"""
    return {c: {"inter": 0, "pred": 0, "gt": 0} for c in range(1, 5)}


def accumulate_class_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    counts: dict[int, dict[str, int]],
) -> None:
    """1 スライス分のクラス別ピクセル数を 3D カウンタへ加算する。"""
    for c in range(1, 5):
        pm = pred_mask == c
        tm = gt_mask == c
        counts[c]["inter"] += int(np.logical_and(pm, tm).sum())
        counts[c]["pred"] += int(pm.sum())
        counts[c]["gt"] += int(tm.sum())


def volumetric_dice_from_counts(counts: dict[int, dict[str, int]]) -> dict[str, float]:
    """累積カウントから椎骨単位の 3D Dice（クラス別 + 平均）を計算する。

    GT も予測も全スライスで空のクラスは評価対象外（NaN）とする。
    """
    scores: dict[str, float] = {}
    valid_vals: list[float] = []
    for c in range(1, 5):
        denom = counts[c]["pred"] + counts[c]["gt"]
        if denom == 0:
            scores[f"class_{c}"] = float("nan")
            continue
        dice = 2.0 * counts[c]["inter"] / denom
        scores[f"class_{c}"] = dice
        valid_vals.append(dice)
    scores["mean"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
    return scores


# ─── 1 椎骨の評価 ─────────────────────────────────────────────────────────

def evaluate_vertebra(
    sample: str,
    vertebra: str,
    slice_preds: dict[int, dict],
    outlier_thresh_deg: float = OUTLIER_THRESH_DEG,
) -> dict[str, Any]:
    """
    1 椎骨の予測線から領域分割を生成し GT と比較する。

    返り値:
      - direct: 予測スライスを直接評価した Dice
      - propagated: z 伝播後の全スライスを評価した Dice
    """
    z_range = get_valid_z_range(sample, vertebra)
    if z_range is None:
        return {"error": "valid_z_range not found"}

    z_lo, z_hi = z_range
    all_z = list(range(z_lo, z_hi + 1))

    # 全 z スライスのマスクと形状を事前ロード
    bin_masks: dict[int, np.ndarray] = {}
    geometries: dict[int, MaskGeometry] = {}
    mask_areas: dict[int, float] = {}
    for z in all_z:
        bm = load_binary_mask(sample, vertebra, z)
        if bm is not None and np.sum(bm > 0) >= MIN_MASK_AREA:
            bin_masks[z] = bm
            geometries[z] = compute_mask_geometry(bm)
            mask_areas[z] = float(np.sum(bm > 0))

    if not geometries:
        return {"error": "no valid binary masks"}

    # 予測スライスの分類（外れ値 / 非外れ値）
    valid_preds: dict[int, dict] = {}
    outlier_slices: list[int] = []
    for z, pred_data in slice_preds.items():
        if not is_valid_prediction(pred_data):
            outlier_slices.append(z)
            continue
        max_err = get_max_angle_error(pred_data)
        if max_err >= outlier_thresh_deg:
            outlier_slices.append(z)
        else:
            valid_preds[z] = pred_data

    # ─ Direct 評価（予測スライスを直接マスク生成して評価）
    direct_results: list[dict] = []
    for z, pred_data in slice_preds.items():
        if not is_valid_prediction(pred_data) or z not in bin_masks:
            continue
        gt = load_gt_mask(sample, vertebra, z)
        if gt is None:
            continue
        polylines = pred_lines_to_polylines(pred_data)
        try:
            seg, _ = generate_region_mask(
                line_1=polylines["line_1"],
                line_2=polylines["line_2"],
                line_3=polylines["line_3"],
                line_4=polylines["line_4"],
                vertebra_mask=bin_masks[z],
            )
            pred_lbl = np.argmax(seg, axis=0).astype(np.uint8)
        except Exception as e:
            direct_results.append({"slice": z, "error": str(e)})
            continue
        dice = compute_dice_scores(pred_lbl, gt)
        is_outlier = z in outlier_slices
        max_err = get_max_angle_error(pred_data)
        direct_results.append({
            "slice": z,
            "is_outlier": is_outlier,
            "max_angle_error_deg": max_err,
            **dice,
        })

    # ─ z 伝播評価
    # アンカー = 非外れ値予測スライス（かつ形状計算可能なスライス）
    anchor_slices = {z: pred_data for z, pred_data in valid_preds.items() if z in geometries}

    prop_results: list[dict] = []
    prop_error: str | None = None
    # 3D (volumetric) Dice 用カウンタ（全由来 / 外挿のみ）
    vol_counts_all = init_volume_counts()
    vol_counts_extrap = init_volume_counts()

    if len(anchor_slices) < 2:
        # アンカー不足 → 外れ値も含めて全予測スライスをアンカーとしてフォールバック
        anchor_slices = {
            z: pred_data
            for z, pred_data in slice_preds.items()
            if is_valid_prediction(pred_data) and z in geometries
        }
        prop_error = f"only {len(valid_preds)} non-outlier anchor(s); using all {len(anchor_slices)} predicted slices"

    if len(anchor_slices) < 2:
        prop_error = f"insufficient anchors ({len(anchor_slices)}); skipping propagation"
    else:
        # アンカーから SliceState を抽出してスプライン構築
        anchor_states: dict[int, SliceState] = {}
        for z, pred_data in anchor_slices.items():
            polylines = pred_lines_to_polylines(pred_data)
            try:
                state = extract_slice_state(polylines, geometries[z])
                anchor_states[z] = state
            except Exception as e:
                prop_error = (prop_error or "") + f" | z={z} state extract failed: {e}"

        if len(anchor_states) >= 2:
            try:
                spline, extrap_params = build_smooth_trajectory(anchor_states)
                area_at_lo = mask_areas.get(int(extrap_params["z_lo"]), 1.0)
                area_at_hi = mask_areas.get(int(extrap_params["z_hi"]), 1.0)
            except Exception as e:
                prop_error = (prop_error or "") + f" | spline build failed: {e}"
                anchor_states = {}

            if anchor_states:
                # 全 z スライスに適用
                for z in all_z:
                    if z not in bin_masks:
                        continue
                    gt = load_gt_mask(sample, vertebra, z)
                    if gt is None:
                        continue

                    # スライスの由来を決定
                    if z in anchor_slices:
                        # 非外れ値アンカー: 予測線を直接使用
                        polylines = pred_lines_to_polylines(anchor_slices[z])
                        provenance = "anchor"
                    else:
                        # 外れ値または非テストスライス: スプラインから復元
                        try:
                            state_vec, prov, _ = evaluate_trajectory(
                                spline, extrap_params, z,
                                mask_areas.get(z, 1.0), area_at_lo, area_at_hi,
                            )
                            lines_out = reconstruct_lines_from_state(
                                state_vec, geometries[z], TARGET_SIZE
                            )
                            polylines = lines_out
                            provenance = prov
                        except Exception as e:
                            prop_results.append({"slice": z, "provenance": "failed", "error": str(e)})
                            continue

                    try:
                        seg, _ = generate_region_mask(
                            line_1=polylines["line_1"],
                            line_2=polylines["line_2"],
                            line_3=polylines["line_3"],
                            line_4=polylines["line_4"],
                            vertebra_mask=bin_masks[z],
                        )
                        pred_lbl = np.argmax(seg, axis=0).astype(np.uint8)
                    except Exception as e:
                        prop_results.append({"slice": z, "provenance": provenance, "error": str(e)})
                        continue

                    dice = compute_dice_scores(pred_lbl, gt)
                    prop_results.append({
                        "slice": z,
                        "provenance": provenance,
                        **dice,
                    })
                    # 3D Dice 用にピクセル数を累積
                    accumulate_class_counts(pred_lbl, gt, vol_counts_all)
                    if provenance == "extrapolated":
                        accumulate_class_counts(pred_lbl, gt, vol_counts_extrap)

    return {
        "sample": sample,
        "vertebra": vertebra,
        "z_range": [z_lo, z_hi],
        "pred_slice_count": len(slice_preds),
        "anchor_count": len(anchor_slices),
        "outlier_count": len(outlier_slices),
        "outlier_slices": sorted(outlier_slices),
        "direct": direct_results,
        "propagated": prop_results,
        "prop_error": prop_error,
        # 椎骨単位 3D Dice（全スライスのピクセルを積算した volumetric Dice）
        "volumetric_dice_all": volumetric_dice_from_counts(vol_counts_all),
        "volumetric_dice_extrap": volumetric_dice_from_counts(vol_counts_extrap),
        "volume_counts_all": vol_counts_all,
    }


# ─── 統計まとめ ────────────────────────────────────────────────────────────

def summarize_dice_list(records: list[dict], key: str = "mean") -> dict[str, float]:
    """Dice スコアリストを平均・中央値・p5 で要約する。"""
    vals = [r[key] for r in records if key in r and not isinstance(r.get("error"), str)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p5": float("nan"), "n": 0}
    arr = np.array(vals)
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p5": round(float(np.percentile(arr, 5)), 6),
        "n": len(vals),
    }


def build_summary(all_vertebra_results: list[dict]) -> dict[str, Any]:
    """全椎骨の評価結果を集計する。"""
    # direct 評価（外れ値 / 非外れ値 別）
    direct_non_outlier: list[dict] = []
    direct_outlier: list[dict] = []
    prop_anchor: list[dict] = []
    prop_interp: list[dict] = []
    prop_extrap: list[dict] = []

    total_pred = 0
    total_anchor = 0
    total_outlier = 0

    # 椎骨単位 3D Dice の集約用
    vol_per_vertebra: list[dict] = []      # 椎骨ごとの全由来 volumetric Dice
    vol_extrap_per_vertebra: list[dict] = []  # 椎骨ごとの外挿のみ volumetric Dice
    pooled_counts = init_volume_counts()   # 全椎骨をプールした grand-total カウント

    for r in all_vertebra_results:
        if "error" in r:
            continue
        total_pred += r.get("pred_slice_count", 0)
        total_anchor += r.get("anchor_count", 0)
        total_outlier += r.get("outlier_count", 0)

        if "volumetric_dice_all" in r:
            vol_per_vertebra.append(r["volumetric_dice_all"])
        if "volumetric_dice_extrap" in r:
            vol_extrap_per_vertebra.append(r["volumetric_dice_extrap"])
        for c in range(1, 5):
            vc = r.get("volume_counts_all", {}).get(c) or r.get("volume_counts_all", {}).get(str(c))
            if vc:
                for k in ("inter", "pred", "gt"):
                    pooled_counts[c][k] += int(vc[k])

        for d in r.get("direct", []):
            if "error" in d:
                continue
            if d.get("is_outlier"):
                direct_outlier.append(d)
            else:
                direct_non_outlier.append(d)

        for p in r.get("propagated", []):
            if "error" in p:
                continue
            prov = p.get("provenance", "")
            if prov == "anchor":
                prop_anchor.append(p)
            elif prov == "interpolated":
                prop_interp.append(p)
            elif prov == "extrapolated":
                prop_extrap.append(p)

    all_prop = prop_anchor + prop_interp + prop_extrap

    return {
        "total_vertebrae": len(all_vertebra_results),
        # ─ 3D (volumetric) Dice ─
        "volumetric_3d_per_vertebra": {
            "n": len(vol_per_vertebra),
            "dice_mean": summarize_dice_list(vol_per_vertebra, "mean"),
            "dice_class_1": summarize_dice_list(vol_per_vertebra, "class_1"),
            "dice_class_2": summarize_dice_list(vol_per_vertebra, "class_2"),
            "dice_class_3": summarize_dice_list(vol_per_vertebra, "class_3"),
            "dice_class_4": summarize_dice_list(vol_per_vertebra, "class_4"),
        },
        "volumetric_3d_extrap_per_vertebra": {
            "n": len(vol_extrap_per_vertebra),
            "dice_mean": summarize_dice_list(vol_extrap_per_vertebra, "mean"),
            "dice_class_1": summarize_dice_list(vol_extrap_per_vertebra, "class_1"),
            "dice_class_2": summarize_dice_list(vol_extrap_per_vertebra, "class_2"),
            "dice_class_3": summarize_dice_list(vol_extrap_per_vertebra, "class_3"),
            "dice_class_4": summarize_dice_list(vol_extrap_per_vertebra, "class_4"),
        },
        "volumetric_3d_pooled": volumetric_dice_from_counts(pooled_counts),
        "total_pred_slices": total_pred,
        "total_anchor_slices": total_anchor,
        "total_outlier_slices": total_outlier,
        "outlier_fraction": round(total_outlier / max(total_pred, 1), 4),
        "direct_non_outlier": {
            "n": len(direct_non_outlier),
            "dice_mean": summarize_dice_list(direct_non_outlier, "mean"),
            "dice_class_1": summarize_dice_list(direct_non_outlier, "class_1"),
            "dice_class_2": summarize_dice_list(direct_non_outlier, "class_2"),
            "dice_class_3": summarize_dice_list(direct_non_outlier, "class_3"),
            "dice_class_4": summarize_dice_list(direct_non_outlier, "class_4"),
        },
        "direct_outlier": {
            "n": len(direct_outlier),
            "dice_mean": summarize_dice_list(direct_outlier, "mean"),
        },
        "propagated_all": {
            "n": len(all_prop),
            "dice_mean": summarize_dice_list(all_prop, "mean"),
            "dice_class_1": summarize_dice_list(all_prop, "class_1"),
            "dice_class_2": summarize_dice_list(all_prop, "class_2"),
            "dice_class_3": summarize_dice_list(all_prop, "class_3"),
            "dice_class_4": summarize_dice_list(all_prop, "class_4"),
        },
        "propagated_anchor": {
            "n": len(prop_anchor),
            "dice_mean": summarize_dice_list(prop_anchor, "mean"),
        },
        "propagated_interpolated": {
            "n": len(prop_interp),
            "dice_mean": summarize_dice_list(prop_interp, "mean"),
        },
        "propagated_extrapolated": {
            "n": len(prop_extrap),
            "dice_mean": summarize_dice_list(prop_extrap, "mean"),
        },
    }


# ─── メイン ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="予測線から領域分割生成・GT評価")
    parser.add_argument("--outlier_thresh", type=float, default=OUTLIER_THRESH_DEG,
                        help=f"外れ値判定閾値 [deg] (default: {OUTLIER_THRESH_DEG})")
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS,
                        help="評価対象の fold リスト")
    parser.add_argument("--sample", type=str, default=None,
                        help="特定 sample のみ評価 (e.g., sample17)")
    parser.add_argument("--vertebra", type=str, default=None,
                        help="特定 vertebra のみ評価 (e.g., C1)")
    args = parser.parse_args()

    output_dir = get_output_base() / "region_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    processed = 0

    for fold in args.folds:
        print(f"\n=== fold {fold} ===")
        pred_map = load_all_predictions(fold)

        for (sample, vertebra), slice_preds in sorted(pred_map.items()):
            if args.sample and sample != args.sample:
                continue
            if args.vertebra and vertebra != args.vertebra:
                continue

            print(f"  {sample}/{vertebra}: {len(slice_preds)} slices", end="", flush=True)
            result = evaluate_vertebra(sample, vertebra, slice_preds, args.outlier_thresh)

            if "error" in result:
                print(f"  [ERROR: {result['error']}]")
            else:
                n_direct_ok = sum(1 for d in result["direct"] if "error" not in d)
                n_prop_ok = sum(1 for p in result["propagated"] if "error" not in p)
                outlier_c = result["outlier_count"]
                print(f"  outliers={outlier_c}  direct_ok={n_direct_ok}  prop_ok={n_prop_ok}")

            result["fold"] = fold
            all_results.append(result)

            # 椎骨単位の詳細 JSON を保存
            detail_path = output_dir / f"{sample}_{vertebra}_fold{fold}.json"
            detail_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            processed += 1

    summary = build_summary(all_results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== 結果サマリー (outlier_thresh={args.outlier_thresh}deg) ===")
    print(f"評価椎骨数: {summary['total_vertebrae']}")
    print(f"外れ値割合: {summary['outlier_fraction']:.1%}  "
          f"({summary['total_outlier_slices']}/{summary['total_pred_slices']})")
    print()
    print("【Direct評価（非外れ値スライス）】")
    dn = summary["direct_non_outlier"]
    print(f"  n={dn['n']}  Dice mean={dn['dice_mean']['mean']:.4f}  "
          f"median={dn['dice_mean']['median']:.4f}  p5={dn['dice_mean']['p5']:.4f}")
    for c in range(1, 5):
        dc = dn[f"dice_class_{c}"]
        print(f"    class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    print()
    print("【Direct評価（外れ値スライス）】")
    do = summary["direct_outlier"]
    print(f"  n={do['n']}  Dice mean={do['dice_mean']['mean']:.4f}")
    print()
    print("【z伝播後 全スライス評価】")
    pa = summary["propagated_all"]
    print(f"  n={pa['n']}  Dice mean={pa['dice_mean']['mean']:.4f}  "
          f"median={pa['dice_mean']['median']:.4f}  p5={pa['dice_mean']['p5']:.4f}")
    for c in range(1, 5):
        dc = pa[f"dice_class_{c}"]
        print(f"    class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    prov_labels = [
        ("anchor", summary["propagated_anchor"]),
        ("interpolated", summary["propagated_interpolated"]),
        ("extrapolated", summary["propagated_extrapolated"]),
    ]
    print()
    for label, s in prov_labels:
        print(f"  [{label}] n={s['n']}  Dice mean={s['dice_mean']['mean']:.4f}")

    print()
    print("=" * 60)
    print("【3D Dice（椎骨単位 volumetric, 全由来）】 ← 推奨指標")
    v = summary["volumetric_3d_per_vertebra"]
    print(f"  椎骨数={v['n']}  3D Dice mean={v['dice_mean']['mean']:.4f}  "
          f"median={v['dice_mean']['median']:.4f}  p5={v['dice_mean']['p5']:.4f}")
    for c in range(1, 5):
        dc = v[f"dice_class_{c}"]
        print(f"    class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    print()
    print("【3D Dice（外挿スライスのみ, 椎骨単位）】")
    ve = summary["volumetric_3d_extrap_per_vertebra"]
    print(f"  椎骨数={ve['n']}  3D Dice mean={ve['dice_mean']['mean']:.4f}  "
          f"median={ve['dice_mean']['median']:.4f}")
    for c in range(1, 5):
        dc = ve[f"dice_class_{c}"]
        print(f"    class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    print()
    print("【3D Dice（全椎骨プール / micro-average）】")
    vp = summary["volumetric_3d_pooled"]
    print(f"  mean={vp['mean']:.4f}  "
          + "  ".join(f"class{c}={vp[f'class_{c}']:.4f}" for c in range(1, 5)))
    print(f"\n詳細結果: {output_dir}/")


if __name__ == "__main__":
    main()
