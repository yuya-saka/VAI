"""
3D Dice 評価 + 全スライス可視化スクリプト。

utils/region_eval.py の実装を使い:
  - 各 (sample, vertebra) を 3D Dice で評価
  - 全 z スライスの GT/予測オーバーレイ PNG を保存

使い方:
  python Unet/debug/eval_region_3d.py                   # 全件評価
  python Unet/debug/eval_region_3d.py --sample sample17/C1
  python Unet/debug/eval_region_3d.py --no_viz          # 画像出力なし（速い）
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

from Unet.line_only.utils.region_eval import (
    OUTLIER_THRESH_DEG,
    VertebralEvalResult,
    VolumetricDiceAccumulator,
    evaluate_vertebra,
    save_all_slice_overlays,
)

# ─── 定数 ─────────────────────────────────────────────────────────────────
EXPERIMENT = "sig4.0_ALL(CC適用)"
FOLDS = [0, 1, 2, 3, 4]

OUTPUT_BASE = ROOT_DIR / "Unet" / "outputs" / "line_20260616" / EXPERIMENT
ZPROP_BASE  = ROOT_DIR / "data" / "dataset_zprop"


# ─── データロード ─────────────────────────────────────────────────────────

def load_predictions(fold: int) -> dict[tuple[str, str], dict[int, dict]]:
    """fold の test_lines_reeval を (sample, vertebra) -> {z: data} で返す。"""
    pred_dir = OUTPUT_BASE / "vis" / f"fold{fold}" / "test_lines_reeval"
    groups: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for p in sorted(pred_dir.glob("*_PRED_lines.json")):
        m = re.match(r"(sample[\w.]+)_(C\d+)_slice(\d+)_PRED_lines", p.stem)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        groups[key][int(m.group(3))] = json.loads(p.read_text(encoding="utf-8"))
    return dict(groups)


def load_zprop_data(
    sample: str, vertebra: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], tuple[int, int] | None]:
    """
    dataset_zprop から binary masks / GT masks / CT images を読み込む。

    戻り値: (bin_masks, gt_masks, ct_images, (z_lo, z_hi) | None)
    """
    base = ZPROP_BASE / sample / vertebra
    report_path = base / "generation_report.json"
    if not report_path.exists():
        return {}, {}, {}, None
    rng = json.loads(report_path.read_text())["valid_z_range"]
    z_lo, z_hi = int(rng[0]), int(rng[1])

    bin_masks, gt_masks, ct_images = {}, {}, {}
    for z in range(z_lo, z_hi + 1):
        bm_path = base / "masks"   / f"slice_{z:03d}.png"
        gt_path = base / "gt_masks"/ f"slice_{z:03d}.png"
        ct_path = base / "images"  / f"slice_{z:03d}.png"
        if bm_path.exists():
            bin_masks[z] = np.array(Image.open(bm_path))
        if gt_path.exists():
            gt_masks[z]  = np.array(Image.open(gt_path))
        if ct_path.exists():
            ct_images[z] = np.array(Image.open(ct_path).convert("L"))

    return bin_masks, gt_masks, ct_images, (z_lo, z_hi)


# ─── 集計 ─────────────────────────────────────────────────────────────────

def summarize_vertebra_results(results: list[VertebralEvalResult]) -> dict[str, Any]:
    """全椎骨の 3D Dice を集計し summary dict を返す。"""
    valid = [r for r in results if r.error is None]

    def _macro(key: str) -> dict:
        """椎骨単位 3D Dice の macro 平均。"""
        vals = [r.volumetric_dice.get(key, float("nan")) for r in valid]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "p5": float("nan"), "n": 0}
        arr = np.array(vals)
        return {
            "mean":   round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "p5":     round(float(np.percentile(arr, 5)), 6),
            "n": len(arr),
        }

    def _macro_ext(key: str) -> dict:
        vals = [r.volumetric_dice_extrap.get(key, float("nan")) for r in valid]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "p5": float("nan"), "n": 0}
        arr = np.array(vals)
        return {
            "mean":   round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "p5":     round(float(np.percentile(arr, 5)), 6),
            "n": len(arr),
        }

    # micro average（全椎骨プール）
    pooled = VolumetricDiceAccumulator()
    for r in valid:
        for rec in r.per_slice:
            if "error" in rec:
                continue
            # per-slice では 2D Dice しか記録していないため pooled は別途計算済みカウントなし
            # → macro のみ提供し micro は近似で省略（必要なら再実装）

    return {
        "total_vertebrae":  len(results),
        "valid_vertebrae":  len(valid),
        "failed_vertebrae": len(results) - len(valid),
        "outlier_fraction": round(
            sum(r.outlier_count for r in valid)
            / max(sum(r.outlier_count + r.anchor_count for r in valid), 1), 4
        ),
        "3d_dice_all": {
            "mean":    _macro("mean"),
            "class_1": _macro("class_1"),
            "class_2": _macro("class_2"),
            "class_3": _macro("class_3"),
            "class_4": _macro("class_4"),
        },
        "3d_dice_extrap": {
            "mean":    _macro_ext("mean"),
            "class_1": _macro_ext("class_1"),
            "class_2": _macro_ext("class_2"),
            "class_3": _macro_ext("class_3"),
            "class_4": _macro_ext("class_4"),
        },
    }


# ─── メイン ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="3D Dice 評価 + 全スライス可視化")
    parser.add_argument("--sample", nargs="+", default=None,
                        help="sample/vertebra を指定（例: sample17/C1 sample7/C3）")
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    parser.add_argument("--outlier_thresh", type=float, default=OUTLIER_THRESH_DEG)
    parser.add_argument("--no_viz", action="store_true", help="スライス画像を生成しない")
    args = parser.parse_args()

    # フィルタ用セット
    target_keys: set[tuple[str, str]] | None = None
    if args.sample:
        target_keys = set()
        for t in args.sample:
            s, v = t.split("/")
            target_keys.add((s, v))

    out_base    = OUTPUT_BASE / "region_eval_3d"
    detail_base = out_base / "details"
    viz_base    = out_base / "viz"
    detail_base.mkdir(parents=True, exist_ok=True)

    # 全 fold の予測を収集（重複は後の fold で上書き）
    all_preds: dict[tuple[str, str], dict[int, dict]] = {}
    for fold in args.folds:
        for key, preds in load_predictions(fold).items():
            all_preds[key] = preds

    if target_keys:
        all_preds = {k: v for k, v in all_preds.items() if k in target_keys}

    all_results: list[VertebralEvalResult] = []

    for (sample, vertebra), slice_preds in sorted(all_preds.items()):
        print(f"  {sample}/{vertebra}: {len(slice_preds)} pred slices", end="", flush=True)

        bin_masks, gt_masks, ct_images, z_range = load_zprop_data(sample, vertebra)
        if z_range is None:
            print("  [SKIP: no zprop data]")
            all_results.append(VertebralEvalResult(
                sample=sample, vertebra=vertebra,
                volumetric_dice={}, volumetric_dice_extrap={},
                per_slice=[], anchor_count=0, outlier_count=0,
                fallback_used=False, error="no zprop data",
            ))
            continue

        result = evaluate_vertebra(
            sample, vertebra, bin_masks, gt_masks, slice_preds, args.outlier_thresh
        )

        if result.error:
            print(f"  [ERROR: {result.error}]")
        else:
            d = result.volumetric_dice
            print(f"  anchors={result.anchor_count}  outliers={result.outlier_count}"
                  f"  3D-Dice {d.get('mean', float('nan')):.4f}"
                  f"  (c1={d.get('class_1', float('nan')):.3f}"
                  f" c2={d.get('class_2', float('nan')):.3f}"
                  f" c3={d.get('class_3', float('nan')):.3f}"
                  f" c4={d.get('class_4', float('nan')):.3f})")

        all_results.append(result)

        # 詳細 JSON 保存
        detail = {
            "sample": result.sample,
            "vertebra": result.vertebra,
            "z_range": list(z_range),
            "anchor_count": result.anchor_count,
            "outlier_count": result.outlier_count,
            "fallback_used": result.fallback_used,
            "error": result.error,
            "volumetric_dice": result.volumetric_dice,
            "volumetric_dice_extrap": result.volumetric_dice_extrap,
            "per_slice": result.per_slice,
        }
        detail_path = detail_base / f"{sample}_{vertebra}.json"
        detail_path.write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 全スライス可視化
        if not args.no_viz and result.error is None:
            vert_viz_dir = viz_base / sample / vertebra
            saved = save_all_slice_overlays(
                sample, vertebra,
                bin_masks, gt_masks, ct_images, slice_preds,
                vert_viz_dir, args.outlier_thresh,
            )
            # 保存数を小さく表示
            if saved:
                print(f"    -> {len(saved)} slices -> {vert_viz_dir}")

    # サマリー
    summary = summarize_vertebra_results(all_results)
    summary_path = out_base / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"3D Dice 評価結果（{summary['valid_vertebrae']} 椎骨）")
    print(f"{'='*60}")
    print(f"外れ値割合: {summary['outlier_fraction']:.1%}")
    print()
    print("【3D Dice 全由来（macro, 椎骨単位）】")
    d = summary["3d_dice_all"]
    print(f"  mean  : {d['mean']['mean']:.4f}  median={d['mean']['median']:.4f}  p5={d['mean']['p5']:.4f}")
    for c in range(1, 5):
        dc = d[f"class_{c}"]
        print(f"  class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    print()
    print("【3D Dice 外挿のみ（macro, 椎骨単位）】")
    de = summary["3d_dice_extrap"]
    print(f"  mean  : {de['mean']['mean']:.4f}  median={de['mean']['median']:.4f}")
    for c in range(1, 5):
        dc = de[f"class_{c}"]
        print(f"  class{c}: mean={dc['mean']:.4f}  median={dc['median']:.4f}")
    print(f"\n出力先: {out_base}/")
    if not args.no_viz:
        print(f"画像: {viz_base}/")


if __name__ == "__main__":
    main()
