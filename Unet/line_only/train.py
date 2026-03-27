#!/usr/bin/env python
"""
5-Fold Cross-Validation 全自動実行スクリプト
fold0〜4を順次実行し、結果をサマリファイルに保存する
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# train.py を直接実行した場合でも相対 import が通るよう、
# Unet/ ディレクトリを sys.path に追加して line_only をパッケージとして認識させる
_here = Path(__file__).resolve().parent  # line_only/
_unet = _here.parent                     # Unet/
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))

from line_only.src.data_utils import load_config, set_seed
from line_only.src.trainer import train_one_fold


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold CV 全自動実行")
    parser.add_argument(
        "--config", default="config/config.yaml", help="設定ファイルパス"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="使用するGPU ID（configを上書き）"
    )
    parser.add_argument(
        "--start_fold", type=int, default=0, help="開始fold番号（途中再開用）"
    )
    parser.add_argument("--end_fold", type=int, default=4, help="終了fold番号")
    parser.add_argument(
        "--all_vertebrae", action="store_true", help="全椎体(C1-C7)を一括で学習する"
    )
    return parser.parse_args()


def save_summary(all_results: dict, output_path: Path):
    """全Fold結果のサマリーを保存し、平均値を計算"""
    # 主要メトリクスの平均値を計算
    metric_keys = [
        "test_mse",
        "test_peak_dist_mean",
        "line_perpendicular_dist_px_mean",
        "line_angle_error_deg_mean",
        "line_rho_error_px_mean",
    ]

    avg = {}
    for k in metric_keys:
        values = [r.get(k) for r in all_results.values() if r.get(k) is not None]
        if values:
            avg[k] = float(sum(values) / len(values))

    # 椎体ごとの平均を計算
    vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    per_vert_avg = {}

    for v in vertebrae:
        v_metrics = {}
        for fold_result in all_results.values():
            per_vert = fold_result.get("per_vertebra", {})
            if v in per_vert:
                for k, val in per_vert[v].items():
                    if k not in v_metrics:
                        v_metrics[k] = []
                    v_metrics[k].append(val)

        if v_metrics:
            per_vert_avg[v] = {
                k: float(sum(vals) / len(vals)) for k, vals in v_metrics.items() if vals
            }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(all_results),
        "per_fold": all_results,
        "average": avg,
        "per_vertebra_average": per_vert_avg,  # 追加
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Summary saved to: {output_path}")
    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 全椎体モードの場合、groupを"ALL"に設定
    if args.all_vertebrae:
        cfg["data"]["group"] = "ALL"
        print("[INFO] 全椎体モード: C1-C7を一括で学習します")

    # GPU ID上書き
    if args.gpu_id is not None:
        cfg["training"]["gpu_id"] = args.gpu_id

    seed = int(cfg.get("data", {}).get("random_seed", 42))

    if not cfg.get("data", {}).get("use_png", True):
        raise RuntimeError("PNG dataset only (use_png: true)")

    all_results = {}

    for fold in range(args.start_fold, args.end_fold + 1):
        print(f"\n{'=' * 60}")
        print(f"[ALL FOLDS] Starting fold {fold}/{args.end_fold}")
        print(f"{'=' * 60}\n")

        # 各foldで乱数シードを再設定（再現性のため）
        set_seed(seed)

        # configのtest_foldを上書き
        cfg["data"]["test_fold"] = fold

        # 学習実行
        results = train_one_fold(cfg)
        all_results[f"fold{fold}"] = results

    # 結果サマリを保存（Unetディレクトリを基準に）
    script_dir = Path(__file__).resolve().parent.parent  # Unet/ directory
    ckpt_dir = script_dir / cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
    if args.all_vertebrae:
        summary_path = ckpt_dir / "all_vertebrae_summary.json"
    else:
        summary_path = ckpt_dir / "all_folds_summary.json"
    summary = save_summary(all_results, summary_path)

    # Display final results
    print(f"\n{'=' * 70}")
    print("[FINAL SUMMARY] 5-Fold Cross-Validation Results")
    print(f"{'=' * 70}")
    avg = summary["average"]

    print("\n[Primary Metrics - Line Geometry]")
    if "line_perpendicular_dist_px_mean" in avg:
        print(f"  Perpendicular Distance: {avg['line_perpendicular_dist_px_mean']:.2f} px  ⭐")
    if "line_angle_error_deg_mean" in avg:
        print(f"  Angle Error:           {avg['line_angle_error_deg_mean']:.2f} deg ⭐")
    if "line_rho_error_px_mean" in avg:
        print(f"  Rho Error:             {avg['line_rho_error_px_mean']:.2f} px  ⭐")

    print("\n[Auxiliary Metrics - Heatmap Quality]")
    if "test_mse" in avg:
        print(f"  MSE Loss:              {avg['test_mse']:.6f}")
    if "test_peak_dist_mean" in avg:
        print(f"  Peak Distance:         {avg['test_peak_dist_mean']:.2f} px")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
