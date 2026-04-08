#!/usr/bin/env python
"""5-Fold Cross-Validation を順次実行し、結果を集約して保存する。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve().parent
_unet = _here.parent
if str(_unet) not in sys.path:
    sys.path.insert(0, str(_unet))

from multitask.src.data_utils import load_config, set_seed
from multitask.src.trainer import train_one_fold


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="5-Fold CV 全自動実行（Multitask）")
    parser.add_argument("--config", default="multitask/config/config.yaml", help="設定ファイルパス")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="使用するGPU ID（configを上書き）",
    )
    parser.add_argument("--start_fold", type=int, default=0, help="開始fold番号（途中再開用）")
    parser.add_argument("--end_fold", type=int, default=4, help="終了fold番号")
    return parser.parse_args()


def _resolve_output_base(cfg: dict[str, Any], script_dir: Path) -> Path | None:
    """experiment 設定がある場合に outputs ベースパスを返す。"""
    exp = cfg.get("experiment")
    if exp and exp.get("phase") and exp.get("name"):
        return script_dir / "outputs" / exp["phase"] / exp["name"]
    return None


def save_summary(all_results: dict[str, dict[str, Any]], output_path: Path) -> dict[str, Any]:
    """全fold結果を保存し、平均値を計算して返す。"""
    metric_keys = [
        "test_line_loss",
        "test_seg_miou",
        "test_peak_dist_mean",
        "line_angle_error_deg_mean",
        "line_rho_error_px_mean",
        "line_perpendicular_dist_px_mean",
    ]

    avg: dict[str, float] = {}
    for k in metric_keys:
        values = [r.get(k) for r in all_results.values() if r.get(k) is not None]
        if values:
            avg[k] = float(sum(values) / len(values))

    vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    per_vert_avg: dict[str, dict[str, float]] = {}

    for v in vertebrae:
        v_metrics: dict[str, list[float]] = {}
        for fold_result in all_results.values():
            per_vert = fold_result.get("per_vertebra", {})
            if v in per_vert:
                for k, val in per_vert[v].items():
                    if isinstance(val, (int, float)):
                        v_metrics.setdefault(k, []).append(float(val))

        if v_metrics:
            per_vert_avg[v] = {
                k: float(sum(vals) / len(vals))
                for k, vals in v_metrics.items()
                if len(vals) > 0
            }

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(all_results),
        "per_fold": all_results,
        "average": avg,
        "per_vertebra_average": per_vert_avg,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Summary saved to: {output_path}")
    return summary


def main() -> None:
    """全fold実行のエントリポイント。"""
    args = parse_args()
    cfg = load_config(args.config)

    if args.gpu_id is not None:
        cfg["training"]["gpu_id"] = args.gpu_id

    seed = int(cfg.get("data", {}).get("random_seed", 42))

    if not cfg.get("data", {}).get("use_png", True):
        raise RuntimeError("PNG dataset only (use_png: true)")

    all_results: dict[str, dict[str, Any]] = {}

    for fold in range(args.start_fold, args.end_fold + 1):
        print(f"\n{'=' * 60}")
        print(f"[ALL FOLDS] Starting fold {fold}/{args.end_fold}")
        print(f"{'=' * 60}\n")

        set_seed(seed)
        cfg["data"]["test_fold"] = fold

        results = train_one_fold(cfg)
        all_results[f"fold{fold}"] = results

    script_dir = Path(__file__).resolve().parent.parent
    output_base = _resolve_output_base(cfg, script_dir)
    if output_base is not None:
        summary_path = output_base / "checkpoints" / "all_folds_summary.json"
    else:
        ckpt_dir = script_dir / cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
        summary_path = ckpt_dir / "all_folds_summary.json"

    summary = save_summary(all_results, summary_path)

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

    print("\n[Auxiliary Metrics - Multitask]")
    if "test_line_loss" in avg:
        print(f"  Test Line Loss:        {avg['test_line_loss']:.6f}")
    if "test_seg_miou" in avg:
        print(f"  Test Seg mIoU:         {avg['test_seg_miou']:.4f}")
    if "test_peak_dist_mean" in avg:
        print(f"  Peak Distance:         {avg['test_peak_dist_mean']:.2f} px")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
