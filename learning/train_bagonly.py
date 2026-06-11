"""
椎体骨折分類 bag mean pooling 版 学習スクリプト。

mean pooling bag BCE のみ（top-k MIL・center loss なし）。

使い方:
    uv run python -m learning.train_bagonly
    uv run python -m learning.train_bagonly --gpu_id 0 --run_name bagonly_run01
    uv run python -m learning.train_bagonly --start_fold 0 --end_fold 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="椎体骨折分類 bag mean pooling 版学習")
    p.add_argument(
        "--config",
        default="learning/bagonly_src/config/config.yaml",
        help="設定ファイルパス",
    )
    p.add_argument("--gpu_id", type=int, default=None, help="GPU ID（config を上書き）")
    p.add_argument("--run_name", default=None, help="実験名（省略時は config 値）")
    p.add_argument("--start_fold", type=int, default=0, help="開始 fold")
    p.add_argument("--end_fold", type=int, default=4, help="終了 fold（含む）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from learning.src.data_utils import (
        collect_bags_from_cfg,
        load_config,
        save_effective_config,
        set_seed,
        split_bags_cv,
    )
    from learning.bagonly_src.trainer import train_one_fold
    from learning.utils.metrics import compute_level_metrics, compute_oof_metrics

    cfg = load_config(args.config)

    if args.gpu_id is not None:
        cfg["training"]["gpu_id"] = args.gpu_id
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    run_name = cfg["output"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / cfg["output"]["base_dir"] / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(cfg, output_dir)

    seed = cfg["data"]["random_seed"]
    set_seed(seed)

    dataset_dir = ROOT_DIR / cfg["data"]["dataset_dir"]
    bags = collect_bags_from_cfg(dataset_dir)
    print(
        f"[INFO] 総 bag 数: {len(bags)} "
        f"(陽性={sum(b['label'] for b in bags)}, "
        f"陰性={sum(1 - b['label'] for b in bags)})"
    )

    n_folds = cfg["data"]["n_folds"]
    if args.start_fold < 0 or args.end_fold >= n_folds or args.start_fold > args.end_fold:
        raise ValueError(f"fold範囲は0〜{n_folds - 1}で指定してください")

    all_val_samples: list[str] = []
    all_val_vertebrae: list[str] = []
    all_val_labels: list[int] = []
    all_val_probs: list[float] = []
    fold_results: list[dict] = []

    for fold in range(args.start_fold, args.end_fold + 1):
        print(f"\n{'=' * 60}")
        print(f"[CV] Fold {fold}")
        print(f"{'=' * 60}")
        set_seed(seed)

        train_bags, val_bags = split_bags_cv(
            bags, n_splits=n_folds, val_fold=fold, seed=seed,
        )
        print(f"  train={len(train_bags)} val={len(val_bags)}")

        result = train_one_fold(cfg, train_bags, val_bags, fold, output_dir)

        fold_y = np.array(result["val_labels"])
        fold_p = np.array(result["val_probs"])
        fold_g = np.array(result["val_samples"])
        fold_lv = np.array(result["val_vertebrae"])
        fold_metrics = compute_oof_metrics(fold_y, fold_p, groups=fold_g)
        fold_level = compute_level_metrics(fold_y, fold_p, fold_lv)

        print(f"\n[Fold {fold} val結果] best_epoch={result['best_epoch']}")
        print(
            f"  AUROC={fold_metrics['auroc']:.4f}  AUPRC={fold_metrics['auprc']:.4f}  "
            f"prevalence={fold_metrics['prevalence']:.3f}"
        )
        print(
            f"  threshold=0.5  P={fold_metrics['at_05']['precision']:.3f}  "
            f"R={fold_metrics['at_05']['recall']:.3f}  F1={fold_metrics['at_05']['f1']:.3f}"
        )
        print(
            f"  opt threshold={fold_metrics['at_opt']['threshold']:.3f}  "
            f"P={fold_metrics['at_opt']['precision']:.3f}  "
            f"R={fold_metrics['at_opt']['recall']:.3f}  F1={fold_metrics['at_opt']['f1']:.3f}"
        )
        for lv, m in sorted(fold_level.items()):
            print(
                f"    {lv:5s}: n_pos={m['n_pos']}/{m['n_total']}  "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
            )

        fold_dir = output_dir / f"fold{fold}"
        with (fold_dir / "fold_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": fold,
                    "oof": fold_metrics,
                    "level": fold_level,
                    "best_epoch": result["best_epoch"],
                    "best_auprc": result["best_auprc"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        fold_results.append({
            "fold": fold,
            "best_auprc": result["best_auprc"],
            "best_epoch": result["best_epoch"],
            "auroc": fold_metrics["auroc"],
            "auprc": fold_metrics["auprc"],
        })
        all_val_samples.extend(result["val_samples"])
        all_val_vertebrae.extend(result["val_vertebrae"])
        all_val_labels.extend(result["val_labels"])
        all_val_probs.extend(result["val_probs"])

    # OOF 集約評価
    y_true = np.array(all_val_labels)
    y_prob = np.array(all_val_probs)
    groups = np.array(all_val_samples)
    levels = np.array(all_val_vertebrae)

    oof_metrics = compute_oof_metrics(y_true, y_prob, groups=groups)
    level_metrics = compute_level_metrics(y_true, y_prob, levels)

    print(f"\n{'=' * 60}")
    print("[OOF 集約結果]")
    print(f"  有病率: {oof_metrics['prevalence']:.3f}")
    print(f"  AUROC:  {oof_metrics['auroc']:.4f}")
    print(f"  AUPRC:  {oof_metrics['auprc']:.4f}")
    print("  --- threshold=0.5 ---")
    print(f"  Precision: {oof_metrics['at_05']['precision']:.4f}")
    print(f"  Recall:    {oof_metrics['at_05']['recall']:.4f}")
    print(f"  F1:        {oof_metrics['at_05']['f1']:.4f}")
    print(f"  --- F1最適 threshold={oof_metrics['at_opt']['threshold']:.3f} ---")
    print(f"  Precision: {oof_metrics['at_opt']['precision']:.4f}")
    print(f"  Recall:    {oof_metrics['at_opt']['recall']:.4f}")
    print(f"  F1:        {oof_metrics['at_opt']['f1']:.4f}")
    if "bootstrap_ci" in oof_metrics:
        ci = oof_metrics["bootstrap_ci"]
        print(f"  AUROC 95% CI: [{ci['auroc_lo']:.4f}, {ci['auroc_hi']:.4f}]")
        print(f"  AUPRC 95% CI: [{ci['auprc_lo']:.4f}, {ci['auprc_hi']:.4f}]")

    print("\n[レベル別 P/R/F1 (threshold=0.5)]")
    for lv, m in sorted(level_metrics.items()):
        print(
            f"  {lv:5s}: n_pos={m['n_pos']:3d}/{m['n_total']:3d} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"oof": oof_metrics, "level": level_metrics, "folds": fold_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[INFO] metrics.json を保存しました: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
