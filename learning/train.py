"""
椎体骨折分類ベースライン 学習スクリプト。

全データで患者単位5-fold CVを実行し、fold別指標とOOF集約指標を保存する。
学習のたびに実効 config を出力ディレクトリへ保存する。
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

# NFS上でのmultiprocessing一時ディレクトリ削除エラーを防ぐ
os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="椎体骨折分類ベースライン学習")
    p.add_argument(
        "--config",
        default="learning/config/config.yaml",
        help="設定ファイルパス",
    )
    p.add_argument("--gpu_id", type=int, default=None, help="GPU ID（config を上書き）")
    p.add_argument("--beta", type=float, default=None, help="center loss β（config を上書き）")
    p.add_argument("--topk_mode", default=None, choices=["capped", "ratio"], help="top-k モード")
    p.add_argument("--topk_alpha", type=float, default=None, help="ratio モード用 α")
    p.add_argument("--run_name", default=None, help="実験名（省略時は日時）")
    p.add_argument("--start_fold", type=int, default=0, help="開始 fold")
    p.add_argument("--end_fold", type=int, default=4, help="終了 fold（含む）")
    return p.parse_args()


def resolve_output_dir(cfg: dict) -> Path:
    """実験出力ディレクトリを決定する。"""
    run_name = cfg["output"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ROOT_DIR / cfg["output"]["base_dir"]
    return base / run_name


def main() -> None:
    args = parse_args()

    # 遅延 import（GPU 初期化を引数解析後に行う）
    from learning.src.data_utils import (
        collect_bags_from_cfg,
        load_config,
        save_effective_config,
        set_seed,
        split_bags_cv,
    )
    from learning.src.trainer import train_one_fold
    from learning.utils.metrics import compute_level_metrics, compute_oof_metrics

    cfg = load_config(args.config)

    # CLI 上書き
    if args.gpu_id is not None:
        cfg["training"]["gpu_id"] = args.gpu_id
    if args.beta is not None:
        cfg["training"]["beta"] = args.beta
    if args.topk_mode is not None:
        cfg["data"]["topk_mode"] = args.topk_mode
    if args.topk_alpha is not None:
        cfg["data"]["topk_alpha"] = args.topk_alpha
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 実効 config を保存（毎回）
    save_effective_config(cfg, output_dir)

    seed = cfg["data"]["random_seed"]
    set_seed(seed)

    # bag 収集
    dataset_dir = ROOT_DIR / cfg["data"]["dataset_dir"]
    bags = collect_bags_from_cfg(dataset_dir)
    print(f"[INFO] 総 bag 数: {len(bags)} "
          f"(陽性={sum(b['label'] for b in bags)}, "
          f"陰性={sum(1 - b['label'] for b in bags)})")

    n_folds = cfg["data"]["n_folds"]
    if args.start_fold < 0 or args.end_fold >= n_folds or args.start_fold > args.end_fold:
        raise ValueError(f"fold範囲は0〜{n_folds - 1}で指定してください")

    # 全データで5-fold CV
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
            bags,
            n_splits=n_folds,
            val_fold=fold,
            seed=seed,
        )
        print(f"  train={len(train_bags)} val={len(val_bags)}")

        result = train_one_fold(cfg, train_bags, val_bags, fold, output_dir)

        # fold 終了後に val 指標を即時表示・保存
        fold_y = np.array(result["val_labels"])
        fold_p = np.array(result["val_probs"])
        fold_g = np.array(result["val_samples"])
        fold_lv = np.array(result["val_vertebrae"])
        fold_metrics = compute_oof_metrics(fold_y, fold_p, groups=fold_g)
        fold_level = compute_level_metrics(fold_y, fold_p, fold_lv)

        print(f"\n[Fold {fold} val結果] best_epoch={result['best_epoch']}")
        print(f"  AUROC={fold_metrics['auroc']:.4f}  AUPRC={fold_metrics['auprc']:.4f}  "
              f"prevalence={fold_metrics['prevalence']:.3f}")
        print(f"  threshold=0.5  P={fold_metrics['at_05']['precision']:.3f}  "
              f"R={fold_metrics['at_05']['recall']:.3f}  F1={fold_metrics['at_05']['f1']:.3f}")
        print(f"  opt threshold={fold_metrics['at_opt']['threshold']:.3f}  "
              f"P={fold_metrics['at_opt']['precision']:.3f}  "
              f"R={fold_metrics['at_opt']['recall']:.3f}  F1={fold_metrics['at_opt']['f1']:.3f}")
        for lv, m in sorted(fold_level.items()):
            print(f"    {lv:5s}: n_pos={m['n_pos']}/{m['n_total']}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        # fold 単体の metrics.json を保存
        fold_dir = output_dir / f"fold{fold}"
        fold_metrics_path = fold_dir / "fold_metrics.json"
        with fold_metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"fold": fold, "oof": fold_metrics, "level": fold_level,
                       "best_epoch": result["best_epoch"],
                       "best_auprc": result["best_auprc"]}, f, ensure_ascii=False, indent=2)

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

    # 結果表示
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

    # metrics.json に保存
    metrics_out = {
        "oof": oof_metrics,
        "level": level_metrics,
        "folds": fold_results,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] metrics.json を保存しました: {metrics_path}")


if __name__ == "__main__":
    main()
