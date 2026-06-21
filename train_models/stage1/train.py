"""
RSNA 頸椎骨折分類 Stage1 ベースライン 学習エントリポイント。

5-fold CV を実行し、OOF pooled metrics と fold 別 metrics を保存する。

Usage:
    uv run python train_models/stage1/train.py
    uv run python train_models/stage1/train.py --config train_models/stage1/config/config.yaml
    uv run python train_models/stage1/train.py --gpu_id 1 --start_fold 0 --end_fold 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_utils import collect_items, load_config, save_effective_config, set_seed
from src.experiment import resolve_output_base
from src.trainer import train_one_fold
from utils.metrics import compute_level_metrics, compute_oof_metrics

ROOT = Path(__file__).resolve().parent.parent.parent  # VAI/


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSNA Stage1 Training")
    p.add_argument("--config", type=str, default=None, help="config.yaml のパス")
    p.add_argument("--gpu_id", type=int, default=None, help="使用する GPU ID")
    p.add_argument("--start_fold", type=int, default=0, help="開始 fold (inclusive)")
    p.add_argument("--end_fold", type=int, default=None, help="終了 fold (inclusive)")
    p.add_argument("--run_name", type=str, default=None, help="実験名上書き")
    p.add_argument("--epochs", type=int, default=None, help="最大 epoch 数上書き")
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI 引数で config を上書きする。"""
    if args.gpu_id is not None:
        cfg.setdefault("training", {})["gpu_id"] = args.gpu_id
    if args.run_name is not None:
        cfg.setdefault("experiment", {})["name"] = args.run_name
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    return cfg


def print_metrics(metrics: dict, label: str = "") -> None:
    """評価指標をコンソールに表示する。"""
    prefix = f"[{label}] " if label else ""
    auroc = metrics.get("auroc", float("nan"))
    auprc = metrics.get("auprc", float("nan"))
    prevalence = metrics.get("prevalence", float("nan"))
    at_05 = metrics.get("at_05", {})
    at_opt = metrics.get("at_opt", {})
    print(
        f"{prefix}"
        f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
        f"Prevalence={prevalence:.3f}\n"
        f"  @0.5: P={at_05.get('precision', 0):.4f} "
        f"R={at_05.get('recall', 0):.4f} "
        f"F1={at_05.get('f1', 0):.4f}\n"
        f"  @opt(thr={at_opt.get('threshold', 0.5):.3f}): "
        f"P={at_opt.get('precision', 0):.4f} "
        f"R={at_opt.get('recall', 0):.4f} "
        f"F1={at_opt.get('f1', 0):.4f}"
    )


def print_level_metrics(level_metrics: dict[str, dict]) -> None:
    """椎体レベル別指標をコンソールに表示する。"""
    header = f"{'Level':>6} {'n_pos':>6} {'total':>6} {'AUROC':>7} {'AUPRC':>7} {'P':>6} {'R':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for lv, m in level_metrics.items():
        print(
            f"{lv:>6} {m.get('n_pos', 0):>6} {m.get('n_total', 0):>6} "
            f"{m.get('auroc', float('nan')):>7.4f} "
            f"{m.get('auprc', float('nan')):>7.4f} "
            f"{m.get('precision', 0):>6.4f} "
            f"{m.get('recall', 0):>6.4f} "
            f"{m.get('f1', 0):>6.4f}"
        )


def main() -> None:
    args = parse_args()

    # 設定読み込みと CLI 上書き
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    data_cfg = cfg.get("data", {})
    n_folds = int(data_cfg.get("n_folds", 5))
    seed = int(data_cfg.get("random_seed", 42))
    dataset_dir = ROOT / str(data_cfg.get("dataset_dir", "data/rsna_data/fracture_dataset"))
    csv_path = ROOT / str(data_cfg.get("csv_path", "data/rsna_data/train.csv"))

    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else n_folds - 1

    output_base = resolve_output_base(cfg, ROOT)
    output_base.mkdir(parents=True, exist_ok=True)
    save_effective_config(cfg, output_base)

    print(f"\n{'='*60}")
    print(f"RSNA Stage1 学習開始")
    print(f"  config    : {args.config or 'default'}")
    print(f"  output    : {output_base}")
    print(f"  folds     : {start_fold} ~ {end_fold}")
    print(f"  dataset   : {dataset_dir}")
    print(f"{'='*60}\n")

    # データ収集
    set_seed(seed)
    items = collect_items(dataset_dir, csv_path)

    # fold 別学習
    fold_metrics_list: list[dict] = []
    all_oof: list[dict] = []  # {study_uid, vertebra, label, pred_prob}

    for fold in range(start_fold, end_fold + 1):
        fold_metrics, oof_preds = train_one_fold(cfg, fold, items, ROOT)
        fold_metrics_list.append(fold_metrics)
        all_oof.extend(oof_preds)

        print_metrics(fold_metrics, label=f"FOLD {fold}")
        print()

    # OOF pooled metrics
    if len(all_oof) == 0:
        print("[WARNING] OOF 予測がありません。")
        return

    y_true = np.array([x["label"] for x in all_oof], dtype=int)
    y_prob = np.array([x["pred_prob"] for x in all_oof], dtype=float)
    groups = np.array([x["study_uid"] for x in all_oof])
    levels = np.array([x["vertebra"] for x in all_oof])

    oof_metrics = compute_oof_metrics(y_true, y_prob, groups=groups)
    level_metrics = compute_level_metrics(y_true, y_prob, levels)

    print(f"\n{'='*60}")
    print("OOF Pooled Metrics (全 fold 結合)")
    print(f"{'='*60}")
    print_metrics(oof_metrics, label="OOF")
    print()
    print("椎体レベル別 Metrics:")
    print_level_metrics(level_metrics)
    print(f"{'='*60}\n")

    # 結果保存
    results = {
        "oof_metrics": oof_metrics,
        "level_metrics": level_metrics,
        "fold_metrics": fold_metrics_list,
        "n_oof_items": len(all_oof),
    }

    metrics_path = output_base / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 結果を保存しました: {metrics_path}")

    # OOF 予測 CSV 保存
    import pandas as pd

    oof_df = pd.DataFrame(all_oof)
    oof_path = output_base / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[INFO] OOF 予測を保存しました: {oof_path}")


if __name__ == "__main__":
    main()
