"""
RSNA 頸椎骨折分類 Stage1 ベースライン 学習エントリポイント。

データ分割:
  全データ (2,012 studies)
    → 20% を held-out test set として固定分離（学習・選択に不使用）
    → 残り 80% で 5-fold CV (train / val)
  最終評価: 全 fold best model のアンサンブルで test set を評価

Usage:
    uv run python train_models/stage1/train.py
    uv run python train_models/stage1/train.py --start_fold 0 --end_fold 0
    uv run python train_models/stage1/train.py --epochs 5 --end_fold 0  # 動作確認
    # GPU数は config.yaml の n_gpu で制御（1 or 2）
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Tee:
    """stdout / stderr を画面とファイルの両方に書き出す。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


import numpy as np
from src.data_utils import (
    collect_items,
    load_config,
    save_effective_config,
    set_seed,
    split_test_holdout,
)
from src.experiment import resolve_fold_paths, resolve_output_base
from src.trainer import predict_on_items, train_one_fold
from utils.metrics import compute_level_metrics, compute_oof_metrics

ROOT = Path(__file__).resolve().parent.parent.parent  # VAI/


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def configure_local_temp_dir() -> Path:
    """multiprocessing一時ファイルをNFSではなくローカル/tmpへ配置する。"""
    local_tmp_dir = Path("/tmp") / f"vai-stage1-{os.getuid()}"
    local_tmp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(local_tmp_dir)
    tempfile.tempdir = str(local_tmp_dir)
    return local_tmp_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSNA Stage1 Training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--gpu_ids", type=str, default=None, help="使用GPU番号 例: '0,1'")
    p.add_argument("--start_fold", type=int, default=None)
    p.add_argument("--end_fold", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
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


def _do_training(local_rank: int, world_size: int, args: argparse.Namespace, cfg: dict) -> None:
    """実際の学習ロジック。シングル・マルチ GPU 共通。"""
    is_main = local_rank == 0

    local_tmp_dir = configure_local_temp_dir()
    if is_main:
        print(f"[INFO] multiprocessing temp: {local_tmp_dir}")

    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    n_folds = int(data_cfg.get("n_folds", 5))
    seed = int(data_cfg.get("random_seed", 42))
    test_size = float(data_cfg.get("test_holdout_size", 0.2))
    gpu_id = local_rank  # CUDA_VISIBLE_DEVICES でマッピング済みなので index=local_rank
    dataset_dir = ROOT / str(data_cfg.get("dataset_dir", "data/rsna_data/fracture_dataset"))
    csv_path = ROOT / str(data_cfg.get("csv_path", "data/rsna_data/train.csv"))

    start_fold = args.start_fold if args.start_fold is not None else int(data_cfg.get("start_fold", 0))
    end_fold = args.end_fold if args.end_fold is not None else int(data_cfg.get("end_fold", n_folds - 1))

    output_base = resolve_output_base(cfg, ROOT)
    if is_main:
        output_base.mkdir(parents=True, exist_ok=True)
        save_effective_config(cfg, output_base)

        run_log_path = output_base / f"run_fold{start_fold}_{end_fold}.log"
        _log_file = open(run_log_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)
        print(f"[INFO] 実行ログを保存中: {run_log_path}")

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        print(f"\n{'='*60}")
        print("RSNA Stage1 学習開始")
        print(f"  output    : {output_base}")
        print(f"  folds     : {start_fold} ~ {end_fold}")
        print(f"  test_size : {test_size:.0%}")
        if world_size > 1:
            print(f"  DDP       : {world_size} GPU")
        print(f"{'='*60}\n")

    set_seed(seed)
    items = collect_items(dataset_dir, csv_path)
    train_val_items, test_items = split_test_holdout(items, test_size=test_size, seed=seed)

    if is_main:
        print(
            f"\n[INFO] 分割確認\n"
            f"  train+val : {len(train_val_items)} アイテム\n"
            f"  test      : {len(test_items)} アイテム（held-out, 学習不使用）\n"
        )

    cfg.setdefault("training", {})["gpu_id"] = gpu_id

    fold_metrics_list: list[dict] = []
    all_oof: list[dict] = []

    for fold in range(start_fold, end_fold + 1):
        fold_metrics, oof_preds = train_one_fold(cfg, fold, train_val_items, ROOT)
        fold_metrics_list.append(fold_metrics)
        all_oof.extend(oof_preds)
        if is_main:
            print_metrics(fold_metrics, label=f"FOLD {fold} val")
            print()

    if not is_main:
        return

    # OOF metrics
    oof_metrics: dict = {}
    oof_level_metrics: dict = {}
    opt_thresh = 0.5
    if all_oof:
        y_true_oof = np.array([x["label"] for x in all_oof], dtype=int)
        y_prob_oof = np.array([x["pred_prob"] for x in all_oof], dtype=float)
        groups_oof = np.array([x["study_uid"] for x in all_oof])
        levels_oof = np.array([x["vertebra"] for x in all_oof])

        oof_metrics = compute_oof_metrics(y_true_oof, y_prob_oof, groups=groups_oof)
        opt_thresh = float(oof_metrics.get("at_opt", {}).get("threshold", 0.5))
        oof_level_metrics = compute_level_metrics(y_true_oof, y_prob_oof, levels_oof, threshold=opt_thresh)

        print(f"\n{'='*60}")
        print("OOF Metrics (train_val, 全 fold 結合)")
        print(f"{'='*60}")
        print_metrics(oof_metrics, label="OOF")
        print()
        print("椎体レベル別:")
        print_level_metrics(oof_level_metrics)

    # Test set 評価
    test_metrics: dict = {}
    test_level_metrics: dict = {}
    trained_folds = list(range(start_fold, end_fold + 1))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model_paths = []
    for fold in trained_folds:
        best_path, _ = resolve_fold_paths(cfg, fold, ROOT)
        if best_path.exists():
            model_paths.append(best_path)
        else:
            print(f"[WARNING] fold {fold} の best model が見つかりません: {best_path}")

    if model_paths and test_items:
        print(f"\n{'='*60}")
        print(f"Test set 評価 ({len(model_paths)} fold アンサンブル)")
        print(f"{'='*60}")
        test_preds = predict_on_items(model_paths, test_items, cfg, device)

        y_true_test = np.array([x["label"] for x in test_preds], dtype=int)
        y_prob_test = np.array([x["pred_prob"] for x in test_preds], dtype=float)
        groups_test = np.array([x["study_uid"] for x in test_preds])
        levels_test = np.array([x["vertebra"] for x in test_preds])

        test_metrics = compute_oof_metrics(y_true_test, y_prob_test, groups=groups_test)
        test_level_metrics = compute_level_metrics(y_true_test, y_prob_test, levels_test, threshold=opt_thresh)

        print_metrics(test_metrics, label="TEST")
        print()
        print("椎体レベル別:")
        print_level_metrics(test_level_metrics)

        import pandas as pd
        test_df = pd.DataFrame(test_preds)
        test_df.to_csv(output_base / "test_predictions.csv", index=False)
        print(f"\n[INFO] test 予測を保存しました: {output_base / 'test_predictions.csv'}")

    results = {
        "oof_metrics": oof_metrics,
        "oof_level_metrics": oof_level_metrics,
        "test_metrics": test_metrics,
        "test_level_metrics": test_level_metrics,
        "fold_metrics": fold_metrics_list,
        "n_train_val": len(train_val_items),
        "n_test": len(test_items),
        "n_oof": len(all_oof),
        "trained_folds": trained_folds,
        "opt_threshold": oof_metrics.get("at_opt", {}).get("threshold", 0.5) if oof_metrics else 0.5,
    }

    metrics_path = output_base / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 結果を保存しました: {metrics_path}")

    if all_oof:
        import pandas as pd
        oof_df = pd.DataFrame(all_oof)
        oof_df.to_csv(output_base / "oof_predictions.csv", index=False)
        print(f"[INFO] OOF 予測を保存しました: {output_base / 'oof_predictions.csv'}")

    print(f"\n{'='*60}")
    print("完了")
    if oof_metrics:
        print(f"  OOF  AUROC: {oof_metrics.get('auroc', float('nan')):.4f}")
    if test_metrics:
        print(f"  TEST AUROC: {test_metrics.get('auroc', float('nan')):.4f}")
    print(f"{'='*60}\n")


def _worker(local_rank: int, world_size: int, port: int, args: argparse.Namespace, cfg: dict) -> None:
    """mp.spawn から呼ばれるDDPワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    try:
        _do_training(local_rank, world_size, args, cfg)
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    tr_cfg = cfg.get("training", {})

    # CUDA_VISIBLE_DEVICES を CUDA 初期化前に設定
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        elif tr_cfg.get("gpu_ids"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in tr_cfg["gpu_ids"])

    n_gpu = int(tr_cfg.get("n_gpu", 1))

    if n_gpu > 1:
        port = _find_free_port()
        mp.spawn(_worker, args=(n_gpu, port, args, cfg), nprocs=n_gpu, join=True)
    else:
        _do_training(local_rank=0, world_size=1, args=args, cfg=cfg)


if __name__ == "__main__":
    main()
