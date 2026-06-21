"""
学習ループ・1fold 学習オーケストレーター。

Unet/line_only/src/trainer.py + learning/src/trainer.py のハイブリッド。
AMP + mixup + BCE損失 + early stopping。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.losses import criterion, mixup
from .evaluation import compute_epoch_metrics
from .experiment import (
    append_log,
    build_epoch_log,
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    resolve_fold_paths,
    write_log_header,
)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    p_mixup: float,
    positive_weight: float,
    use_amp: bool,
) -> float:
    """
    1 epoch の学習を実行する。

    Returns:
        epoch平均 training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)   # (bs, 15, 6, 224, 224)
        labels = labels.to(device, non_blocking=True)   # (bs, 15)

        # mixup
        if p_mixup > 0.0 and torch.rand(1).item() < p_mixup:
            images, labels_a, labels_b, lam = mixup(images, labels)
            with autocast(enabled=use_amp):
                logits = model(images)  # (bs, 15)
                loss = lam * criterion(logits, labels_a, positive_weight) + (
                    1 - lam
                ) * criterion(logits, labels_b, positive_weight)
        else:
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels, positive_weight)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_weight: float,
    use_amp: bool,
) -> tuple[float, dict]:
    """
    validation を実行して val_loss と epoch metrics を返す。

    vertebra-level 確率: sigmoid(logits).mean(dim=1) — 15 スライス平均。

    Returns:
        (val_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_labels: list[int] = []
    all_probs: list[float] = []
    all_study_uids: list[str] = []
    all_vertebrae: list[str] = []

    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
            study_uids_b = [""] * images.shape[0]
            vertebrae_b = [""] * images.shape[0]
        else:
            images, labels, study_uids_b, vertebrae_b = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels, positive_weight)

        total_loss += loss.item()
        n_batches += 1

        # vertebra-level 確率: 15 スライスの sigmoid 平均
        probs = torch.sigmoid(logits).mean(dim=1)  # (bs,)
        # 椎体ラベルは全スライス同値なので最初のスライスを使う
        item_labels = labels[:, 0]                  # (bs,)

        all_labels.extend(item_labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_study_uids.extend(list(study_uids_b))
        all_vertebrae.extend(list(vertebrae_b))

    val_loss = total_loss / max(n_batches, 1)
    metrics = compute_epoch_metrics(
        np.array(all_labels, dtype=int),
        np.array(all_probs, dtype=float),
        np.array(all_study_uids),
        np.array(all_vertebrae),
    )

    return val_loss, metrics


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: dict,
    device: torch.device,
    best_model_path: Path,
    log_path: Path,
    fold: int,
    use_wandb: bool,
    wandb_run,
) -> dict:
    """
    全 epoch のトレーニングループを実行する。

    early stopping: val_loss が `early_stopping_patience` epoch 改善しなければ停止。
    best model: val_loss が最小の epoch のモデルを保存。

    Returns:
        best epoch の val metrics
    """
    tr_cfg = cfg.get("training", {})
    epochs = int(tr_cfg.get("epochs", 75))
    p_mixup = float(tr_cfg.get("p_mixup", 0.5))
    positive_weight = float(tr_cfg.get("positive_weight", 2.0))
    patience = int(tr_cfg.get("early_stopping_patience", 15))
    use_amp = bool(tr_cfg.get("use_amp", True))

    scaler = GradScaler(enabled=use_amp)
    write_log_header(log_path, fold)

    best_val_loss = float("inf")
    best_metrics: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = _train_epoch(
            model, train_loader, optimizer, scaler, device,
            p_mixup, positive_weight, use_amp
        )
        val_loss, metrics = _validate(
            model, val_loader, device, positive_weight, use_amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        auroc = metrics.get("auroc", float("nan"))
        auprc = metrics.get("auprc", float("nan"))
        f1 = metrics.get("at_05", {}).get("f1", float("nan"))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_metrics = metrics
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val_metrics": metrics},
                best_model_path,
            )
        else:
            no_improve += 1

        # ログ出力
        log_line = build_epoch_log(epoch, epochs, lr, train_loss, val_loss, metrics, elapsed)
        print(log_line)
        append_log(log_path, fold, epoch, train_loss, val_loss, auroc, auprc, f1, lr, elapsed, is_best)

        if use_wandb and wandb_run is not None:
            log_wandb_epoch(wandb_run, epoch, lr, train_loss, val_loss, metrics)

        if no_improve >= patience:
            print(f"[EARLY STOP] {patience} epoch 改善なし (fold={fold}, epoch={epoch})")
            break

    if use_wandb and wandb_run is not None:
        finish_wandb(wandb_run, best_metrics)

    return best_metrics


def train_one_fold(
    cfg: dict,
    fold: int,
    items: list[dict],
    root: Path,
) -> tuple[dict, list[dict]]:
    """
    1 fold の学習を実行する。

    Args:
        cfg: config dict
        fold: 現在の fold インデックス (0-indexed)
        items: collect_items() の全アイテムリスト
        root: プロジェクトルートパス

    Returns:
        (best_val_metrics, oof_predictions)
        oof_predictions: val_items の各要素に pred_prob / label / vertebra / study_uid を付加したリスト
    """
    from .data_utils import (
        create_data_loaders,
        create_model_optimizer_scheduler,
        set_seed,
        split_items_cv,
    )

    tr_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    seed = int(data_cfg.get("random_seed", 42))
    positive_weight = float(tr_cfg.get("positive_weight", 2.0))
    use_amp = bool(tr_cfg.get("use_amp", True))
    gpu_id = int(tr_cfg.get("gpu_id", 0))
    n_folds = int(data_cfg.get("n_folds", 5))

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    best_model_path, fold_dir = resolve_fold_paths(cfg, fold, root)
    log_path = fold_dir / "training.log"

    use_wandb, wandb_run = initialize_wandb(cfg, fold)

    # データ分割
    train_items, val_items = split_items_cv(items, n_splits=n_folds, val_fold=fold, seed=seed)
    train_loader, val_loader = create_data_loaders(train_items, val_items, cfg)

    # モデル・最適化器
    model, optimizer, scheduler = create_model_optimizer_scheduler(cfg, device)

    print(
        f"\n{'='*60}\n[FOLD {fold}] train={len(train_items)} val={len(val_items)}\n{'='*60}"
    )

    # 学習ループ
    best_metrics = run_training_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        cfg, device, best_model_path, log_path, fold,
        use_wandb, wandb_run,
    )

    # best モデルで val_items を再推論して OOF 予測を収集
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    oof_preds = _collect_oof_predictions(model, val_items, cfg, device, positive_weight, use_amp)

    print(
        f"[FOLD {fold}] 完了 "
        f"AUROC={best_metrics.get('auroc', float('nan')):.4f} "
        f"AUPRC={best_metrics.get('auprc', float('nan')):.4f}"
    )

    return best_metrics, oof_preds


@torch.no_grad()
def _collect_oof_predictions(
    model: nn.Module,
    val_items: list[dict],
    cfg: dict,
    device: torch.device,
    positive_weight: float,
    use_amp: bool,
) -> list[dict]:
    """
    best model で val_items を推論し、OOF 予測リストを返す。

    Returns:
        各要素: {study_uid, vertebra, label, pred_prob}
    """
    from .data_utils import seed_worker
    from .dataset import RSNAFractureDataset, get_valid_transforms

    tr_cfg = cfg.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 4))

    ds = RSNAFractureDataset(val_items, mode="valid", transform=get_valid_transforms(), p_rand_order=0.0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    all_probs: list[float] = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
        probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy().tolist()
        all_probs.extend(probs)

    oof_preds = []
    for item, prob in zip(val_items, all_probs):
        oof_preds.append(
            {
                "study_uid": item["study_uid"],
                "vertebra": item["vertebra"],
                "label": item["label"],
                "pred_prob": float(prob),
            }
        )

    return oof_preds
