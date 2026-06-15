"""
1-fold 学習ループ（mean pooling bag BCE 版）。

DMIL top-k・center loss を排除し、全 slice の mean logit に BCE を掛ける。
grad accum (4 bag)・AMP fp16・differential LR・warmup・early stop は既存と同様。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast
from torch.utils.data import DataLoader

from learning.bagonly_src.losses import batch_bag_mean_loss
from learning.src.dataset import VERTEBRA_TO_INDEX
from learning.src.model import FractureResNet18
from learning.utils.metrics import _prf_at_threshold, find_optimal_threshold
from learning.utils.training import (
    build_data_loaders,
    build_model,
    compute_ranking_metrics,
    lr_scale,
    make_optimizer,
    resolve_device,
)

_LOG_HEADER = (
    f"{'fold':>4} {'epoch':>5} {'loss':>8} "
    f"{'val_loss':>8} {'val_AUPRC':>9} {'val_AUROC':>9} "
    f"{'val_F1':>7} {'thr':>5} "
    f"{'lr_bb':>10} {'lr_hd':>10} {'sec':>6} {'best':>5}"
)


def _write_log_header(log_path: Path, fold: int) -> None:
    """ログファイルを新規作成してヘッダーを書く。"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Fold {fold} training log (bagonly - mean pooling)\n")
        f.write(_LOG_HEADER + "\n")
        f.write("-" * len(_LOG_HEADER) + "\n")


def _append_log(
    log_path: Path,
    fold: int,
    epoch: int,
    loss: float,
    val_loss: float,
    val_auprc: float,
    val_auroc: float,
    val_f1: float,
    val_thr: float,
    lr_bb: float,
    lr_hd: float,
    elapsed: float,
    is_best: bool,
) -> None:
    """1 epoch 分のログ行を追記する。"""
    line = (
        f"{fold:>4} {epoch:>5} {loss:>8.4f} "
        f"{val_loss:>8.4f} {val_auprc:>9.4f} {val_auroc:>9.4f} "
        f"{val_f1:>7.4f} {val_thr:>5.3f} "
        f"{lr_bb:>10.2e} {lr_hd:>10.2e} {elapsed:>6.0f} {'*' if is_best else '':>5}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def train_one_fold(
    cfg: dict,
    train_bags: list[dict],
    val_bags: list[dict],
    fold: int,
    output_dir: Path,
) -> dict:
    """
    1-fold 分の学習を実行し、val の OOF 予測と評価指標を返す。

    Args:
        cfg: load_config() で読み込んだ設定 dict
        train_bags: 学習用 bag リスト
        val_bags: val 用 bag リスト
        fold: fold インデックス（ログ・保存用）
        output_dir: モデル・ログの保存先

    Returns:
        {fold, val_samples, val_vertebrae, val_labels, val_probs, best_auprc, best_epoch}
    """
    tc = cfg["training"]
    device = resolve_device(tc.get("gpu_id", 0))
    aug_cfg = cfg.get("augmentation", {})
    train_loader, val_loader = build_data_loaders(
        train_bags,
        val_bags,
        tc,
        aug_cfg,
    )
    model = build_model(tc, device, default_dropout=0.3)

    optimizer = make_optimizer(
        model,
        backbone_lr=tc.get("backbone_lr", 5e-5),
        head_lr=tc.get("head_lr", 2e-4),
        weight_decay=tc.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: lr_scale(ep, tc.get("warmup_epochs", 2)),
    )
    scaler = GradScaler()

    max_epochs = tc.get("max_epochs", 60)
    patience = tc.get("early_stop_patience", 10)
    grad_accum = tc.get("grad_accum", 4)
    grad_clip = tc.get("grad_clip", 1.0)

    best_auprc = -1.0
    best_epoch = 0
    no_improve = 0

    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "training.log"
    _write_log_header(log_path, fold)

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # --- 学習 ---
        model.train()
        optimizer.zero_grad()
        accum_count = 0
        total_loss_sum = 0.0

        for batch_idx, (stacks, label, _sample, vertebra) in enumerate(train_loader):
            stacks = stacks[0].to(device)
            label_val = float(label[0])
            labels_t = torch.tensor([label_val], device=device)
            vertebra_index = VERTEBRA_TO_INDEX[vertebra[0]]

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(stacks, vertebra_index)  # [t]
                loss, _ = batch_bag_mean_loss([logits], labels_t)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_count += 1
            total_loss_sum += loss.item() * grad_accum

            if accum_count % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        n_batches = len(train_loader)
        avg_train_loss = total_loss_sum / n_batches
        backbone_lr_now = optimizer.param_groups[0]["lr"]
        head_lr_now = optimizer.param_groups[1]["lr"]
        scheduler.step()

        # --- 検証 ---
        val_probs_ep, val_labels_ep, val_loss = _run_val(model, val_loader, device)

        y_true_np = np.array(val_labels_ep)
        y_prob_np = np.array(val_probs_ep)
        val_auprc, val_auroc = compute_ranking_metrics(
            val_labels_ep,
            val_probs_ep,
        )
        opt_thr = find_optimal_threshold(y_true_np, y_prob_np)
        val_f1 = _prf_at_threshold(y_true_np, y_prob_np, opt_thr)["f1"]

        elapsed = time.time() - epoch_start
        is_best = val_auprc > best_auprc

        print(
            f"[Fold {fold}] Epoch {epoch:3d} | "
            f"loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_AUPRC={val_auprc:.4f} AUROC={val_auroc:.4f} "
            f"F1={val_f1:.4f}(thr={opt_thr:.3f}) | "
            f"lr_bb={backbone_lr_now:.2e} lr_hd={head_lr_now:.2e} | "
            f"{elapsed:.0f}s{'  ← best' if is_best else ''}"
        )
        _append_log(
            log_path,
            fold,
            epoch,
            avg_train_loss,
            val_loss,
            val_auprc,
            val_auroc,
            val_f1,
            opt_thr,
            backbone_lr_now,
            head_lr_now,
            elapsed,
            is_best,
        )

        if is_best:
            best_auprc = val_auprc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                msg = f"[Fold {fold}] Early stop at epoch {epoch} (best={best_epoch}, AUPRC={best_auprc:.4f})"
                print(msg)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                break

    # 最良モデルで val を再評価（OOF 予測取得）
    model.load_state_dict(torch.load(fold_dir / "best_model.pt", weights_only=True))
    val_probs, val_labels_final, _ = _run_val(model, val_loader, device)

    return {
        "fold": fold,
        "val_samples": [b["sample"] for b in val_bags],
        "val_vertebrae": [b["vertebra"] for b in val_bags],
        "val_labels": val_labels_final,
        "val_probs": val_probs,
        "best_auprc": best_auprc,
        "best_epoch": best_epoch,
    }


def _run_val(
    model: FractureResNet18,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[int], float]:
    """val セットの予測確率・ラベル・val loss を返す。"""
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for stacks, label, _sample, vertebra in val_loader:
            stacks = stacks[0].to(device)
            label_val = float(label[0])
            labels_t = torch.tensor([label_val], device=device)
            vertebra_index = VERTEBRA_TO_INDEX[vertebra[0]]

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(stacks, vertebra_index)  # [t]
                loss, _ = batch_bag_mean_loss([logits], labels_t)

            total_loss += loss.item()
            bag_prob = float(torch.sigmoid(logits.mean()).cpu())
            probs.append(bag_prob)
            labels.append(int(label[0]))

    avg_val_loss = total_loss / max(1, len(val_loader))
    return probs, labels, avg_val_loss
