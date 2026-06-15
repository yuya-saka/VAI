"""
1-fold 学習ループ。

grad accum (4 bag)・AMP fp16・differential LR・warmup・early stop を実装する。
各 epoch のログを fold_dir/training.log に出力する。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import GradScaler, autocast
from torch.utils.data import DataLoader

from learning.src.dataset import VERTEBRA_TO_INDEX
from learning.src.model import FractureResNet18
from learning.utils.losses import dmil_center_loss, select_topk
from learning.utils.training import (
    build_data_loaders,
    build_model,
    compute_ranking_metrics,
    lr_scale,
    make_optimizer,
    resolve_device,
)

_LOG_HEADER = (
    f"{'fold':>4} {'epoch':>5} {'loss':>8} {'dmil':>8} {'center':>8} "
    f"{'val_loss':>8} {'val_AUPRC':>9} {'val_AUROC':>9} "
    f"{'lr_bb':>10} {'lr_hd':>10} {'sec':>6} {'best':>5}"
)


def _write_log_header(log_path: Path, fold: int) -> None:
    """ログファイルを新規作成してヘッダーを書く。"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Fold {fold} training log\n")
        f.write(_LOG_HEADER + "\n")
        f.write("-" * len(_LOG_HEADER) + "\n")


def _append_log(
    log_path: Path,
    fold: int,
    epoch: int,
    loss: float,
    dmil: float,
    center: float,
    val_loss: float,
    val_auprc: float,
    val_auroc: float,
    lr_bb: float,
    lr_hd: float,
    elapsed: float,
    is_best: bool,
) -> None:
    """1 epoch 分のログ行を追記する。"""
    line = (
        f"{fold:>4} {epoch:>5} {loss:>8.4f} {dmil:>8.4f} {center:>8.4f} "
        f"{val_loss:>8.4f} {val_auprc:>9.4f} {val_auroc:>9.4f} "
        f"{lr_bb:>10.2e} {lr_hd:>10.2e} {elapsed:>6.0f} {'*' if is_best else '':>5}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _beta_warmup(epoch: int, warmup_epochs: int = 3) -> float:
    """center loss の beta warmup スケール (0→1 線形、warmup_epochs で完了)。"""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


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
        {
          'fold': int,
          'val_samples': list[str],
          'val_vertebrae': list[str],
          'val_labels': list[int],
          'val_probs': list[float],
          'best_auprc': float,
          'best_epoch': int,
        }
    """
    tc = cfg["training"]
    dc = cfg["data"]

    device = resolve_device(tc.get("gpu_id", 0))
    aug_cfg = cfg.get("augmentation", {})
    train_loader, val_loader = build_data_loaders(
        train_bags,
        val_bags,
        tc,
        aug_cfg,
    )

    # モデル・オプティマイザ
    model = build_model(tc, device, default_dropout=0.2)

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

    max_epochs = tc.get("max_epochs", 35)
    patience = tc.get("early_stop_patience", 8)
    grad_accum = tc.get("grad_accum", 4)
    grad_clip = tc.get("grad_clip", 1.0)
    beta = tc.get("beta", 5.0)
    beta_warmup_eps = tc.get("beta_warmup_epochs", 3)
    topk_mode = dc.get("topk_mode", "capped")
    alpha = dc.get("topk_alpha", None)

    best_auprc = -1.0
    best_epoch = 0
    no_improve = 0

    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ログファイルの初期化
    log_path = fold_dir / "training.log"
    _write_log_header(log_path, fold)

    for epoch in range(max_epochs):
        epoch_start = time.time()
        bw = _beta_warmup(epoch, beta_warmup_eps)

        # --- 学習 ---
        model.train()
        optimizer.zero_grad()
        accum_count = 0
        total_loss_sum = 0.0
        total_dmil_sum = 0.0
        total_center_sum = 0.0

        for batch_idx, (stacks, label, _sample, vertebra) in enumerate(train_loader):
            # stacks: [1, t, 3, H, W] → [t, 3, H, W]
            stacks = stacks[0].to(device)
            label_val = float(label[0])
            labels_t = torch.tensor([label_val], device=device)
            vertebra_index = VERTEBRA_TO_INDEX[vertebra[0]]

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(stacks, vertebra_index)  # [t]
                loss, breakdown = dmil_center_loss(
                    [logits],
                    labels_t,
                    beta=beta,
                    beta_warmup=bw,
                    topk_mode=topk_mode,
                    alpha=alpha,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_count += 1
            total_loss_sum += loss.item() * grad_accum
            total_dmil_sum += breakdown["dmil"]
            total_center_sum += breakdown["center"]

            if accum_count % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        n_batches = len(train_loader)
        avg_train_loss = total_loss_sum / n_batches
        avg_dmil = total_dmil_sum / n_batches
        avg_center = total_center_sum / n_batches

        # 実効 LR を記録（step() 前 = このepochで実際に使ったLR）
        backbone_lr_now = optimizer.param_groups[0]["lr"]
        head_lr_now = optimizer.param_groups[1]["lr"]

        scheduler.step()

        # --- 検証 ---
        val_probs_ep, val_labels_ep, val_loss = _run_val(
            model,
            val_loader,
            device,
            beta=beta,
            topk_mode=topk_mode,
            alpha=alpha,
        )

        val_auprc, val_auroc = compute_ranking_metrics(
            val_labels_ep,
            val_probs_ep,
        )

        elapsed = time.time() - epoch_start
        is_best = val_auprc > best_auprc

        # コンソール出力
        print(
            f"[Fold {fold}] Epoch {epoch:3d} | "
            f"loss={avg_train_loss:.4f} (dmil={avg_dmil:.4f} ctr={avg_center:.4f}) | "
            f"val_loss={val_loss:.4f} val_AUPRC={val_auprc:.4f} AUROC={val_auroc:.4f} | "
            f"lr_bb={backbone_lr_now:.2e} lr_hd={head_lr_now:.2e} | "
            f"{elapsed:.0f}s{'  ← best' if is_best else ''}"
        )

        # ファイルへ追記
        _append_log(
            log_path,
            fold,
            epoch,
            avg_train_loss,
            avg_dmil,
            avg_center,
            val_loss,
            val_auprc,
            val_auroc,
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
                msg = f"[Fold {fold}] Early stop at epoch {epoch} (best epoch={best_epoch}, best_AUPRC={best_auprc:.4f})"
                print(msg)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                break

    # 最良モデルで val を再評価（OOF 予測取得）
    model.load_state_dict(torch.load(fold_dir / "best_model.pt", weights_only=True))
    val_probs, val_labels_final, _ = _run_val(
        model,
        val_loader,
        device,
        beta=beta,
        topk_mode=topk_mode,
        alpha=alpha,
    )

    val_samples = [b["sample"] for b in val_bags]
    val_vertebrae = [b["vertebra"] for b in val_bags]

    return {
        "fold": fold,
        "val_samples": val_samples,
        "val_vertebrae": val_vertebrae,
        "val_labels": val_labels_final,
        "val_probs": val_probs,
        "best_auprc": best_auprc,
        "best_epoch": best_epoch,
    }


def _run_val(
    model: FractureResNet18,
    val_loader: DataLoader,
    device: torch.device,
    beta: float = 5.0,
    topk_mode: str = "capped",
    alpha: float | None = None,
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
                loss, _ = dmil_center_loss(
                    [logits],
                    labels_t,
                    beta=beta,
                    beta_warmup=1.0,
                    topk_mode=topk_mode,
                    alpha=alpha,
                )
            total_loss += loss.item()

            scores = torch.sigmoid(logits).cpu().float().numpy()
            k = select_topk(logits, mode=topk_mode, alpha=alpha)
            topk_scores = np.sort(scores)[-k:]
            probs.append(float(topk_scores.mean()))
            labels.append(int(label[0]))

    avg_val_loss = total_loss / max(1, len(val_loader))
    return probs, labels, avg_val_loss
