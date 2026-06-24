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
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Transfer uint8 images and normalize them on the target device."""
    return images.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).div_(255.0)


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
    amp_enabled = use_amp and device.type == "cuda"

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True, disable=not is_main)
    for images, labels in progress:
        images = _prepare_images(images, device)
        labels = labels.to(device, non_blocking=True)  # (bs, 15)

        # mixup
        if p_mixup > 0.0 and torch.rand(1).item() < p_mixup:
            images, labels_a, labels_b, lam = mixup(images, labels)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)  # (bs, 15)
                loss = lam * criterion(logits, labels_a, positive_weight) + (
                    1 - lam
                ) * criterion(logits, labels_b, positive_weight)
        else:
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels, positive_weight)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")

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
    amp_enabled = use_amp and device.type == "cuda"

    all_labels: list[int] = []
    all_probs: list[float] = []
    all_study_uids: list[str] = []
    all_vertebrae: list[str] = []

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(loader, desc="valid", leave=False, dynamic_ncols=True, disable=not is_main)
    for batch in progress:
        if len(batch) == 2:
            images, labels = batch
            study_uids_b = [""] * images.shape[0]
            vertebrae_b = [""] * images.shape[0]
        else:
            images, labels, study_uids_b, vertebrae_b = batch

        images = _prepare_images(images, device)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels, positive_weight)

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        # vertebra-level 確率: 15 スライスの sigmoid 平均
        probs = torch.sigmoid(logits).mean(dim=1)  # (bs,)
        # 椎体ラベルは全スライス同値なので最初のスライスを使う
        item_labels = labels[:, 0]  # (bs,)

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

    scaler = GradScaler(
        device=device.type,
        enabled=use_amp and device.type == "cuda",
    )
    write_log_header(log_path, fold)

    best_val_auroc = float("-inf")
    best_val_loss = float("inf")
    best_val_auprc = float("-inf")
    best_metrics: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # DDP時はDistributedSamplerのシャッフルシードをepochごとに更新
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            p_mixup,
            positive_weight,
            use_amp,
        )
        val_loss, metrics = _validate(
            model, val_loader, device, positive_weight, use_amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        auroc = metrics.get("auroc", float("nan"))
        auprc = metrics.get("auprc", float("nan"))
        f1 = metrics.get("at_05", {}).get("f1", float("nan"))

        # 全epoch通じたbest値を追跡
        if not (val_loss != val_loss):
            best_val_loss = min(best_val_loss, val_loss)
        if not (auprc != auprc):
            best_val_auprc = max(best_val_auprc, auprc)

        # val_auroc が nan（陽性なし）の場合は改善なし扱い
        current_auroc = auroc if not (auroc != auroc) else float("-inf")
        is_best = current_auroc > best_val_auroc
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_best:
            best_val_auroc = current_auroc
            best_metrics = metrics
            no_improve = 0
            if is_main:
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "cfg": cfg,
                        "val_metrics": metrics,
                        "best_val_loss": best_val_loss,
                        "best_val_auprc": best_val_auprc,
                    },
                    best_model_path,
                )
        else:
            no_improve += 1

        # ログ出力（rank-0のみ）
        if is_main:
            log_line = build_epoch_log(
                epoch, epochs, lr, train_loss, val_loss, metrics, elapsed
            )
            print(log_line)
            append_log(
                log_path,
                fold,
                epoch,
                train_loss,
                val_loss,
                auroc,
                auprc,
                f1,
                lr,
                elapsed,
                is_best,
            )

            if use_wandb and wandb_run is not None:
                log_wandb_epoch(wandb_run, epoch, lr, train_loss, val_loss, metrics)

        if no_improve >= patience:
            if is_main:
                print(
                    f"[EARLY STOP] {patience} epoch 改善なし (fold={fold}, epoch={epoch})"
                )
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
    train_items, val_items = split_items_cv(
        items, n_splits=n_folds, val_fold=fold, seed=seed
    )
    train_loader, val_loader = create_data_loaders(train_items, val_items, cfg)

    # モデル・最適化器
    model, optimizer, scheduler = create_model_optimizer_scheduler(cfg, device)

    # DDP時はモデルをラップ
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    print(
        f"\n{'=' * 60}\n[FOLD {fold}] train={len(train_items)} val={len(val_items)}\n{'=' * 60}"
    )
    print(
        f"[INFO] batches: train={len(train_loader)} val={len(val_loader)} "
        f"(batch_size={train_loader.batch_size}, workers={train_loader.num_workers})"
    )

    # 学習ループ
    best_metrics = run_training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        cfg,
        device,
        best_model_path,
        log_path,
        fold,
        use_wandb,
        wandb_run,
    )
    del train_loader, val_loader

    # best モデルで val_items を再推論して OOF 予測を収集
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    oof_preds = _collect_oof_predictions(
        model, val_items, cfg, device, positive_weight, use_amp
    )

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
    from .data_utils import create_eval_data_loader

    loader = create_eval_data_loader(val_items, cfg)

    model.eval()
    all_probs: list[float] = []
    amp_enabled = use_amp and device.type == "cuda"

    for images, _ in loader:
        images = _prepare_images(images, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy().tolist()
        all_probs.extend(probs)

    oof_preds = []
    for item, prob in zip(val_items, all_probs, strict=True):
        oof_preds.append(
            {
                "study_uid": item["study_uid"],
                "vertebra": item["vertebra"],
                "label": item["label"],
                "pred_prob": float(prob),
            }
        )

    return oof_preds


@torch.no_grad()
def predict_on_items(
    model_paths: list[Path],
    test_items: list[dict],
    cfg: dict,
    device: torch.device,
) -> list[dict]:
    """
    複数 fold の best model でテストセットを推論し、確率を平均する（アンサンブル）。

    学習・モデル選択に使っていない held-out test set に対して使う。

    Args:
        model_paths: 各 fold の best_model.pt のパスリスト
        test_items: split_test_holdout() で分離した test アイテム
        cfg: config dict

    Returns:
        各要素: {study_uid, vertebra, label, pred_prob}
    """
    from .data_utils import create_eval_data_loader
    from .model import TimmModel

    tr_cfg = cfg.get("training", {})
    use_amp = bool(tr_cfg.get("use_amp", True))
    loader = create_eval_data_loader(test_items, cfg)

    # 各 fold の予測を蓄積
    all_fold_probs: list[list[float]] = []
    amp_enabled = use_amp and device.type == "cuda"

    for model_path in model_paths:
        ckpt = torch.load(model_path, map_location=device)
        model = TimmModel(**_extract_model_kwargs(ckpt["cfg"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        fold_probs: list[float] = []
        for images, _ in loader:
            images = _prepare_images(images, device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy().tolist()
            fold_probs.extend(probs)

        all_fold_probs.append(fold_probs)
        print(f"  推論完了: {model_path.name}")

    # fold 間で平均（アンサンブル）
    import numpy as np

    avg_probs = np.mean(all_fold_probs, axis=0).tolist()

    test_preds = []
    for item, prob in zip(test_items, avg_probs, strict=True):
        test_preds.append(
            {
                "study_uid": item["study_uid"],
                "vertebra": item["vertebra"],
                "label": item["label"],
                "pred_prob": float(prob),
            }
        )

    return test_preds


def _extract_model_kwargs(cfg: dict) -> dict:
    """config から TimmModel のコンストラクタ引数を抽出する。"""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return {
        "backbone": str(model_cfg.get("backbone", "tf_efficientnetv2_s")),
        "in_chans": int(data_cfg.get("in_channels", 6)),
        "n_slices": int(data_cfg.get("n_slices", 15)),
        "drop_rate": float(model_cfg.get("drop_rate", 0.0)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.0)),
        "drop_rate_last": float(model_cfg.get("drop_rate_last", 0.3)),
        "lstm_hidden": int(model_cfg.get("lstm_hidden", 256)),
        "lstm_layers": int(model_cfg.get("lstm_layers", 2)),
        "out_dim": int(model_cfg.get("out_dim", 1)),
        "pretrained": False,  # 推論時は不要
    }
