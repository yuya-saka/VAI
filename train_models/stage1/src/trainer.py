"""
学習ループ・1fold 学習オーケストレーター。

Unet/line_only/src/trainer.py + learning/src/trainer.py のハイブリッド。
AMP + mixup + BCE損失 + early stopping。
"""

from __future__ import annotations

import json
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
from utils.losses import criterion

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


def _amp_settings(cfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve autocast enablement and dtype with an explicit BF16 safety check."""
    tr_cfg = cfg.get("training", {})
    enabled = bool(tr_cfg.get("use_amp", True)) and device.type == "cuda"
    dtype_name = str(tr_cfg.get("amp_dtype", "bfloat16")).lower()
    dtype_by_name = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported amp_dtype: {dtype_name}")
    dtype = dtype_by_name[dtype_name]
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "amp_dtype=bfloat16 requires a BF16-capable CUDA device; "
            "set training.use_amp=false or training.amp_dtype=float16"
        )
    return enabled, dtype


def _unpack_batch(
    batch: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[str], list[str]]:
    """Support batches with optional patient labels and optional metadata."""
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None, [""] * images.shape[0], [""] * images.shape[0]
    if len(batch) == 3:
        images, labels, patient_labels = batch
        return images, labels, patient_labels, [""] * images.shape[0], [""] * images.shape[0]
    if len(batch) == 4:
        images, labels, study_uids, vertebrae = batch
        return images, labels, None, list(study_uids), list(vertebrae)
    if len(batch) == 5:
        images, labels, patient_labels, study_uids, vertebrae = batch
        return images, labels, patient_labels, list(study_uids), list(vertebrae)
    raise ValueError(f"Unsupported batch format: len={len(batch)}")


def _split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return slice logits and optional patient logits from model output."""
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def _nonfinite_gradient_names(model: nn.Module) -> list[str]:
    """Return parameter names whose gradients contain NaN or infinity."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


def _nonfinite_output_names(
    slice_logits: torch.Tensor, patient_logits: torch.Tensor | None
) -> list[str]:
    """Return model output names containing NaN or infinity."""
    names = []
    if not torch.isfinite(slice_logits).all():
        names.append("slice_logits")
    if patient_logits is not None and not torch.isfinite(patient_logits).all():
        names.append("patient_logits")
    return names


def _raise_on_nonfinite_probs(
    probs: list[float], items: list[dict], context: str
) -> None:
    """Raise with offending study/vertebra if any predicted probability is non-finite."""
    arr = np.asarray(probs, dtype=float)
    bad = np.flatnonzero(~np.isfinite(arr))
    if bad.size == 0:
        return
    offenders = [f"{items[i]['study_uid']}/{items[i]['vertebra']}" for i in bad[:20]]
    raise FloatingPointError(
        f"non-finite prediction probabilities in {context}: "
        f"{bad.size}/{arr.size} items, e.g. {offenders}"
    )


def _compute_loss(
    slice_logits: torch.Tensor,
    labels: torch.Tensor,
    patient_logits: torch.Tensor | None,
    patient_labels: torch.Tensor | None,
    positive_weight: float,
    slice_loss_weight: float,
    patient_loss_weight: float,
) -> torch.Tensor:
    """Compute weighted slice loss plus optional patient-level auxiliary loss."""
    slice_loss = criterion(slice_logits, labels, positive_weight)
    if patient_logits is None or patient_labels is None or patient_loss_weight <= 0.0:
        return slice_loss

    patient_loss = criterion(patient_logits, patient_labels, positive_weight)
    total_weight = slice_loss_weight + patient_loss_weight
    return (slice_loss * slice_loss_weight + patient_loss * patient_loss_weight) / total_weight


def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    patient_labels: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    float,
]:
    """Apply one shared mixup permutation to slice and patient labels."""
    indices = torch.randperm(images.size(0), device=images.device)
    lam = float(np.random.uniform(0.0, 1.0))
    mixed_images = images * lam + images[indices] * (1.0 - lam)
    mixed_patient_labels = patient_labels[indices] if patient_labels is not None else None
    return mixed_images, labels, labels[indices], patient_labels, mixed_patient_labels, lam


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    p_mixup: float,
    positive_weight: float,
    slice_loss_weight: float,
    patient_loss_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gradient_clip_norm: float,
    max_consecutive_amp_skips: int,
) -> float:
    """
    1 epoch の学習を実行する。

    loss または勾配が非有限になった場合、AMPのinf/nan検出による
    スキップは`max_consecutive_amp_skips`回まで許容し、それを超えるか
    forward自体が非有限を出した場合は`FloatingPointError`で即座に停止する
    （破損した重みで学習を継続し、検証時に原因不明のNaNとして
    表面化するのを防ぐため）。

    Returns:
        epoch平均 training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    consecutive_amp_skips = 0

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True, disable=not is_main)
    for batch in progress:
        images, labels, patient_labels, _, _ = _unpack_batch(batch)
        images = _prepare_images(images, device)
        labels = labels.to(device, non_blocking=True)  # (bs, 15)
        if patient_labels is not None:
            patient_labels = patient_labels.to(device, non_blocking=True)

        # mixup
        if p_mixup > 0.0 and torch.rand(1).item() < p_mixup:
            images, labels_a, labels_b, patient_labels_a, patient_labels_b, lam = _mixup_batch(
                images, labels, patient_labels
            )
            with autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                slice_logits, patient_logits = _split_model_output(model(images))
                loss = lam * _compute_loss(
                    slice_logits,
                    labels_a,
                    patient_logits,
                    patient_labels_a,
                    positive_weight,
                    slice_loss_weight,
                    patient_loss_weight,
                ) + (1 - lam) * _compute_loss(
                    slice_logits,
                    labels_b,
                    patient_logits,
                    patient_labels_b,
                    positive_weight,
                    slice_loss_weight,
                    patient_loss_weight,
                )
        else:
            with autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                slice_logits, patient_logits = _split_model_output(model(images))
                loss = _compute_loss(
                    slice_logits,
                    labels,
                    patient_logits,
                    patient_labels,
                    positive_weight,
                    slice_loss_weight,
                    patient_loss_weight,
                )

        if not torch.isfinite(loss):
            nonfinite_outputs = _nonfinite_output_names(slice_logits, patient_logits)
            raise FloatingPointError(
                f"non-finite training loss: {loss.item()} outputs={nonfinite_outputs}"
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            nonfinite_names = _nonfinite_gradient_names(model)
            scaler.step(optimizer)
            scaler.update()
            consecutive_amp_skips += 1
            if consecutive_amp_skips > max_consecutive_amp_skips:
                raise FloatingPointError(
                    f"non-finite gradients persisted for {consecutive_amp_skips} "
                    f"AMP steps parameters={nonfinite_names[:20]}"
                )
            continue
        scaler.step(optimizer)
        scaler.update()
        consecutive_amp_skips = 0

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
    slice_loss_weight: float,
    patient_loss_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    val_items: list[dict],
) -> tuple[float, dict]:
    """
    validation を実行して val_loss と epoch metrics を返す。

    vertebra-level 確率: sigmoid(logits).mean(dim=1) — 15 スライス平均。
    非有限な確率が出た場合は`val_items`（loaderと同順）から該当アイテムを
    特定して`FloatingPointError`で即座に停止する（sklearnのAUROC計算まで
    伝播させ、原因不明な`ValueError: Input contains NaN`として
    表面化させないため）。

    Returns:
        (val_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    offset = 0

    all_labels: list[int] = []
    all_probs: list[float] = []
    all_study_uids: list[str] = []
    all_vertebrae: list[str] = []

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    progress = tqdm(loader, desc="valid", leave=False, dynamic_ncols=True, disable=not is_main)
    for batch in progress:
        images, labels, patient_labels, study_uids_b, vertebrae_b = _unpack_batch(batch)
        images = _prepare_images(images, device)
        labels = labels.to(device, non_blocking=True)
        if patient_labels is not None:
            patient_labels = patient_labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, patient_logits = _split_model_output(model(images))
            loss = _compute_loss(
                logits,
                labels,
                patient_logits,
                patient_labels,
                positive_weight,
                slice_loss_weight,
                patient_loss_weight,
            )

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        # vertebra-level 確率: 15 スライスの sigmoid 平均
        # bf16のままだと numpy 変換 (`.numpy()`) が非対応のため float32 にキャストする
        probs = torch.sigmoid(logits.float()).mean(dim=1)  # (bs,)
        if not torch.isfinite(probs).all():
            bad_local = torch.nonzero(~torch.isfinite(probs), as_tuple=True)[0].tolist()
            offenders = [
                f"{val_items[offset + i]['study_uid']}/{val_items[offset + i]['vertebra']}"
                for i in bad_local[:20]
            ]
            raise FloatingPointError(
                f"non-finite validation probabilities: {len(bad_local)}/{probs.numel()} "
                f"items in batch, e.g. {offenders}"
            )
        offset += probs.shape[0]
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
    val_items: list[dict],
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
    slice_loss_weight = float(tr_cfg.get("slice_loss_weight", 15.0))
    patient_loss_weight = float(tr_cfg.get("patient_loss_weight", 1.0))
    patience = int(tr_cfg.get("early_stopping_patience", 15))
    amp_enabled, amp_dtype = _amp_settings(cfg, device)
    raw_gradient_clip_norm = tr_cfg.get("gradient_clip_norm")
    gradient_clip_norm = (
        float("inf")
        if raw_gradient_clip_norm is None
        else float(raw_gradient_clip_norm)
    )
    max_consecutive_amp_skips = int(tr_cfg.get("max_consecutive_amp_skips", 8))

    scaler = GradScaler(
        device=device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
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
            slice_loss_weight,
            patient_loss_weight,
            amp_enabled,
            amp_dtype,
            gradient_clip_norm,
            max_consecutive_amp_skips,
        )
        val_loss, metrics = _validate(
            model,
            val_loader,
            device,
            positive_weight,
            slice_loss_weight,
            patient_loss_weight,
            amp_enabled,
            amp_dtype,
            val_items,
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
        val_items,
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

    oof_preds = _collect_oof_predictions(model, val_items, cfg, device, positive_weight)

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
    amp_enabled, amp_dtype = _amp_settings(cfg, device)

    for batch in loader:
        images, _, _, _, _ = _unpack_batch(batch)
        images = _prepare_images(images, device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = _split_model_output(model(images))
        # bf16のままだと numpy 変換 (`.numpy()`) が非対応のため float32 にキャストする
        probs = torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy().tolist()
        all_probs.extend(probs)

    _raise_on_nonfinite_probs(all_probs, val_items, "OOF prediction")

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
) -> tuple[list[dict], list[dict]]:
    """
    複数 fold の best model でテストセットを推論する。

    学習・モデル選択に使っていない held-out test set に対して使う。
    アンサンブル平均は fold 間の相関を均してしまい、fold 単位の非アンサンブル
    予測（OOF と同じ「学習していないモデルによる予測」という性質を持つ）を
    捨ててしまうため、両方を返す。

    Args:
        model_paths: 各 fold の best_model.pt のパスリスト
        test_items: split_test_holdout() で分離した test アイテム
        cfg: config dict

    Returns:
        (ensemble_preds, per_fold_preds)
        - ensemble_preds: 各要素 {study_uid, vertebra, label, pred_prob}（fold 間平均）
        - per_fold_preds: 各要素 {study_uid, vertebra, label, fold, pred_prob}
          （test_items × model_paths、fold ごとの非アンサンブル予測）
    """
    from .data_utils import create_eval_data_loader
    from .model import TimmModel

    loader = create_eval_data_loader(test_items, cfg)

    # 各 fold の予測を蓄積
    all_fold_probs: list[list[float]] = []
    amp_enabled, amp_dtype = _amp_settings(cfg, device)

    for model_path in model_paths:
        ckpt = torch.load(model_path, map_location=device)
        model = TimmModel(**_extract_model_kwargs(ckpt["cfg"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        fold_probs: list[float] = []
        for batch in loader:
            images, _, _, _, _ = _unpack_batch(batch)
            images = _prepare_images(images, device)
            with autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logits, _ = _split_model_output(model(images))
            # bf16のままだと numpy 変換 (`.numpy()`) が非対応のため float32 にキャストする
            probs = torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy().tolist()
            fold_probs.extend(probs)

        _raise_on_nonfinite_probs(
            fold_probs, test_items, f"test prediction ({model_path.name})"
        )
        all_fold_probs.append(fold_probs)
        print(f"  推論完了: {model_path.name}")

    # fold 間で平均（アンサンブル）
    import numpy as np

    avg_probs = np.mean(all_fold_probs, axis=0).tolist()

    ensemble_preds = []
    for item, prob in zip(test_items, avg_probs, strict=True):
        ensemble_preds.append(
            {
                "study_uid": item["study_uid"],
                "vertebra": item["vertebra"],
                "label": item["label"],
                "pred_prob": float(prob),
            }
        )

    per_fold_preds = []
    for fold_index, fold_probs in enumerate(all_fold_probs):
        for item, prob in zip(test_items, fold_probs, strict=True):
            per_fold_preds.append(
                {
                    "study_uid": item["study_uid"],
                    "vertebra": item["vertebra"],
                    "label": item["label"],
                    "fold": fold_index,
                    "pred_prob": float(prob),
                }
            )

    return ensemble_preds, per_fold_preds


def save_fold_artifacts(
    cfg: dict, fold: int, fold_metrics: dict, oof_preds: list[dict], root: Path
) -> None:
    """1fold分のbest epoch metricsとOOF予測をディスクへ永続化する。

    別プロセス（別セッションでの再開実行など）から
    `load_or_reconstruct_fold_artifacts` で読み戻せるようにするため。
    """
    import pandas as pd

    _, fold_dir = resolve_fold_paths(cfg, fold, root)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame(oof_preds).to_csv(fold_dir / "oof_predictions.csv", index=False)


def load_or_reconstruct_fold_artifacts(
    cfg: dict,
    fold: int,
    items: list[dict],
    root: Path,
    device: torch.device,
) -> tuple[dict, list[dict]] | None:
    """1fold分のmetrics/OOF予測を読み込む。無ければbest_model.ptから再構成する。

    `save_fold_artifacts` 導入前に学習済みのfold（`metrics.json`/
    `oof_predictions.csv` が存在しない）でも、`best_model.pt` さえ残っていれば
    そのcheckpointのval_metricsとOOF再推論から復元できる。これにより、
    5foldのうち一部だけを今回のプロセスで学習した場合でも、既存の学習済みfoldを
    含めた集計が可能になる（再構成結果は次回のためにディスクへ書き戻す）。

    Returns:
        (fold_metrics, oof_preds)。そのfoldが一度も学習されていなければNone。
    """
    from .data_utils import split_items_cv

    data_cfg = cfg.get("data", {})
    n_folds = int(data_cfg.get("n_folds", 5))
    seed = int(data_cfg.get("random_seed", 42))

    best_model_path, fold_dir = resolve_fold_paths(cfg, fold, root)
    metrics_path = fold_dir / "metrics.json"
    oof_path = fold_dir / "oof_predictions.csv"

    if metrics_path.exists() and oof_path.exists():
        import pandas as pd

        with metrics_path.open("r", encoding="utf-8") as f:
            fold_metrics = json.load(f)
        oof_preds = pd.read_csv(oof_path).to_dict("records")
        return fold_metrics, oof_preds

    if not best_model_path.exists():
        return None

    ckpt = torch.load(best_model_path, map_location=device)
    fold_metrics = ckpt["val_metrics"]
    _, val_items = split_items_cv(items, n_splits=n_folds, val_fold=fold, seed=seed)
    oof_preds, _ = predict_on_items([best_model_path], val_items, cfg, device)

    save_fold_artifacts(cfg, fold, fold_metrics, oof_preds, root)
    return fold_metrics, oof_preds


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
        "use_patient_head": bool(model_cfg.get("use_patient_head", False)),
        "pretrained": False,  # 推論時は不要
    }
