"""訓練ループ・評価・テスト評価パイプライン（セグメンテーション専用版）"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.losses import build_class_weights, compute_seg_only_loss
from ..utils.metrics import compute_seg_fg_metrics
from ..utils.visualization import save_seg_overlay
from .data_utils import (
    create_data_loaders,
    create_model_optimizer_scheduler,
    prepare_datasets_and_splits,
)


def _resolve_output_base(cfg: dict[str, Any], script_dir: Path) -> Path | None:
    """experiment セクションが存在すればベースパスを返す"""
    exp = cfg.get('experiment')
    if exp and exp.get('phase') and exp.get('name'):
        return script_dir / 'outputs' / exp['phase'] / exp['name']
    return None


def _get_wandb():
    """wandb を遅延インポート"""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """モデル評価を実行し、セグメンテーション主要メトリクスを返す"""
    model.eval()
    loss_cfg = cfg.get('loss', {})
    gamma_dice = float(loss_cfg.get('gamma_dice', 0.4))

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    dice_fg_sum = 0.0
    n = 0

    fg_mdice_sum = 0.0
    fg_miou_sum = 0.0
    miou_sum = 0.0
    per_class_dice: dict[str, list[float]] = {}
    per_class_iou: dict[str, list[float]] = {}
    sample_count = 0

    for batch in loader:
        x = batch['image'].to(device).float()
        gt_mask = batch['gt_region_mask'].to(device).long()

        out = model(x)
        seg_logits = out['seg_logits']

        loss_dict = compute_seg_only_loss(
            seg_logits=seg_logits,
            gt_mask=gt_mask,
            class_weights=class_weights,
            gamma_dice=gamma_dice,
        )

        total_loss_sum += loss_dict['total'].item()
        ce_loss_sum += loss_dict['ce_loss'].item()
        dice_fg_sum += loss_dict['dice_fg_loss'].item()
        n += 1

        seg_metrics = compute_seg_fg_metrics(seg_logits, gt_mask)
        bs = x.shape[0]
        fg_mdice_sum += seg_metrics['fg_mdice'] * bs
        fg_miou_sum += seg_metrics['fg_miou'] * bs
        miou_sum += seg_metrics['miou'] * bs
        sample_count += bs

        for cls_name, vals in seg_metrics['per_class'].items():
            per_class_dice.setdefault(cls_name, []).append(vals['dice'])
            per_class_iou.setdefault(cls_name, []).append(vals['iou'])

    avg_per_class = {
        cls_name: {
            'dice': float(np.mean(dices)),
            'iou': float(np.mean(per_class_iou[cls_name])),
        }
        for cls_name, dices in per_class_dice.items()
    }

    return {
        'val_loss': total_loss_sum / max(1, n),
        'val_ce_loss': ce_loss_sum / max(1, n),
        'val_dice_fg_loss': dice_fg_sum / max(1, n),
        'fg_mdice': float(fg_mdice_sum / max(1, sample_count)),
        'fg_miou': float(fg_miou_sum / max(1, sample_count)),
        'miou': float(miou_sum / max(1, sample_count)),
        'per_class': avg_per_class,
    }


def run_training_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict[str, Any],
    best_path: Path,
    class_weights: torch.Tensor | None = None,
    wandb_enabled: bool = False,
    _wandb=None,
) -> None:
    """訓練ループを実行（早期停止あり）"""
    tr_cfg = cfg.get('training', {})
    loss_cfg = cfg.get('loss', {})

    epochs = int(tr_cfg.get('epochs', 200))
    es_pat = int(tr_cfg.get('early_stopping_patience', 20))
    grad_clip = float(tr_cfg.get('grad_clip', 1.0))
    gamma_dice = float(loss_cfg.get('gamma_dice', 0.4))

    best_val = float('inf')
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        train_loss_sum = 0.0
        steps = 0

        for batch in train_loader:
            x = batch['image'].to(device).float()
            gt_mask = batch['gt_region_mask'].to(device).long()

            out = model(x)
            loss_dict = compute_seg_only_loss(
                seg_logits=out['seg_logits'],
                gt_mask=gt_mask,
                class_weights=class_weights,
                gamma_dice=gamma_dice,
            )
            loss = loss_dict['total']

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            train_loss_sum += loss.item()
            steps += 1

        train_loss = train_loss_sum / max(1, steps)
        val_metrics = evaluate(model, val_loader, device, cfg, class_weights=class_weights)
        scheduler.step(val_metrics['val_loss'])

        cur_lr = opt.param_groups[0]['lr']
        elapsed = int(round(time.time() - t0))
        print(
            f'[EPOCH {ep:03d}/{epochs}] '
            f'lr={cur_lr:.2e} '
            f'train_loss={train_loss:.6f} '
            f'val_loss={val_metrics["val_loss"]:.6f} '
            f'fg_mdice={val_metrics["fg_mdice"]:.4f} '
            f'fg_miou={val_metrics["fg_miou"]:.4f} '
            f'miou={val_metrics["miou"]:.4f} '
            f'time={elapsed}s'
        )

        if wandb_enabled and _wandb is not None:
            _wandb.log({
                'epoch': ep, 'lr': cur_lr,
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'val_ce_loss': val_metrics['val_ce_loss'],
                'val_dice_fg_loss': val_metrics['val_dice_fg_loss'],
                'fg_mdice': val_metrics['fg_mdice'],
                'fg_miou': val_metrics['fg_miou'],
                'miou': val_metrics['miou'],
            }, step=ep)

        if val_metrics['val_loss'] < best_val - 1e-8:
            best_val = val_metrics['val_loss']
            no_improve = 0
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'val': val_metrics}, best_path)
            print(f'  [SAVE] best -> {best_path} (val_loss={best_val:.6f}, fg_mdice={val_metrics["fg_mdice"]:.4f})')
            if wandb_enabled and _wandb is not None:
                _wandb.run.summary['best_val_loss'] = best_val
                _wandb.run.summary['best_fg_mdice'] = val_metrics['fg_mdice']
                _wandb.run.summary['best_fg_miou'] = val_metrics['fg_miou']
        else:
            no_improve += 1
            if no_improve >= es_pat:
                print(f'[EARLY STOP] no improvement for {es_pat} epochs. best_val={best_val:.6f}')
                break


@torch.no_grad()
def save_seg_examples(
    model: nn.Module,
    loader,
    device: torch.device,
    out_dir: Path,
    tag: str = 'TEST',
) -> None:
    """セグメンテーション可視化を保存する"""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        x = batch['image'].to(device).float()
        gt_mask = batch['gt_region_mask'].to(device).long()

        out = model(x)
        pred_mask = out['seg_logits'].argmax(dim=1)

        x_np = x.cpu().numpy()
        gt_mask_np = gt_mask.cpu().numpy()
        pred_mask_np = pred_mask.cpu().numpy()

        for i in range(x_np.shape[0]):
            ct01 = x_np[i, 0]
            name = (
                f'{batch["sample"][i]}_{batch["vertebra"][i]}_'
                f'slice{int(batch["slice_idx"][i]):03d}'
            )
            save_seg_overlay(
                ct=ct01,
                pred_mask=pred_mask_np[i].astype(np.int32),
                gt_mask=gt_mask_np[i].astype(np.int32),
                out_path=out_dir / f'{tag}_{name}_seg.png',
            )


def train_one_fold(cfg: dict[str, Any]) -> dict[str, Any]:
    """1つの fold に対して学習・評価・可視化を実行する"""
    import tempfile
    tempfile.tempdir = '/tmp'

    data_cfg = cfg.get('data', {})
    tr_cfg = cfg.get('training', {})
    loss_cfg = cfg.get('loss', {})

    test_fold = int(data_cfg.get('test_fold', 0))

    train_s, val_s, test_s, root_dir, group, image_size, seed = prepare_datasets_and_splits(cfg)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, seed, cfg,
    )

    gpu_id = int(tr_cfg.get('gpu_id', 0))
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device={device}')

    wandb_cfg = cfg.get('wandb', {})
    wandb_enabled = bool(wandb_cfg.get('enabled', False))
    _wandb = None
    if wandb_enabled:
        _wandb = _get_wandb()
        if _wandb is None:
            print('[WARNING] wandb.enabled=true だが wandb がインストールされていません。')
            wandb_enabled = False
        else:
            run_name = wandb_cfg.get('run_name') or f'fold{test_fold}'
            project = wandb_cfg.get('project') or 'vai-unet-seg_only'
            _wandb.init(project=project, name=run_name, config=cfg, reinit=True)

    bg_weight = float(loss_cfg.get('background_weight', 0.3))
    class_weights = build_class_weights(background_weight=bg_weight, device=device)

    model, opt, scheduler = create_model_optimizer_scheduler(cfg, device)

    script_dir = Path(__file__).resolve().parent.parent.parent
    output_base = _resolve_output_base(cfg, script_dir)
    ckpt_dir = (output_base / 'checkpoints') if output_base else (script_dir / tr_cfg.get('checkpoint_dir', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f'best_fold{test_fold}.pt'

    run_training_loop(
        model, opt, scheduler, train_loader, val_loader,
        device, cfg, best_path, class_weights=class_weights,
        wandb_enabled=wandb_enabled, _wandb=_wandb,
    )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    else:
        print('[WARNING] No best checkpoint saved. Using current model state.')

    test_metrics = evaluate(model, test_loader, device, cfg, class_weights=class_weights)
    print(
        f'[TEST] fold={test_fold} '
        f'loss={test_metrics["val_loss"]:.6f} '
        f'fg_mdice={test_metrics["fg_mdice"]:.4f} '
        f'fg_miou={test_metrics["fg_miou"]:.4f}'
    )

    vis_base = (output_base / 'vis') if output_base else (script_dir / cfg.get('evaluation', {}).get('visualization_dir', 'vis'))
    seg_vis_dir = vis_base / f'fold{test_fold}' / 'test_seg'
    save_seg_examples(model, test_loader, device, seg_vis_dir, tag='TEST')
    print(f'[INFO] seg overlays saved to {seg_vis_dir}/')

    print('\n' + '=' * 60)
    print('[SEG ONLY EVALUATION]')
    print('=' * 60)
    print(f'  fg mDice (primary): {test_metrics["fg_mdice"]:.4f}')
    print(f'  fg mIoU:            {test_metrics["fg_miou"]:.4f}')
    print(f'  mIoU (with bg):     {test_metrics["miou"]:.4f}')
    print('\n[Per-Class Results]')
    for cls_name, vals in test_metrics.get('per_class', {}).items():
        print(f'  {cls_name}: dice={vals["dice"]:.4f}  iou={vals["iou"]:.4f}')
    print('=' * 60)

    if wandb_enabled and _wandb is not None:
        _wandb.run.summary['test_loss'] = test_metrics['val_loss']
        _wandb.run.summary['test_fg_mdice'] = test_metrics['fg_mdice']
        _wandb.run.summary['test_fg_miou'] = test_metrics['fg_miou']
        _wandb.finish()

    return {
        'test_loss': test_metrics['val_loss'],
        'test_fg_mdice': test_metrics['fg_mdice'],
        'test_fg_miou': test_metrics['fg_miou'],
        'test_miou': test_metrics['miou'],
        'per_class': test_metrics.get('per_class', {}),
    }
