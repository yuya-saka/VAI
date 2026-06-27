"""GradCAM interactive viewer for stage1 classifier.

操作:
  →  / n  : 次のサンプル
  ←  / p  : 前のサンプル
  s        : 現在の画像を PNG 保存
  q        : 終了
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "train_models").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml
import torch
import matplotlib_fontja  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from train_models.stage1.src.data_utils import (
    collect_items, split_test_holdout, split_items_cv, set_seed,
)
from train_models.stage1.src.model import TimmModel

OUTPUTS_DIR = PROJECT_ROOT / 'train_models/stage1/outputs/baseline/v1_batch8'
DATASET_DIR = PROJECT_ROOT / 'data/rsna_data/fracture_dataset'
CSV_PATH    = PROJECT_ROOT / 'data/rsna_data/train.csv'
BBOX_CSV_PATH = PROJECT_ROOT / 'data/rsna_data/train_bounding_boxes.csv'
META_DIR    = PROJECT_ROOT / 'data/rsna_data/processing_metadata'


def load_model(fold: int, device: torch.device) -> TimmModel:
    with (OUTPUTS_DIR / 'config.yaml').open() as f:
        cfg = yaml.safe_load(f)
    mc, dc = cfg['model'], cfg['data']
    model = TimmModel(
        backbone       = mc['backbone'],
        in_chans       = dc['in_channels'],
        n_slices       = dc['n_slices'],
        drop_rate      = mc['drop_rate'],
        drop_path_rate = mc['drop_path_rate'],
        drop_rate_last = mc['drop_rate_last'],
        lstm_hidden    = mc['lstm_hidden'],
        lstm_layers    = mc['lstm_layers'],
        out_dim        = mc['out_dim'],
        pretrained     = False,
    ).to(device)
    ckpt = OUTPUTS_DIR / f'fold{fold}' / 'best_model.pt'
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f'loaded: {ckpt}')
    return model


def compute_gradcam(
    model: TimmModel,
    ct: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([ct, mask[:, np.newaxis]], axis=1)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device) / 255.0

    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_, __, out):  activations['v'] = out
    def bwd_hook(_, __, gout): gradients['v'] = gout[0]

    fh = model.encoder.bn2.register_forward_hook(fwd_hook)
    bh = model.encoder.bn2.register_full_backward_hook(bwd_hook)

    cams, probs = [], []
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        for s in range(ct.shape[0]):
            model.zero_grad()
            activations.clear()
            gradients.clear()
            logits = model(x_t)
            probs.append(torch.sigmoid(logits[0, s]).item())
            logits[0, s].backward(retain_graph=(s < ct.shape[0] - 1))

            act = activations['v'][s].detach()
            grad = gradients['v'][s].detach()
            cam = torch.relu((grad.mean(dim=(1, 2))[:, None, None] * act).sum(0))
            cam = cam.cpu().numpy()
            cams.append(cam / cam.max() if cam.max() > 0 else cam)

    fh.remove()
    bh.remove()
    return np.stack(cams), np.array(probs)


def build_figure(
    study_uid: str,
    vertebra: str,
    cams: np.ndarray,
    probs: np.ndarray,
    ct: np.ndarray,
    mask: np.ndarray,
    bbox_forced: set[int],
    orientation_forced: set[int],
    bbox_overlays: dict[int, list[np.ndarray]] | int | None = None,
    gt: int | float | None = None,
    pred_prob: float | str | None = None,
    cat: str | int | None = None,
    idx: int | None = None,
    total: int | None = None,
    display_indices: list[int] | None = None,
) -> plt.Figure:
    if not isinstance(bbox_overlays, dict):
        if total is not None:
            raise TypeError("bbox_overlays must be a dict when total is provided")
        total = int(idx) if idx is not None else 0
        idx = int(cat) if cat is not None else 0
        cat = str(pred_prob)
        pred_prob = float(gt) if gt is not None else float('nan')
        gt = int(bbox_overlays) if bbox_overlays is not None else -1
        bbox_overlays = {}

    gt = int(gt) if gt is not None else -1
    pred_prob = float(pred_prob) if pred_prob is not None else float('nan')
    cat = str(cat) if cat is not None else '?'
    idx = int(idx) if idx is not None else 0
    total = int(total) if total is not None else 0

    all_indices = list(range(len(probs)))
    indices = all_indices if display_indices is None else display_indices
    if not indices:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.text(
            0.5,
            0.5,
            f'[{idx+1}/{total}]  {study_uid} / {vertebra}\nNo selected slices',
            ha='center',
            va='center',
            color='white',
            fontsize=12,
        )
        ax.axis('off')
        return fig
    cols = min(5, max(1, len(indices)))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 1))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f'[{idx+1}/{total}]  {study_uid}  /  {vertebra}'
        f'    GT={gt}  pred={pred_prob:.3f}  [{cat}]'
        f'\n← → で移動  |  s で保存  |  q で終了',
        fontsize=9, color='white',
    )
    fig.patch.set_facecolor('#1a1a1a')

    for ax, s in zip(axes, indices, strict=False):
        ct_gray = ct[s, 2].astype(np.float32) / 255.0
        ct_rgb  = np.stack([ct_gray] * 3, axis=-1)
        cam_up  = cv2.resize(cams[s], (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap = plt.cm.jet(cam_up)[:, :, :3]
        blended = 0.55 * ct_rgb + 0.45 * heatmap

        contours, _ = cv2.findContours(
            (mask[s] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        blended_u8 = (blended * 255).clip(0, 255).astype(np.uint8)
        cv2.drawContours(blended_u8, contours, -1, (0, 255, 0), 1)
        for polygon in bbox_overlays.get(s, []):
            cv2.polylines(
                blended_u8,
                [polygon.astype(np.int32)],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )

        ax.imshow(blended_u8)
        ax.set_xticks([]); ax.set_yticks([])
        is_bbox = s in bbox_forced
        is_orientation = s in orientation_forced
        tags = []
        if is_orientation:
            tags.append('orient')
        if is_bbox:
            tags.append('bbox')
        ax.set_title(
            f's{s}  p={probs[s]:.2f}' + (f" [{'|'.join(tags)}]" if tags else ''),
            fontsize=8,
            color='cyan' if is_orientation else 'red' if is_bbox else 'white',
            backgroundcolor='#333333',
            pad=2,
        )
        for sp in ax.spines.values():
            if is_orientation:
                sp.set_edgecolor('cyan');   sp.set_linewidth(3.0)
            elif is_bbox:
                sp.set_edgecolor('red');    sp.set_linewidth(2.5)
            elif probs[s] >= 0.5:
                sp.set_edgecolor('yellow'); sp.set_linewidth(2.0)
            else:
                sp.set_edgecolor('#444');   sp.set_linewidth(0.5)

    for ax in axes[len(indices):]:
        ax.axis('off')

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    ax_bar = fig.add_axes((0.05, 0.01, 0.9, 0.04))
    ax_bar.bar(
        all_indices, probs,
        color=[
            'cyan' if s in orientation_forced
            else 'red' if s in bbox_forced
            else 'steelblue'
            for s in all_indices
        ],
        alpha=0.85,
    )
    ax_bar.axhline(0.5, color='yellow', lw=1, ls='--')
    ax_bar.set_xlim(-0.5, len(probs) - 0.5); ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks(all_indices)
    ax_bar.set_xticklabels([str(i) for i in all_indices], fontsize=7)
    ax_bar.set_ylabel('prob', fontsize=7)
    ax_bar.tick_params(axis='y', labelsize=7)
    ax_bar.set_facecolor('#222')
    return fig


def build_oof(fold: int) -> pd.DataFrame:
    oof_df = pd.read_csv(OUTPUTS_DIR / 'oof_predictions.csv')
    set_seed(42)
    all_items = collect_items(DATASET_DIR, CSV_PATH)
    tv_items, _ = split_test_holdout(all_items, test_size=0.2, seed=42)
    study_to_fold: dict[str, int] = {}
    for f in range(5):
        _, val = split_items_cv(tv_items, n_splits=5, val_fold=f, seed=42)
        for it in val:
            study_to_fold[it['study_uid']] = f
    oof_df['fold'] = oof_df['study_uid'].map(study_to_fold)
    oof_df['category'] = oof_df.apply(
        lambda r: (
            'TP' if r['label'] == 1 and r['pred_prob'] >= 0.5
            else 'FN' if r['label'] == 1 and r['pred_prob'] < 0.5
            else 'FP' if r['label'] == 0 and r['pred_prob'] >= 0.5
            else 'TN'
        ),
        axis=1,
    )
    return oof_df


def get_bbox_forced(study_uid: str, vertebra: str) -> set[int]:
    meta_path = META_DIR / f'{study_uid}.json'
    if not meta_path.exists():
        return set()
    vdata = json.loads(meta_path.read_text()).get('vertebrae', {}).get(vertebra, {})
    return {
        p['sequence_index']
        for p in vdata.get('classifier_planes', {}).get('planes', [])
        if p.get('bbox_forced')
    }


def get_orientation_forced(study_uid: str, vertebra: str) -> set[int]:
    meta_path = META_DIR / f'{study_uid}.json'
    if not meta_path.exists():
        return set()
    vdata = json.loads(meta_path.read_text()).get('vertebrae', {}).get(vertebra, {})
    return {
        p['sequence_index']
        for p in vdata.get('classifier_planes', {}).get('planes', [])
        if p.get('max_area_forced')
    }


def get_bbox_overlays(study_uid: str, vertebra: str) -> dict[int, list[np.ndarray]]:
    meta_path = META_DIR / f'{study_uid}.json'
    if not meta_path.exists() or not BBOX_CSV_PATH.exists():
        return {}

    metadata = json.loads(meta_path.read_text())
    vdata = metadata.get('vertebrae', {}).get(vertebra, {})
    planes = vdata.get('classifier_planes', {}).get('planes', [])
    if not planes:
        return {}

    bbox_df = pd.read_csv(BBOX_CSV_PATH)
    bbox_df = bbox_df[bbox_df['StudyInstanceUID'] == study_uid]
    if bbox_df.empty:
        return {}

    output_directory = vdata.get('sampling', {})
    output_size = output_directory.get('output_size_row_column', [224, 224])
    output_spacing = output_directory.get('pixel_spacing_row_column_mm', [0.4, 0.4])
    height, width = int(output_size[0]), int(output_size[1])
    row_spacing, column_spacing = float(output_spacing[0]), float(output_spacing[1])

    dicom_geometry = metadata.get('dicom_geometry', {})
    row_direction = np.asarray(dicom_geometry.get('row_direction_lps'), dtype=np.float64)
    column_direction = np.asarray(dicom_geometry.get('column_direction_lps'), dtype=np.float64)
    pixel_spacing = dicom_geometry.get('pixel_spacing_row_column_mm', [1.0, 1.0])
    dicom_row_spacing = float(pixel_spacing[0])
    dicom_column_spacing = float(pixel_spacing[1])
    origins = {
        int(Path(slice_data['source_file']).stem): np.asarray(
            slice_data['image_position_lps_mm'],
            dtype=np.float64,
        )
        for slice_data in dicom_geometry.get('slices', [])
    }

    overlays: dict[int, list[np.ndarray]] = {}
    for plane in planes:
        sequence_index = int(plane['sequence_index'])
        slice_numbers = set(int(value) for value in plane.get('bbox_slice_numbers', []))
        if not slice_numbers:
            continue

        center = np.asarray(plane['center_lps_mm'], dtype=np.float64)
        plane_row = np.asarray(plane['row_basis_lps'], dtype=np.float64)
        plane_column = np.asarray(plane['column_basis_lps'], dtype=np.float64)

        for _, row in bbox_df[bbox_df['slice_number'].isin(slice_numbers)].iterrows():
            slice_number = int(row['slice_number'])
            origin = origins.get(slice_number)
            if origin is None:
                continue

            x0 = float(row['x'])
            y0 = float(row['y'])
            x1 = x0 + float(row['width'])
            y1 = y0 + float(row['height'])
            corners = np.asarray(
                [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ],
                dtype=np.float64,
            )
            patient_points = np.asarray(
                [
                    origin
                    + x * dicom_column_spacing * row_direction
                    + y * dicom_row_spacing * column_direction
                    for x, y in corners
                ],
                dtype=np.float64,
            )
            deltas = patient_points - center
            column_offsets = deltas @ plane_row
            row_offsets = deltas @ plane_column
            polygon = np.stack(
                [
                    column_offsets / column_spacing + (width - 1) / 2.0,
                    row_offsets / row_spacing + (height - 1) / 2.0,
                ],
                axis=1,
            )
            overlays.setdefault(sequence_index, []).append(polygon)

    return overlays


def has_selected_slices(row: pd.Series, *, orientation_only: bool, bbox_only: bool) -> bool:
    if bbox_only:
        return bool(get_bbox_forced(str(row['study_uid']), str(row['vertebra'])))
    if orientation_only:
        return bool(get_orientation_forced(str(row['study_uid']), str(row['vertebra'])))
    return True


def run_viewer(
    samples: list[dict],
    model: TimmModel,
    oof_df: pd.DataFrame,
    device: torch.device,
    start_idx: int = 0,
    orientation_only: bool = False,
    bbox_only: bool = False,
) -> None:
    state = {'idx': start_idx, 'fig': None}
    total = len(samples)

    def show(idx: int) -> None:
        if state['fig'] is not None:
            plt.close(state['fig'])

        row = samples[idx]
        study_uid = row['study_uid']
        vertebra  = row['vertebra']
        print(f'[{idx+1}/{total}] {study_uid} / {vertebra}  (computing GradCAM...)')

        ct   = np.load(DATASET_DIR / study_uid / vertebra / 'ct.npy',            allow_pickle=False)
        mask = np.load(DATASET_DIR / study_uid / vertebra / 'vertebra_mask.npy', allow_pickle=False)
        bbox_forced = get_bbox_forced(study_uid, vertebra)
        orientation_forced = get_orientation_forced(study_uid, vertebra)
        bbox_overlays = get_bbox_overlays(study_uid, vertebra)
        display_indices = (
            sorted(bbox_forced) if bbox_only
            else sorted(orientation_forced) if orientation_only
            else None
        )
        cams, probs = compute_gradcam(model, ct, mask, device)

        oof_row = oof_df[(oof_df['study_uid'] == study_uid) & (oof_df['vertebra'] == vertebra)]
        gt        = int(oof_row['label'].values[0])       if len(oof_row) else -1
        pred_prob = float(oof_row['pred_prob'].values[0]) if len(oof_row) else float('nan')
        cat       = oof_row['category'].values[0]         if len(oof_row) else '?'

        fig = build_figure(
            study_uid, vertebra, cams, probs, ct, mask,
            bbox_forced, orientation_forced, bbox_overlays, gt, pred_prob, cat, idx, total,
            display_indices=display_indices,
        )
        state['fig'] = fig
        state['study_uid'] = study_uid
        state['vertebra']  = vertebra

        def on_key(event: plt.matplotlib.backend_bases.KeyEvent) -> None:
            if event.key in ('right', 'n'):
                state['idx'] = min(state['idx'] + 1, total - 1)
                show(state['idx'])
            elif event.key in ('left', 'p'):
                state['idx'] = max(state['idx'] - 1, 0)
                show(state['idx'])
            elif event.key == 's':
                out = Path(f"gradcam_{state['study_uid']}_{state['vertebra']}.png")
                state['fig'].savefig(out, dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
                print(f'saved: {out}')
            elif event.key == 'q':
                plt.close('all')

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)

    show(state['idx'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',     type=int, default=0)
    parser.add_argument('--category', default='FN', choices=['FN', 'FP', 'TP', 'TN'])
    parser.add_argument('--idx',      type=int, default=0, help='開始インデックス')
    parser.add_argument('--n',        type=int, default=50, help='最大表示件数')
    parser.add_argument('--gpu',      type=int, default=1)
    parser.add_argument('--list',     action='store_true', help='一覧表示して終了')
    parser.add_argument(
        '--orientation-only',
        action='store_true',
        help='向き補正に使用したスライス(max_area_forced)のみ表示',
    )
    parser.add_argument(
        '--bbox-only',
        action='store_true',
        help='bboxがあるスライス(bbox_forced)のみ表示し、bboxを重ねる',
    )
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    oof_df = build_oof(args.fold)

    ascending = args.category in ('FN', 'TN')
    filtered_base = (
        oof_df[(oof_df['category'] == args.category) & (oof_df['fold'] == args.fold)]
        .sort_values('pred_prob', ascending=ascending)
    )
    if args.bbox_only or args.orientation_only:
        filtered_base = filtered_base[
            filtered_base.apply(
                has_selected_slices,
                axis=1,
                orientation_only=args.orientation_only,
                bbox_only=args.bbox_only,
            )
        ]
    filtered = filtered_base.head(args.n).reset_index(drop=True)

    if args.list:
        print(filtered[['study_uid', 'vertebra', 'pred_prob']].to_string())
        return

    print(f'{args.category} x fold{args.fold}: {len(filtered)} 件')
    model = load_model(args.fold, device)
    samples = filtered.to_dict('records')
    run_viewer(
        samples,
        model,
        oof_df,
        device,
        start_idx=args.idx,
        orientation_only=args.orientation_only,
        bbox_only=args.bbox_only,
    )


if __name__ == '__main__':
    main()
