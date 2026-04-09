"""セグメンテーション専用データセット（gt_masksが存在するスライスのみ）"""

import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def vertebra_names_from_group(group: str) -> list[str]:
    """椎体グループ名から椎体名リストを返す"""
    if group == 'C1':
        return ['C1']
    if group == 'C2':
        return ['C2']
    if group == 'C3_C7':
        return ['C3', 'C4', 'C5', 'C6', 'C7']
    if group == 'ALL':
        return ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    raise ValueError(f'Unknown group: {group}')


def _is_sample_valid_seg(sample_dir: Path, vertebra_group: str) -> bool:
    """seg_only用: gt_masksが存在するスライスが1枚以上あるか確認"""
    vertebra_names = vertebra_names_from_group(vertebra_group)
    for v_name in vertebra_names:
        v_dir = sample_dir / v_name
        gt_masks_dir = v_dir / 'gt_masks'
        if not gt_masks_dir.exists():
            continue
        for gp in gt_masks_dir.glob('slice_*.png'):
            slice_idx = int(gp.stem.split('_')[1])
            ip = v_dir / 'images' / f'slice_{slice_idx:03d}.png'
            mp = v_dir / 'masks' / f'slice_{slice_idx:03d}.png'
            if ip.exists() and mp.exists():
                return True
    return False


def get_transforms(phase: str = 'train', cfg_aug: dict | None = None) -> A.Compose | A.ReplayCompose | None:
    """CT・mask・gt_mask に同じ幾何変換を適用する（ヒートマップなし）"""
    if phase != 'train':
        return None
    cfg_aug = cfg_aug or {}

    ts = []
    if cfg_aug.get('rotation', False):
        ts.append(A.Rotate(limit=float(cfg_aug.get('rotation_limit', 20)), p=0.5))
    if cfg_aug.get('scale', False):
        ts.append(A.Affine(
            scale=(1.0 - float(cfg_aug.get('scale_limit', 0.1)),
                   1.0 + float(cfg_aug.get('scale_limit', 0.1))),
            translate_percent=0.0, rotate=0, p=0.5,
            border_mode=cv2.BORDER_CONSTANT, fill=0.0, fill_mask=0.0,
        ))
    if cfg_aug.get('horizontal_flip', False):
        ts.append(A.HorizontalFlip(p=float(cfg_aug.get('horizontal_flip_prob', 0.1))))
    if cfg_aug.get('brightness_contrast', False):
        ts.append(A.RandomBrightnessContrast(
            brightness_limit=float(cfg_aug.get('brightness_limit', 0.2)),
            contrast_limit=float(cfg_aug.get('contrast_limit', 0.2)),
            p=0.5,
        ))
    if cfg_aug.get('gaussian_noise', False):
        var_lim = cfg_aug.get('noise_var_limit', [10, 50])
        ts.append(A.GaussNoise(var_limit=tuple(var_lim), p=0.3))

    additional_targets = {
        'mask': 'mask',
        'gt_mask': 'mask',
    }

    if cfg_aug.get('horizontal_flip', False):
        return A.ReplayCompose(ts, additional_targets=additional_targets)
    return A.Compose(ts, additional_targets=additional_targets)


class SegOnlyDataset(Dataset):
    """セグメンテーション専用データセット

    gt_masksが存在するスライスのみをインデックスに追加する。
    直線アノテーション（lines.json）は使用しない。

    dataset/
      sampleXX/
        C3/
          images/slice_000.png
          masks/slice_000.png
          gt_masks/slice_000.png   <- これがある場合のみ追加
    """

    def __init__(
        self,
        root_dir: Path,
        sample_names,
        group: str = 'C3_C7',
        image_size: int = 224,
        transform=None,
        cfg_aug: dict | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.sample_names = list(sample_names)
        self.group = group
        self.image_size = image_size
        self.vertebra_names = vertebra_names_from_group(group)
        self.transform = transform
        self.cfg_aug = cfg_aug or {}

        self._bad_slices = self._load_bad_slices()
        self.items: list[dict] = []
        self._build_index()
        print(f'[INFO] SegOnlyDataset: {len(self.items)} slices')

    def _load_bad_slices(self) -> set:
        """bad_slices_all.json を読み込み、除外スライスのセットを返す"""
        bad_json = self.root_dir / 'bad_slices_all.json'
        if not bad_json.exists():
            return set()
        try:
            data = json.loads(bad_json.read_text())
            entries = data if isinstance(data, list) else data.get('bad_slices', [])
            result = set()
            for entry in entries:
                slice_val = entry.get('slice_idx', entry.get('slice'))
                if slice_val is None:
                    continue
                result.add((entry['sample'], entry['vertebra'], int(slice_val)))
            if result:
                print(f'[INFO] bad_slices: {len(result)} スライスを除外')
            return result
        except Exception as e:
            print(f'[WARN] bad_slices_all.json の読み込みに失敗: {e}')
            return set()

    def _load_qc_excludes(self, vertebra_dir: Path) -> frozenset[int]:
        """qc_scores.json を読み込み、label==exclude の slice_idx 集合を返す"""
        qc_json = vertebra_dir / 'qc_scores.json'
        if not qc_json.exists():
            return frozenset()
        try:
            data = json.loads(qc_json.read_text())
            excludes = {int(e['slice_idx']) for e in data if e.get('label') == 'exclude'}
            return frozenset(excludes)
        except Exception:
            return frozenset()

    def _build_index(self) -> None:
        """gt_masksディレクトリをスキャンして有効スライスをインデックスに追加"""
        for s in self.sample_names:
            sd = self.root_dir / s
            if not sd.exists():
                continue
            for v in self.vertebra_names:
                vd = sd / v
                gt_masks_dir = vd / 'gt_masks'
                if not gt_masks_dir.exists():
                    continue
                qc_excludes = self._load_qc_excludes(vd)
                for gp in sorted(gt_masks_dir.glob('slice_*.png')):
                    slice_idx = int(gp.stem.split('_')[1])
                    if (s, v, slice_idx) in self._bad_slices:
                        continue
                    if slice_idx in qc_excludes:
                        continue
                    ip = vd / 'images' / f'slice_{slice_idx:03d}.png'
                    mp = vd / 'masks' / f'slice_{slice_idx:03d}.png'
                    if not ip.exists() or not mp.exists():
                        continue
                    self.items.append({
                        'sample': s,
                        'vertebra': v,
                        'slice_idx': slice_idx,
                        'img_path': ip,
                        'mask_path': mp,
                        'gt_mask_path': gp,
                    })

    def _did_apply_horizontal_flip(self, replay: dict[str, Any] | None) -> bool:
        """ReplayCompose の記録から水平反転の有無を判定する"""
        if replay is None:
            return False
        for tr in replay.get('transforms', []):
            if not tr.get('__class_fullname__', '').endswith('HorizontalFlip'):
                continue
            if tr.get('applied', False):
                return True
        return False

    def _swap_gt_mask_left_right(self, gt_mask: np.ndarray) -> np.ndarray:
        """水平反転後に right/left ラベルを入れ替える"""
        if gt_mask.size == 0:
            return gt_mask
        label_map = np.array([0, 1, 3, 2, 4], dtype=np.uint8)
        return label_map[gt_mask]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        it = self.items[i]

        ct = np.array(Image.open(it['img_path']).convert('L'), np.float32) / 255.0
        mk = np.array(Image.open(it['mask_path']).convert('L'), np.float32) / 255.0
        gt_mask = np.array(Image.open(it['gt_mask_path']), dtype=np.uint8)

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]

        if ct.shape != (self.image_size, self.image_size):
            ct = cv2.resize(ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if mk.shape != (self.image_size, self.image_size):
            mk = cv2.resize(mk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if gt_mask.shape != (self.image_size, self.image_size):
            gt_mask = cv2.resize(gt_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        gt_mask = np.clip(gt_mask, 0, 4).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=ct, mask=mk, gt_mask=gt_mask)
            ct = out['image']
            mk = out['mask']
            gt_mask = out['gt_mask'].astype(np.uint8)

            if isinstance(self.transform, A.ReplayCompose) and self.cfg_aug.get('horizontal_flip', False):
                did_flip = self._did_apply_horizontal_flip(out.get('replay'))
                if did_flip:
                    gt_mask = self._swap_gt_mask_left_right(gt_mask)

        ct = np.clip(ct, 0.0, 1.0).astype(np.float32)
        mk = np.clip(mk, 0.0, 1.0).astype(np.float32)
        gt_mask = np.clip(gt_mask, 0, 4).astype(np.uint8)

        x = np.stack([ct, mk], 0).astype(np.float32)  # (2, H, W)

        return {
            'image': torch.from_numpy(x),
            'gt_region_mask': torch.from_numpy(gt_mask.astype(np.int64)),
            'sample': it['sample'],
            'vertebra': it['vertebra'],
            'slice_idx': it['slice_idx'],
        }
