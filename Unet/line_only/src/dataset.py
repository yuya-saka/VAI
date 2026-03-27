"""データセットとデータ拡張の定義"""

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils import losses as line_losses


# -------------------------
# 椎体グループ展開
# -------------------------
def vertebra_names_from_group(group: str):
    """椎体グループ名から椎体名リストを返す"""
    if group == "C1":
        return ["C1"]
    if group == "C2":
        return ["C2"]
    if group == "C3_C7":
        return ["C3", "C4", "C5", "C6", "C7"]
    if group == "ALL":
        return ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    raise ValueError(f"Unknown group: {group}")


# -------------------------
# サンプル有効性チェック
# -------------------------
def _is_sample_valid_png(sample_dir: Path, vertebra_group: str) -> bool:
    """
    PNG判定:
    - sample_dir/Cx/lines.json が存在
    - lines.json内で、4本線が揃っていて各lineが2点以上のsliceが1枚でもある
    - images/masks が存在している slice が1枚でもある
    """
    vertebra_names = vertebra_names_from_group(vertebra_group)

    for v_name in vertebra_names:
        v_dir = sample_dir / v_name
        if not v_dir.exists():
            continue

        lj = v_dir / "lines.json"
        if not lj.exists():
            continue

        try:
            lines_data = json.loads(lj.read_text())
        except Exception:
            continue

        for slice_idx_str, lines in lines_data.items():
            ok = True
            for k in ["line_1", "line_2", "line_3", "line_4"]:
                if k not in lines or lines[k] is None or len(lines[k]) < 2:
                    ok = False
                    break
            if not ok:
                continue

            slice_idx = int(slice_idx_str)
            ip = v_dir / "images" / f"slice_{slice_idx:03d}.png"
            mp = v_dir / "masks" / f"slice_{slice_idx:03d}.png"
            if ip.exists() and mp.exists():
                return True

    return False


# -------------------------
# データ拡張（オンライン）
# -------------------------
def get_transforms(phase="train", cfg_aug=None):
    """
    CT（画像）、mask（マスク）、heatmaps（画像として扱う）に同じ幾何変換を適用
    """
    if phase != "train":
        return None
    cfg_aug = cfg_aug or {}

    ts = []

    # 幾何変換
    if cfg_aug.get("rotation", False):
        ts.append(A.Rotate(limit=float(cfg_aug.get("rotation_limit", 20)), p=0.5))

    if cfg_aug.get("scale", False):
        ts.append(
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=float(cfg_aug.get("scale_limit", 0.1)),
                rotate_limit=0,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0.0,
                mask_value=0.0,
            )
        )

    # 左右反転
    if cfg_aug.get("horizontal_flip", False):
        ts.append(A.HorizontalFlip(p=float(cfg_aug.get("horizontal_flip_prob", 0.1))))

    # 輝度コントラスト（CTだけに効く想定だが、簡便のためimageに適用）
    if cfg_aug.get("brightness_contrast", False):
        ts.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(cfg_aug.get("brightness_limit", 0.2)),
                contrast_limit=float(cfg_aug.get("contrast_limit", 0.2)),
                p=0.5,
            )
        )

    # ガウシアンノイズ（CTに効く想定）
    if cfg_aug.get("gaussian_noise", False):
        var_lim = cfg_aug.get("noise_var_limit", [10, 50])
        ts.append(A.GaussNoise(var_limit=tuple(var_lim), p=0.3))

    additional_targets = {
        "mask": "mask",  # 椎体マスクはnearest補間
        "hm1": "image",  # ヒートマップはbilinear補間でOK
        "hm2": "image",
        "hm3": "image",
        "hm4": "image",
    }

    # flip適用の有無を記録する場合はReplayCompose
    if cfg_aug.get("horizontal_flip", False):
        return A.ReplayCompose(ts, additional_targets=additional_targets)
    return A.Compose(ts, additional_targets=additional_targets)


# -------------------------
# データセット（PNG）: 厳格な有効性フィルタリング + オンライン拡張
# -------------------------
class PngLineDataset(Dataset):
    """
    dataset/
      sampleXX/
        C3/
          images/slice_000.png
          masks/slice_000.png
          lines.json
    """

    def __init__(
        self,
        root_dir: Path,
        sample_names,
        group="C3_C7",
        image_size=224,
        sigma=4.0,
        transform=None,
        cfg_aug=None,
    ):
        self.root_dir = Path(root_dir)
        self.sample_names = list(sample_names)
        self.group = group
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        self.vertebra_names = vertebra_names_from_group(group)

        self.transform = transform
        self.cfg_aug = cfg_aug or {}

        self.items = []
        self._build_index()
        print(f"[INFO] PngLineDataset: {len(self.items)} slices")

    def _build_index(self):
        """有効なスライスをインデックスに追加"""
        for s in self.sample_names:
            sd = self.root_dir / s
            if not sd.exists():
                continue

            for v in self.vertebra_names:
                vd = sd / v
                lj = vd / "lines.json"
                if not vd.exists() or not lj.exists():
                    continue

                try:
                    lines_data = json.loads(lj.read_text())
                except Exception:
                    continue

                for slice_idx_str, lines in lines_data.items():
                    ok = True
                    for k in ["line_1", "line_2", "line_3", "line_4"]:
                        if k not in lines or lines[k] is None or len(lines[k]) < 2:
                            ok = False
                            break
                    if not ok:
                        continue

                    slice_idx = int(slice_idx_str)
                    ip = vd / "images" / f"slice_{slice_idx:03d}.png"
                    mp = vd / "masks" / f"slice_{slice_idx:03d}.png"
                    if not ip.exists() or not mp.exists():
                        continue

                    self.items.append(
                        {
                            "sample": s,
                            "vertebra": v,
                            "slice_idx": slice_idx,
                            "img_path": ip,
                            "mask_path": mp,
                            "lines": lines,
                        }
                    )

    def _heatmap_from_polyline(self, pts_xy):
        """折れ線からガウシアンヒートマップを生成（距離変換ベース）"""
        H = W = self.image_size
        hm = np.zeros((H, W), np.float32)
        if pts_xy is None or len(pts_xy) < 2:
            return hm

        pts = np.array(pts_xy, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        pts_i32 = pts.astype(np.int32).reshape(-1, 1, 2)

        mask = np.zeros((H, W), np.uint8)
        cv2.polylines(mask, [pts_i32], isClosed=False, color=1, thickness=1)

        inv = (1 - mask).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        s2 = max(1e-6, self.sigma**2)
        hm = np.exp(-(dist**2) / (2.0 * s2)).astype(np.float32)
        return hm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]

        ct = np.array(Image.open(it["img_path"]).convert("L"), np.float32) / 255.0
        mk = np.array(Image.open(it["mask_path"]).convert("L"), np.float32) / 255.0

        if ct.shape != (self.image_size, self.image_size):
            ct = cv2.resize(
                ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
        if mk.shape != (self.image_size, self.image_size):
            mk = cv2.resize(
                mk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            )

        hms = []
        for k in ["line_1", "line_2", "line_3", "line_4"]:
            hms.append(self._heatmap_from_polyline(it["lines"][k]))
        hms = np.stack(hms, 0).astype(np.float32)  # (4,H,W)

        # GT直線パラメータ (φ, ρ) を抽出
        gt_line_params = []
        for k in ["line_1", "line_2", "line_3", "line_4"]:
            polyline = it["lines"][k]
            phi, rho = line_losses.extract_gt_line_params(polyline, self.image_size)
            gt_line_params.append([phi, rho])
        gt_line_params = np.array(gt_line_params, dtype=np.float32)  # (4, 2)

        # オンライン拡張（訓練時のみ）
        if self.transform is not None:
            out = self.transform(
                image=ct,
                mask=mk,
                hm1=hms[0],
                hm2=hms[1],
                hm3=hms[2],
                hm4=hms[3],
            )
            ct = out["image"]
            mk = out["mask"]
            hms = np.stack(
                [out["hm1"], out["hm2"], out["hm3"], out["hm4"]], axis=0
            ).astype(np.float32)

            # flip時にch入れ替えが必要な定義なら configで指定（例: [2,1,4,3]）
            if isinstance(self.transform, A.ReplayCompose) and self.cfg_aug.get(
                "horizontal_flip", False
            ):
                did_flip = False
                for tr in out["replay"]["transforms"]:
                    if tr.get("__class_fullname__", "").endswith(
                        "HorizontalFlip"
                    ) and tr.get("applied", False):
                        did_flip = True
                        break
                if did_flip:
                    swap_map = self.cfg_aug.get("hflip_channel_swap", None)
                    if swap_map is not None:
                        idx = [int(v) - 1 for v in swap_map]
                        hms = hms[idx]

        ct = np.clip(ct, 0.0, 1.0).astype(np.float32)
        mk = np.clip(mk, 0.0, 1.0).astype(np.float32)
        hms = np.clip(hms, 0.0, 1.0).astype(np.float32)

        x = np.stack([ct, mk], 0).astype(np.float32)  # (2,H,W)

        return {
            "image": torch.from_numpy(x),
            "heatmaps": torch.from_numpy(hms),
            "line_params_gt": torch.from_numpy(gt_line_params),  # (4, 2)
            "sample": it["sample"],
            "vertebra": it["vertebra"],
            "slice_idx": it["slice_idx"],
        }
