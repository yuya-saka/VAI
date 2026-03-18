# Unet/train_heat.py

import json
import math
import random
import tempfile
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from . import line_losses, line_metrics
from .line_detection import predict_lines_and_eval_test
from PIL import Image
from torch.utils.data import DataLoader, Dataset

tempfile.tempdir = "/tmp"


# -------------------------
# 再現性の確保
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoaderワーカーのシード設定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------
# 設定ファイル読み込み / データ分割
# -------------------------
def load_config(cfg_path="config/config.yaml"):
    p = Path(cfg_path)
    if not p.exists():
        p = Path("Unet") / cfg_path
    if not p.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    print(f"[INFO] loaded config: {p.resolve()}")
    return cfg


def resolve_dataset_root(cfg_root_dir: str):
    if cfg_root_dir:
        p = Path(cfg_root_dir)
        if p.exists():
            return p.resolve()
    here = Path(__file__).resolve().parent
    cand = (here.parent / "dataset").resolve()  # Unet/../dataset
    if cand.exists():
        return cand
    cand = (here / "dataset").resolve()
    return cand


def vertebra_names_from_group(group: str):
    if group == "C1":
        return ["C1"]
    if group == "C2":
        return ["C2"]
    if group == "C3_C7":
        return ["C3", "C4", "C5", "C6", "C7"]
    if group == "ALL":
        return ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    raise ValueError(f"Unknown group: {group}")


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


def kfold_split_samples(sample_names, n_folds=5, test_fold=0, seed=42):
    sample_names = np.array(sorted(sample_names))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(sample_names))
    rng.shuffle(idx)

    folds = np.array_split(idx, n_folds)
    test_idx = folds[test_fold]
    val_fold = (test_fold + 1) % n_folds
    val_idx = folds[val_fold]
    train_idx = np.setdiff1d(idx, np.concatenate([test_idx, val_idx]))

    train = sample_names[train_idx].tolist()
    val = sample_names[val_idx].tolist()
    test = sample_names[test_idx].tolist()
    return train, val, test


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
# 可視化
# -------------------------
def save_heatmap_overlay(ct01, hm4, save_path, alpha=0.55, vmax=1.0):
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    hm = np.max(hm4, axis=0)
    hm = np.clip(hm / max(vmax, 1e-6), 0, 1)
    hm_u8 = (hm * 255).astype(np.uint8)

    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)


def save_heatmap_grid(ct01, hm4, save_path, alpha=0.55):
    H, W = ct01.shape
    ct_u8 = (np.clip(ct01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    tiles = []
    for c in range(4):
        hm = np.clip(hm4[c], 0, 1)
        hm_u8 = (hm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)
        cv2.putText(
            out,
            f"CH{c + 1}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(out)

    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), grid)


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

        # NEW: Extract GT line params (φ, ρ)
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
            "line_params_gt": torch.from_numpy(gt_line_params),  # NEW: (4, 2)
            "sample": it["sample"],
            "vertebra": it["vertebra"],
            "slice_idx": it["slice_idx"],
        }


# -------------------------
# モデル定義
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=4, feats=(16, 32, 64, 128), dropout=0.0):
        super().__init__()
        f1, f2, f3, f4 = feats
        self.d1 = DoubleConv(in_ch, f1, dropout=dropout)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(f1, f2, dropout=dropout)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(f2, f3, dropout=dropout)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(f3, f4, dropout=dropout)

        self.u3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.up3 = DoubleConv(f3 + f3, f3, dropout=dropout)
        self.u2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.up2 = DoubleConv(f2 + f2, f2, dropout=dropout)
        self.u1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.up1 = DoubleConv(f1 + f1, f1, dropout=dropout)

        self.out = nn.Conv2d(f1, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))

        y = self.u3(x4)
        y = self.up3(torch.cat([y, x3], 1))
        y = self.u2(y)
        y = self.up2(torch.cat([y, x2], 1))
        y = self.u1(y)
        y = self.up1(torch.cat([y, x1], 1))
        return self.out(y)


# -------------------------
# 評価指標
# -------------------------
def peak_dist(pred, gt):
    """Distance between heatmap peaks (for debugging heatmap quality)"""
    gy, gx = np.unravel_index(np.argmax(gt), gt.shape)
    py, px = np.unravel_index(np.argmax(pred), pred.shape)
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


@torch.no_grad()
def evaluate(model, loader, device, image_size=224):
    model.eval()
    mse_sum = 0.0
    n = 0

    peak_dists = []

    # Line metrics
    angle_errors = []
    rho_errors = []

    # Per-vertebra statistics
    vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    per_vertebra = {v: {"peak_dists": []} for v in vertebrae}

    for batch in loader:
        x = batch["image"].to(device).float()
        gt = batch["heatmaps"].to(device).float()
        gt_params = batch.get("line_params_gt")  # NEW: Get GT line params

        pred = torch.sigmoid(model(x))

        mse_sum += F.mse_loss(pred, gt, reduction="mean").item()
        n += 1

        pr = pred.cpu().numpy()
        g = gt.cpu().numpy()

        B = pr.shape[0]
        for i in range(B):
            v_name = batch["vertebra"][i]  # 椎体名を取得
            for c in range(4):
                pd = peak_dist(pr[i, c], g[i, c])
                peak_dists.append(pd)

                # Per-vertebra statistics
                if v_name in per_vertebra:
                    per_vertebra[v_name]["peak_dists"].append(pd)

        # NEW: Compute line metrics
        if gt_params is not None:
            gt_params = gt_params.to(device).float()
            pred_params, confidence = line_losses.extract_pred_line_params_batch(
                pred, image_size
            )

            gt_valid = ~torch.isnan(gt_params).any(dim=-1)
            pred_valid = ~torch.isnan(pred_params).any(dim=-1)
            valid_mask = gt_valid & pred_valid

            angle_err = line_metrics.compute_angle_error(
                pred_params, gt_params, valid_mask
            )
            rho_err = line_metrics.compute_rho_error(
                pred_params, gt_params, image_size, valid_mask
            )

            angle_errors.append(angle_err)
            rho_errors.append(rho_err)

    # Per-vertebra statistics
    per_vert_stats = {}
    for v, vals in per_vertebra.items():
        if len(vals["peak_dists"]) > 0:
            per_vert_stats[v] = {
                "peak_dist_mean": float(np.nanmean(vals["peak_dists"])),
                "n_samples": len(vals["peak_dists"]) // 4,  # 4 channels
            }

    metrics = {
        "val_loss_mse": mse_sum / max(1, n),
        "peak_dist_mean": float(np.nanmean(peak_dists)),
        "per_vertebra": per_vert_stats,
    }

    # Add line metrics if available
    if angle_errors:
        metrics["angle_error_deg"] = float(np.nanmean(angle_errors))
        metrics["rho_error_px"] = float(np.nanmean(rho_errors))

    return metrics


# -------------------------
# 訓練（ヘルパー関数群）
# -------------------------
def prepare_datasets_and_splits(cfg):
    """
    データセットの準備とK-Fold分割を実行

    Args:
        cfg: 設定辞書

    Returns:
        tuple: (train_samples, val_samples, test_samples, root_dir, group, image_size, sigma, seed)
    """
    data_cfg = cfg.get("data", {})
    root_dir = resolve_dataset_root(data_cfg.get("root_dir", ""))
    group = data_cfg.get("group", "C3_C7")
    image_size = int(data_cfg.get("image_size", 224))
    sigma = float(data_cfg.get("sigma", 4.0))

    # サンプルリストの作成と有効性フィルタリング
    sample_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sample")]
    )
    all_samples = [d.name for d in sample_dirs]

    valid_samples = []
    for d in sample_dirs:
        if _is_sample_valid_png(d, group):
            valid_samples.append(d.name)

    if len(valid_samples) == 0:
        raise ValueError(f"No valid samples found under {root_dir} (group={group})")

    print(f"[INFO] all_samples={len(all_samples)}  valid_samples={len(valid_samples)}")

    # K-Fold分割
    n_folds = int(data_cfg.get("n_folds", 5))
    test_fold = int(data_cfg.get("test_fold", 0))
    seed = int(data_cfg.get("random_seed", 42))

    train_s, val_s, test_s = kfold_split_samples(
        valid_samples, n_folds=n_folds, test_fold=test_fold, seed=seed
    )
    print(
        f"[SPLIT] n_folds={n_folds} test_fold={test_fold} val_fold={(test_fold + 1) % n_folds}"
    )
    print(f"[SPLIT] train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    return train_s, val_s, test_s, root_dir, group, image_size, sigma, seed


def create_data_loaders(
    train_samples,
    val_samples,
    test_samples,
    root_dir,
    group,
    image_size,
    sigma,
    seed,
    cfg,
):
    """
    訓練/検証/テスト用のDataLoaderを作成

    Args:
        train_samples: 訓練サンプル名リスト
        val_samples: 検証サンプル名リスト
        test_samples: テストサンプル名リスト
        root_dir: データセットルートディレクトリ
        group: 椎体グループ名
        image_size: 画像サイズ
        sigma: ヒートマップのsigma値
        seed: 乱数シード
        cfg: 設定辞書

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    aug_cfg = cfg.get("augmentation", {})
    tr_cfg = cfg.get("training", {})

    # データ変換（訓練時のみ）
    train_tf = get_transforms("train", aug_cfg)

    train_ds = PngLineDataset(
        root_dir,
        train_samples,
        group=group,
        image_size=image_size,
        sigma=sigma,
        transform=train_tf,
        cfg_aug=aug_cfg,
    )
    val_ds = PngLineDataset(
        root_dir, val_samples, group=group, image_size=image_size, sigma=sigma
    )
    test_ds = PngLineDataset(
        root_dir, test_samples, group=group, image_size=image_size, sigma=sigma
    )

    bs = int(tr_cfg.get("batch_size", 8))
    nw = int(tr_cfg.get("num_workers", 4))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader


def create_model_optimizer_scheduler(cfg, device):
    """
    モデル、最適化器、学習率スケジューラーを作成

    Args:
        cfg: 設定辞書
        device: PyTorchデバイス

    Returns:
        tuple: (model, optimizer, scheduler)
    """
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})

    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))

    model = TinyUNet(in_ch=in_ch, out_ch=out_ch, feats=feats, dropout=dropout).to(
        device
    )

    lr = float(tr_cfg.get("learning_rate", 1e-4))
    wd = float(tr_cfg.get("weight_decay", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ReduceLROnPlateau（val_lossを基準）
    lr_pat = int(tr_cfg.get("lr_patience", 5))
    lr_fac = float(tr_cfg.get("lr_factor", 0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=lr_pat, factor=lr_fac
    )

    return model, opt, scheduler


def run_training_loop(
    model, opt, scheduler, train_loader, val_loader, device, cfg, best_path
):
    """
    訓練ループを実行（早期停止機能付き）

    Args:
        model: PyTorchモデル
        opt: オプティマイザー
        scheduler: 学習率スケジューラー
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader
        device: PyTorchデバイス
        cfg: 設定辞書
        best_path: ベストモデル保存パス

    Returns:
        None (ベストモデルをbest_pathに保存)
    """
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    loss_cfg = cfg.get("loss", {})  # NEW

    epochs = int(tr_cfg.get("epochs", 20))
    es_pat = int(tr_cfg.get("early_stopping_patience", 20))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    image_size = int(cfg.get("data", {}).get("image_size", 224))  # NEW

    # NEW: Line loss configuration
    use_angle_loss = loss_cfg.get("use_angle_loss", False)
    use_rho_loss = loss_cfg.get("use_rho_loss", False)
    lambda_theta = float(loss_cfg.get("lambda_theta", 0.1))
    lambda_rho = float(loss_cfg.get("lambda_rho", 0.05))
    warmup_epochs = int(loss_cfg.get("warmup_epochs", 10))
    warmup_mode = loss_cfg.get("warmup_mode", "linear")

    best_val = float("inf")
    no_improve = 0

    # 評価指標の計算頻度（オプション）
    mfreq = int(eval_cfg.get("metrics_frequency", 1))  # 1=毎エポック、0=毎エポック
    if mfreq <= 0:
        mfreq = 1

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        steps = 0

        # NEW: Compute warmup weight
        warmup_weight = line_losses.get_warmup_weight(ep, warmup_epochs, warmup_mode)

        for batch in train_loader:
            x = batch["image"].to(device).float()
            gt = batch["heatmaps"].to(device).float()
            gt_params = batch.get("line_params_gt")  # NEW

            pred = torch.sigmoid(model(x))

            # MSE loss
            loss_mse = F.mse_loss(pred, gt, reduction="mean")

            # NEW: Line losses
            if (use_angle_loss or use_rho_loss) and gt_params is not None:
                gt_params = gt_params.to(device).float()
                line_loss_dict = line_losses.compute_line_loss(
                    pred,
                    gt_params,
                    image_size,
                    lambda_theta,
                    lambda_rho,
                    use_angle_loss,
                    use_rho_loss,
                )
                # Combined loss with inverse weighting for MSE
                # MSE gradually decreases while line loss gradually increases
                loss = (1 - 0.5 * warmup_weight) * loss_mse + warmup_weight * line_loss_dict["total"]
            else:
                # MSE only
                loss = loss_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_sum += loss.item()
            steps += 1

        train_loss = loss_sum / max(1, steps)

        # 評価
        if ep % mfreq == 0:
            val_metrics = evaluate(model, val_loader, device, image_size)
        else:
            # val_mseは最低限earlystop/schedulerに必要なので毎エポック計算
            val_metrics = evaluate(model, val_loader, device, image_size)

        # validation lossに基づいてスケジューラを更新
        scheduler.step(val_metrics["val_loss_mse"])

        cur_lr = opt.param_groups[0]["lr"]
        log_str = (
            f"[EPOCH {ep:03d}/{epochs}] lr={cur_lr:.2e} "
            f"train_mse={train_loss:.6f}  "
            f"val_mse={val_metrics['val_loss_mse']:.6f}  "
            f"peak={val_metrics['peak_dist_mean']:.2f}px  "
        )

        # Add line metrics if available
        if "angle_error_deg" in val_metrics:
            log_str += (
                f"angle={val_metrics['angle_error_deg']:.2f}°  "
                f"rho={val_metrics['rho_error_px']:.2f}px  "
            )

        log_str += f"time={time.time() - t0:.1f}s"
        print(log_str)

        # val_mseによる早期停止
        if val_metrics["val_loss_mse"] < best_val - 1e-8:
            best_val = val_metrics["val_loss_mse"]
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "val": val_metrics}, best_path
            )
            print(f"  [SAVE] best -> {best_path} (val_mse={best_val:.6f})")
        else:
            no_improve += 1
            if no_improve >= es_pat:
                print(
                    f"[EARLY STOP] no improvement for {es_pat} epochs. best_val={best_val:.6f}"
                )
                break


# -------------------------
# 訓練（メイン関数）
# -------------------------
def train_one_fold(cfg):
    """
    1つのfoldに対する訓練を実行

    Args:
        cfg: 設定辞書

    Returns:
        dict: テスト結果の辞書
    """
    # 設定の取得
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    test_fold = int(data_cfg.get("test_fold", 0))

    # データセット準備と分割
    train_s, val_s, test_s, root_dir, group, image_size, sigma, seed = (
        prepare_datasets_and_splits(cfg)
    )

    # データローダー作成
    train_loader, val_loader, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, sigma, seed, cfg
    )

    # デバイス設定
    gpu_id = int(tr_cfg.get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # モデル、最適化器、スケジューラー作成
    model, opt, scheduler = create_model_optimizer_scheduler(cfg, device)

    # チェックポイントディレクトリ作成（Unetディレクトリを基準に）
    script_dir = Path(__file__).resolve().parent.parent  # Unet/ directory
    ckpt_dir = script_dir / tr_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_fold{test_fold}.pt"

    # 訓練ループ実行
    run_training_loop(
        model, opt, scheduler, train_loader, val_loader, device, cfg, best_path
    )

    # ベストモデルを読み込んでテスト
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print(f"[WARNING] No best checkpoint saved (no improvement during training). Using current model state.")
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"[TEST] fold={test_fold}  "
        f"mse={test_metrics['val_loss_mse']:.6f}  "
        f"peak={test_metrics['peak_dist_mean']:.2f}px"
    )

    # テストデータに対する直線検出（別モジュール）（スクリプトのディレクトリを基準に）
    out_dir = (
        script_dir
        / cfg.get("evaluation", {}).get("visualization_dir", "vis")
        / f"fold{test_fold}"
        / "test_lines"
    )
    line_summary = predict_lines_and_eval_test(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        device=device,
        dataset_root=root_dir,
        out_dir=out_dir,
    )

    print("\n" + "=" * 60)
    print("[LINE GEOMETRY EVALUATION]")
    print("=" * 60)
    print(f"  Perpendicular Distance: {line_summary['perpendicular_dist_px_mean']:.2f} px  ⭐")
    print(f"  Angle Error:           {line_summary['angle_error_deg_mean']:.2f} deg ⭐")
    print(f"  Rho Error:             {line_summary['rho_error_px_mean']:.2f} px  ⭐")
    print("\n[Per-Channel Breakdown]")
    for k, v in line_summary["per_channel"].items():
        print(
            f"  {k}: perp={v['perpendicular_dist_px_mean']:.2f}px  "
            f"angle={v['angle_error_deg_mean']:.2f}deg  "
            f"rho={v['rho_error_px_mean']:.2f}px  (n={v['n']})"
        )
    print(f"\n[Output] {line_summary['out_dir']}")
    print("=" * 60)

    # サンプル画像を保存
    @torch.no_grad()
    def save_examples(model, loader, device, out_dir: Path, n_save=12, tag="VAL"):
        model.eval()
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for batch in loader:
            x = batch["image"].to(device).float()
            gt = batch["heatmaps"].to(device).float()
            pred = torch.sigmoid(model(x))

            x_np = x.cpu().numpy()
            gt_np = gt.cpu().numpy()
            pr_np = pred.cpu().numpy()

            B = x_np.shape[0]
            for i in range(B):
                ct01 = x_np[i, 0]  # CT
                name = f"{batch['sample'][i]}_{batch['vertebra'][i]}_slice{int(batch['slice_idx'][i]):03d}"

                save_heatmap_grid(ct01, gt_np[i], out_dir / f"{tag}_{name}_GT_grid.png")
                save_heatmap_grid(
                    ct01, pr_np[i], out_dir / f"{tag}_{name}_PRED_grid.png"
                )
                save_heatmap_overlay(
                    ct01, gt_np[i], out_dir / f"{tag}_{name}_GT_merged.png"
                )
                save_heatmap_overlay(
                    ct01, pr_np[i], out_dir / f"{tag}_{name}_PRED_merged.png"
                )

                saved += 1
                if saved >= n_save:
                    return

    # 可視化ディレクトリ（スクリプトのディレクトリを基準に）
    vis_root = script_dir / eval_cfg.get("visualization_dir", "vis_2")
    print("[INFO] saving example overlays ...")
    save_examples(
        model,
        val_loader,
        device,
        vis_root / f"fold{test_fold}" / "val",
        n_save=16,
        tag="VAL",
    )
    save_examples(
        model,
        test_loader,
        device,
        vis_root / f"fold{test_fold}" / "test",
        n_save=16,
        tag="TEST",
    )
    print(f"[INFO] saved to {vis_root}/")

    # Return results for train.py
    return {
        "test_mse": test_metrics["val_loss_mse"],
        "test_peak_dist_mean": test_metrics["peak_dist_mean"],
        "line_perpendicular_dist_px_mean": line_summary["perpendicular_dist_px_mean"],
        "line_angle_error_deg_mean": line_summary["angle_error_deg_mean"],
        "line_rho_error_px_mean": line_summary["rho_error_px_mean"],
        "per_vertebra": test_metrics.get("per_vertebra", {}),
    }


def main():
    cfg = load_config("config/config.yaml")
    seed = int(cfg.get("data", {}).get("random_seed", 42))
    set_seed(seed)

    if not cfg.get("data", {}).get("use_png", True):
        raise RuntimeError("This script is for PNG dataset (use_png: true) only.")

    train_one_fold(cfg)


if __name__ == "__main__":
    main()
