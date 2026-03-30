"""設定読込・乱数シード・データ分割・DataLoader作成"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    PngLineDataset,
    _is_sample_valid_png,
    get_transforms,
)
from .model import TinyUNet


# -------------------------
# 再現性の確保
# -------------------------
def set_seed(seed=42):
    """乱数シードを設定して再現性を確保"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoaderワーカーのシード設定"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------
# 設定ファイル読み込み
# -------------------------
def load_config(cfg_path="config/config.yaml"):
    """YAML設定ファイルを読み込む"""
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
    """データセットルートディレクトリを解決"""
    if cfg_root_dir:
        p = Path(cfg_root_dir)
        if p.exists():
            return p.resolve()
    here = Path(__file__).resolve().parent
    cand = (here.parent.parent / "dataset").resolve()  # Unet/../dataset
    if cand.exists():
        return cand
    cand = (here.parent / "dataset").resolve()
    return cand


# -------------------------
# K-Fold分割
# -------------------------
def kfold_split_samples(sample_names, n_folds=5, test_fold=0, seed=42):
    """サンプルをK-Foldで分割（train/val/test）"""
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
# データセット準備
# -------------------------
def prepare_datasets_and_splits(cfg):
    """
    データセットの準備とK-Fold分割を実行

    戻り値:
        tuple: (train_samples, val_samples, test_samples,
                root_dir, group, image_size, sigma, seed)
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


# -------------------------
# DataLoader作成
# -------------------------
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

    戻り値:
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


# -------------------------
# モデル・オプティマイザ・スケジューラ作成
# -------------------------
def create_model_optimizer_scheduler(cfg, device):
    """
    モデル、最適化器、学習率スケジューラーを作成

    戻り値:
        tuple: (model, optimizer, scheduler)
    """
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})

    in_ch = int(model_cfg.get("in_channels", 2))
    out_ch = int(model_cfg.get("out_channels", 4))
    feats = tuple(model_cfg.get("features", [16, 32, 64, 128]))
    dropout = float(model_cfg.get("dropout", 0.0))
    use_vertebra_conditioning = bool(model_cfg.get("use_vertebra_conditioning", False))
    num_vertebra = int(model_cfg.get("num_vertebra", 7)) if use_vertebra_conditioning else 0

    model = TinyUNet(
        in_ch=in_ch,
        out_ch=out_ch,
        feats=feats,
        dropout=dropout,
        num_vertebra=num_vertebra,
    ).to(device)

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
