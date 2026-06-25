"""
設定読込・乱数シード・データ収集・CV分割・DataLoader作成。
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import RSNAFractureDataset, get_train_transforms

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


# -------------------------
# 再現性の確保
# -------------------------
def set_seed(seed: int = 42) -> None:
    """乱数シードを設定して再現性を確保する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoaderワーカーのシード設定。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


# -------------------------
# 設定ファイル読み込み
# -------------------------
def load_config(cfg_path: str | Path | None = None) -> dict:
    """YAML設定ファイルを読み込む。"""
    if cfg_path is None:
        cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[INFO] config 読み込み完了: {p.resolve()}")
    return cfg


def save_effective_config(cfg: dict, output_dir: Path) -> Path:
    """
    CLI上書きを反映した実効configを output_dir/config.yaml に保存する。

    学習のたびに呼び出すことで、各 run の設定を永続化・再現可能にする。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] 実効 config を保存しました: {output_path}")
    return output_path


def _resolve_root() -> Path:
    """プロジェクトルートディレクトリを解決する。"""
    # src/data_utils.py → src/ → stage1/ → train_models/ → VAI/
    return Path(__file__).resolve().parent.parent.parent.parent


# -------------------------
# データ収集
# -------------------------
def collect_items(dataset_dir: Path, csv_path: Path) -> list[dict]:
    """
    fracture_dataset/ と train.csv からアイテムリストを構築する。

    各アイテムは1椎体を表す:
    {
        "study_uid": str,
        "vertebra": str (C1-C7),
        "label": int (0/1),
        "ct_path": Path,
        "mask_path": Path,
    }

    train.csv に存在し、かつ fracture_dataset にファイルが存在する
    study × vertebra のみを収集する。

    Args:
        dataset_dir: fracture_dataset/ ディレクトリのパス
        csv_path: train.csv のパス

    Returns:
        アイテムdictのリスト（最大 2012 studies × 7 vertebrae = 14,084件）
    """
    df = pd.read_csv(csv_path)
    items = []
    missing_studies = 0

    for _, row in df.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        study_dir = dataset_dir / study_uid
        patient_label = int(row["patient_overall"]) if "patient_overall" in df.columns else int(
            max(row[vertebra] for vertebra in VERTEBRAE)
        )

        if not study_dir.exists():
            missing_studies += 1
            continue

        for vertebra in VERTEBRAE:
            ct_path = study_dir / vertebra / "ct.npy"
            mask_path = study_dir / vertebra / "vertebra_mask.npy"

            if not ct_path.exists() or not mask_path.exists():
                continue

            label = int(row[vertebra])
            items.append(
                {
                    "study_uid": study_uid,
                    "vertebra": vertebra,
                    "label": label,
                    "patient_label": patient_label,
                    "ct_path": ct_path,
                    "mask_path": mask_path,
                }
            )

    n_pos = sum(it["label"] for it in items)
    print(
        f"[INFO] アイテム収集完了: {len(items)} 件 "
        f"(陽性={n_pos}, 陰性={len(items) - n_pos})"
    )
    if missing_studies > 0:
        print(f"[WARNING] fracture_dataset に存在しない study: {missing_studies} 件")

    return items


# -------------------------
# Held-out Test 分離
# -------------------------
def split_test_holdout(
    items: list[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Study単位で test_size 割合を held-out test set として分離する。

    この test set は学習・early stopping・モデル選択に一切使用しない。
    stratify: study-level fracture label（患者レベルの骨折有無）

    Args:
        items: collect_items() の出力
        test_size: テストセットの割合（デフォルト 0.2 = 20%）
        seed: 乱数シード（固定して再現性を確保）

    Returns:
        (train_val_items, test_items)
    """
    study_label: dict[str, int] = {}
    for item in items:
        uid = item["study_uid"]
        if uid not in study_label:
            study_label[uid] = 0
        study_label[uid] = max(study_label[uid], item["label"])

    study_uids = np.array(list(study_label.keys()))
    labels = np.array([study_label[uid] for uid in study_uids])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_val_idx, test_idx = next(splitter.split(study_uids, labels))

    train_val_uids = set(study_uids[train_val_idx])
    test_uids = set(study_uids[test_idx])

    train_val_items = [it for it in items if it["study_uid"] in train_val_uids]
    test_items = [it for it in items if it["study_uid"] in test_uids]

    n_tv_pos = sum(it["label"] for it in train_val_items)
    n_test_pos = sum(it["label"] for it in test_items)
    print(
        f"[HOLDOUT] train_val={len(train_val_items)} (陽性={n_tv_pos})  "
        f"test={len(test_items)} (陽性={n_test_pos})"
    )

    return train_val_items, test_items


# -------------------------
# K-Fold CV 分割
# -------------------------
def split_items_cv(
    items: list[dict],
    n_splits: int = 5,
    val_fold: int = 0,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Study単位のStratifiedGroupKFoldでtrain/valに分割する。

    stratify: study-level fracture label (patient_overall)
    group: study_uid（同一 study が train/val に分かれないよう保証）

    Args:
        items: collect_items() の出力
        n_splits: fold数
        val_fold: validation に使う fold インデックス
        seed: 乱数シード

    Returns:
        (train_items, val_items)
    """
    # study単位の統計を収集（1つでも骨折椎体があれば陽性）
    study_label: dict[str, int] = {}
    for item in items:
        uid = item["study_uid"]
        if uid not in study_label:
            study_label[uid] = 0
        study_label[uid] = max(study_label[uid], item["label"])

    study_uids = list(study_label.keys())
    labels = np.array([study_label[uid] for uid in study_uids])

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(study_uids, labels, groups=study_uids))
    train_idx, val_idx = splits[val_fold]

    train_uids = {study_uids[i] for i in train_idx}
    val_uids = {study_uids[i] for i in val_idx}

    train_items = [it for it in items if it["study_uid"] in train_uids]
    val_items = [it for it in items if it["study_uid"] in val_uids]

    n_train_pos = sum(it["label"] for it in train_items)
    n_val_pos = sum(it["label"] for it in val_items)
    print(
        f"[SPLIT] fold={val_fold} "
        f"train={len(train_items)} (陽性={n_train_pos}) "
        f"val={len(val_items)} (陽性={n_val_pos})"
    )

    return train_items, val_items


# -------------------------
# DataLoader 作成
# -------------------------
def create_data_loaders(
    train_items: list[dict],
    val_items: list[dict],
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """
    訓練/検証用DataLoaderを作成する。

    Args:
        train_items: 訓練用アイテムリスト
        val_items: 検証用アイテムリスト
        cfg: config dict

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})

    seed = int(data_cfg.get("random_seed", 42))
    p_rand_order = float(tr_cfg.get("p_rand_order", 0.2))
    include_patient_label = bool(model_cfg.get("use_patient_head", False))
    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 4))
    persistent_workers = bool(tr_cfg.get("persistent_workers", True))
    prefetch_factor = int(tr_cfg.get("prefetch_factor", 4))

    train_transform = get_train_transforms(aug_cfg)

    train_ds = RSNAFractureDataset(
        train_items,
        mode="train",
        transform=train_transform,
        p_rand_order=p_rand_order,
        include_patient_label=include_patient_label,
    )
    val_ds = RSNAFractureDataset(
        val_items,
        mode="valid",
        transform=None,
        p_rand_order=0.0,
        include_patient_label=include_patient_label,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    # DDP時はDistributedSamplerを使用（shuffle=FalseはSampler側で制御）
    if dist.is_initialized():
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=(train_sampler is None),
        **_worker_loader_options(
            num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        **_worker_loader_options(
            num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )

    return train_loader, val_loader


def create_eval_data_loader(items: list[dict], cfg: dict) -> DataLoader:
    """Create a shared optimized DataLoader for validation-style inference."""
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    include_patient_label = bool(model_cfg.get("use_patient_head", False))
    batch_size = int(tr_cfg.get("batch_size", 8))
    num_workers = int(tr_cfg.get("num_workers", 4))
    persistent_workers = bool(tr_cfg.get("persistent_workers", True))
    prefetch_factor = int(tr_cfg.get("prefetch_factor", 4))

    dataset = RSNAFractureDataset(
        items,
        mode="valid",
        transform=None,
        p_rand_order=0.0,
        include_patient_label=include_patient_label,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        **_worker_loader_options(
            num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )


def _worker_loader_options(
    num_workers: int,
    *,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, bool | int]:
    """Return DataLoader options that are valid only with worker processes."""
    if num_workers == 0:
        return {}
    return {
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


# -------------------------
# モデル・オプティマイザ・スケジューラ作成
# -------------------------
def create_model_optimizer_scheduler(cfg: dict, device: torch.device) -> tuple:
    """
    モデル、最適化器、学習率スケジューラーを作成する。

    CosineAnnealingLR を使用（RSNA 1位解法準拠）。

    Returns:
        (model, optimizer, scheduler)
    """
    from .model import TimmModel

    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    model = TimmModel(
        backbone=str(model_cfg.get("backbone", "tf_efficientnetv2_s")),
        in_chans=int(data_cfg.get("in_channels", 6)),
        n_slices=int(data_cfg.get("n_slices", 15)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        drop_rate_last=float(model_cfg.get("drop_rate_last", 0.3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(model_cfg.get("lstm_layers", 2)),
        out_dim=int(model_cfg.get("out_dim", 1)),
        use_patient_head=bool(model_cfg.get("use_patient_head", False)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    lr = float(tr_cfg.get("learning_rate", 2.3e-4))
    wd = float(tr_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(tr_cfg.get("epochs", 75))
    eta_min = float(tr_cfg.get("eta_min", 2.3e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )

    return model, optimizer, scheduler
