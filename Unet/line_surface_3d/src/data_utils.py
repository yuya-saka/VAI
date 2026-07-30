"""設定、分割、manifest、DataLoaderの構築。"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    SlabLineDataset,
    SlabRecord,
    build_slab_records,
    get_transforms,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """YAML設定を読み、必須値を検証する。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """スラブ構成と学習設定の整合性を検証する。"""
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    slab_size = int(data_config.get("slab_size", 15))
    min_labeled = int(data_config.get("min_labeled_slices", 3))
    if slab_size < 3 or slab_size % 2 == 0:
        raise ValueError("slab_sizeは3以上の奇数が必要です")
    if min_labeled < 1 or min_labeled > slab_size:
        raise ValueError("min_labeled_slicesがslab_sizeの範囲外です")
    for key in ("annotation_root", "dense_root"):
        if not data_config.get(key):
            raise ValueError(f"data.{key}は必須です")
    if int(data_config.get("train_stride", 1)) < 1:
        raise ValueError("train_strideは1以上が必要です")
    if int(data_config.get("inference_stride", 1)) < 1:
        raise ValueError("inference_strideは1以上が必要です")
    if int(training_config.get("num_workers", 4)) < 0:
        raise ValueError("num_workersは0以上が必要です")
    if int(training_config.get("prefetch_factor", 1)) < 1:
        raise ValueError("prefetch_factorは1以上が必要です")


def set_seed(seed: int) -> None:
    """Python、NumPy、PyTorchの乱数を固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader workerの乱数を固定する。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def kfold_split_samples(
    sample_names: list[str],
    n_folds: int,
    test_fold: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """sample単位でtrain/validation/testへ分割する。"""
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_foldが範囲外です: {test_fold}")
    names = np.asarray(sorted(sample_names))
    generator = np.random.RandomState(seed)
    indices = np.arange(len(names))
    generator.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    test_indices = folds[test_fold]
    validation_indices = folds[(test_fold + 1) % n_folds]
    training_indices = np.setdiff1d(
        indices,
        np.concatenate([test_indices, validation_indices]),
    )
    return (
        names[training_indices].tolist(),
        names[validation_indices].tolist(),
        names[test_indices].tolist(),
    )


def discover_samples(
    dense_root: Path,
    annotation_root: Path,
) -> list[str]:
    """密画像と手動annotationの両方に存在するsampleを列挙する。"""
    dense_samples = {path.name for path in dense_root.glob("sample*") if path.is_dir()}
    annotation_samples = {
        path.name for path in annotation_root.glob("sample*") if path.is_dir()
    }
    samples = sorted(dense_samples & annotation_samples)
    if not samples:
        raise ValueError("共通sampleが見つかりません")
    return samples


def records_manifest_hash(records: list[SlabRecord]) -> str:
    """スラブindexの安定hashを返す。"""
    payload = [
        {
            "sample": record.sample,
            "vertebra": record.vertebra,
            "slice_indices": record.slice_indices,
            "label_indices": sorted(record.labels),
        }
        for record in records
    ]
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_records_for_samples(
    config: dict[str, Any],
    sample_names: list[str],
    require_labels: bool,
    stride: int | None = None,
) -> list[SlabRecord]:
    """設定とsample群からスラブindexを構築する。"""
    data_config = config["data"]
    selected_stride = stride
    if selected_stride is None:
        selected_stride = int(
            data_config["train_stride" if require_labels else "inference_stride"]
        )
    return build_slab_records(
        dense_root=Path(data_config["dense_root"]),
        annotation_root=Path(data_config["annotation_root"]),
        sample_names=sample_names,
        group=str(data_config.get("group", "ALL")),
        slab_size=int(data_config["slab_size"]),
        stride=selected_stride,
        min_labeled_slices=int(data_config.get("min_labeled_slices", 3)),
        require_labels=require_labels,
    )


def prepare_splits(
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """設定されたfoldのsample分割を返す。"""
    data_config = config["data"]
    samples = discover_samples(
        Path(data_config["dense_root"]),
        Path(data_config["annotation_root"]),
    )
    return kfold_split_samples(
        samples,
        n_folds=int(data_config.get("n_folds", 5)),
        test_fold=int(data_config.get("test_fold", 0)),
        seed=int(data_config.get("random_seed", 42)),
    )


def _make_loader(
    dataset: SlabLineDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader[dict[str, Any]]:
    """再現可能なDataLoaderを構築する。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_training_loaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    dict[str, str],
]:
    """train/validation/test DataLoaderとmanifest hashを返す。"""
    training_samples, validation_samples, test_samples = prepare_splits(config)
    data_config = config["data"]
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})
    record_groups = {
        "train": build_records_for_samples(config, training_samples, True),
        "validation": build_records_for_samples(config, validation_samples, True),
        "test": build_records_for_samples(config, test_samples, True),
    }
    for split_name, records in record_groups.items():
        if not records:
            raise ValueError(f"{split_name}スラブが0件です")
    datasets = {
        "train": SlabLineDataset(
            record_groups["train"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
            transform=get_transforms("train", augmentation_config),
            augmentation_config=augmentation_config,
        ),
        "validation": SlabLineDataset(
            record_groups["validation"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
        ),
        "test": SlabLineDataset(
            record_groups["test"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
        ),
    }
    batch_size = int(training_config.get("batch_size", 2))
    num_workers = int(training_config.get("num_workers", 4))
    prefetch_factor = int(training_config.get("prefetch_factor", 1))
    persistent_workers = bool(training_config.get("persistent_workers", True))
    seed = int(data_config.get("random_seed", 42))
    hashes = {
        name: records_manifest_hash(records) for name, records in record_groups.items()
    }
    return (
        _make_loader(
            datasets["train"],
            batch_size,
            num_workers,
            True,
            seed,
            prefetch_factor,
            persistent_workers,
        ),
        _make_loader(
            datasets["validation"],
            batch_size,
            num_workers,
            False,
            seed,
            prefetch_factor,
            persistent_workers,
        ),
        _make_loader(
            datasets["test"],
            batch_size,
            num_workers,
            False,
            seed,
            prefetch_factor,
            persistent_workers,
        ),
        hashes,
    )


def create_inference_loader(
    config: dict[str, Any],
    sample_names: list[str],
) -> tuple[DataLoader[dict[str, Any]], str]:
    """全高sliding推論用DataLoaderを作る。"""
    records = build_records_for_samples(config, sample_names, False)
    if not records:
        raise ValueError("推論スラブが0件です")
    data_config = config["data"]
    dataset = SlabLineDataset(
        records,
        image_size=int(data_config["image_size"]),
        sigma=float(data_config["sigma"]),
    )
    loader = _make_loader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 2)),
        num_workers=int(config["training"].get("num_workers", 4)),
        shuffle=False,
        seed=int(data_config.get("random_seed", 42)),
        prefetch_factor=int(config["training"].get("prefetch_factor", 1)),
        persistent_workers=bool(config["training"].get("persistent_workers", True)),
    )
    return loader, records_manifest_hash(records)
