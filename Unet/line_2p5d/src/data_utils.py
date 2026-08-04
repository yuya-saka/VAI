"""設定、sample分割、DatasetとDataLoaderの構築。"""

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
    CenteredLineDataset,
    CenteredSliceRecord,
    build_centered_records,
    get_transforms,
    validate_context_offsets,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML設定を読み込み、必須値を検証する。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """2.5D入力と学習設定の整合性を検証する。"""
    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    folds_config = config.get("folds", {})
    training_config = config.get("training", {})
    for key in ("phase", "name"):
        if not str(experiment_config.get(key, "")).strip():
            raise ValueError(f"experiment.{key}は必須です")
    for key in ("annotation_root", "dense_root"):
        if not data_config.get(key):
            raise ValueError(f"data.{key}は必須です")
    offsets = validate_context_offsets(
        data_config.get("context_offsets", [-2, -1, 0, 1, 2])
    )
    if offsets != (-2, -1, 0, 1, 2):
        raise ValueError("最初の実験ではcontext_offsetsを[-2,-1,0,1,2]に固定します")
    if int(training_config.get("num_workers", 4)) < 0:
        raise ValueError("num_workersは0以上が必要です")
    if int(training_config.get("batch_size", 8)) < 1:
        raise ValueError("batch_sizeは1以上が必要です")
    n_folds = int(data_config.get("n_folds", 5))
    start_fold = int(folds_config.get("start", 0))
    end_fold = int(folds_config.get("end", n_folds - 1))
    if not 0 <= start_fold <= end_fold < n_folds:
        raise ValueError("folds.start/endがdata.n_foldsの範囲外です")
    geometry_config = config.get("loss", {}).get("geometry", {})
    if int(geometry_config.get("start_epoch", 0)) < 0:
        raise ValueError("loss.geometry.start_epochは0以上が必要です")
    if int(geometry_config.get("ramp_epochs", 0)) < 0:
        raise ValueError("loss.geometry.ramp_epochsは0以上が必要です")
    evaluation_config = config.get("evaluation", {})
    threshold_config = evaluation_config.get("heatmap_threshold", {})
    threshold_mode = str(threshold_config.get("mode", "adaptive"))
    if threshold_mode not in {"adaptive", "fixed"}:
        raise ValueError("evaluation.heatmap_threshold.modeが不正です")
    if int(evaluation_config.get("metrics_frequency", 1)) != 1:
        raise ValueError("現在のcheckpoint選択ではmetrics_frequencyは1に固定です")
    if float(evaluation_config.get("line_extend_ratio", 1.0)) <= 0:
        raise ValueError("evaluation.line_extend_ratioは0より大きい値が必要です")
    if int(evaluation_config.get("visualization_samples", 16)) < 0:
        raise ValueError("evaluation.visualization_samplesは0以上が必要です")


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


def discover_samples(dense_root: Path, annotation_root: Path) -> list[str]:
    """密画像と手動annotationの両方に存在するsampleを列挙する。"""
    dense = {path.name for path in dense_root.glob("sample*") if path.is_dir()}
    annotations = {
        path.name for path in annotation_root.glob("sample*") if path.is_dir()
    }
    samples = sorted(dense & annotations)
    if not samples:
        raise ValueError("共通sampleが見つかりません")
    return samples


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


def prepare_splits(config: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
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


def build_records_for_samples(
    config: dict[str, Any],
    sample_names: list[str],
) -> list[CenteredSliceRecord]:
    """設定とsample群から中心画像recordを構築する。"""
    data_config = config["data"]
    return build_centered_records(
        dense_root=Path(data_config["dense_root"]),
        annotation_root=Path(data_config["annotation_root"]),
        sample_names=sample_names,
        group=str(data_config.get("group", "ALL")),
        context_offsets=tuple(data_config.get("context_offsets", [-2, -1, 0, 1, 2])),
    )


def records_manifest_hash(records: list[CenteredSliceRecord]) -> str:
    """教師画像と入力スライスの安定hashを返す。"""
    payload = [
        {
            "sample": record.sample,
            "vertebra": record.vertebra,
            "center": record.center_slice_index,
            "context": record.context_slice_indices,
        }
        for record in records
    ]
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _make_loader(
    dataset: CenteredLineDataset,
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


def create_data_loaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    dict[str, str],
]:
    """train/validation/test loaderとmanifest hashを返す。"""
    training_samples, validation_samples, test_samples = prepare_splits(config)
    record_groups = {
        "train": build_records_for_samples(config, training_samples),
        "validation": build_records_for_samples(config, validation_samples),
        "test": build_records_for_samples(config, test_samples),
    }
    for split_name, records in record_groups.items():
        if not records:
            raise ValueError(f"{split_name}の教師画像が0件です")

    data_config = config["data"]
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})
    datasets = {
        "train": CenteredLineDataset(
            record_groups["train"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
            transform=get_transforms("train", augmentation_config),
            augmentation_config=augmentation_config,
        ),
        "validation": CenteredLineDataset(
            record_groups["validation"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
        ),
        "test": CenteredLineDataset(
            record_groups["test"],
            image_size=int(data_config["image_size"]),
            sigma=float(data_config["sigma"]),
        ),
    }
    batch_size = int(training_config.get("batch_size", 8))
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
