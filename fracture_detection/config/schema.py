"""MTLアーム統一configの既定値・検証・CLI上書き。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from fracture_detection.common.splits import resolve_nested_folds

PROTOCOL_VERSION = "fracture-mtl-v1"
ARM_CONTRACTS: dict[str, dict[str, object]] = {
    "baseline0": {
        "kind": "baseline0",
        "package": "baseline0",
        "input_channels": 6,
        "whole_method": "independent",
        "region_enabled": False,
        "attention_enabled": False,
        "beta_mode": "zero",
    },
    "control_b": {
        "kind": "mtl",
        "package": "mtl",
        "input_channels": 6,
        "whole_method": "independent",
        "region_enabled": True,
        "attention_enabled": False,
        "beta_mode": "zero",
    },
    "baseline1_b": {
        "kind": "mtl",
        "package": "mtl",
        "input_channels": 10,
        "whole_method": "independent",
        "region_enabled": True,
        "attention_enabled": False,
        "beta_mode": "zero",
    },
    "proposed_b": {
        "kind": "proposed",
        "package": "proposed",
        "input_channels": 10,
        "whole_method": "independent",
        "region_enabled": True,
        "attention_enabled": True,
        "beta_mode": "calibrated",
    },
    "proposed_max": {
        "kind": "proposed",
        "package": "proposed",
        "input_channels": 10,
        "whole_method": "max",
        "region_enabled": True,
        "attention_enabled": True,
        "beta_mode": "calibrated",
    },
    "proposed_max_beta0": {
        "kind": "proposed",
        "package": "proposed",
        "input_channels": 10,
        "whole_method": "max",
        "region_enabled": True,
        "attention_enabled": True,
        "beta_mode": "zero",
    },
}


def default_config() -> dict[str, Any]:
    """全構成で共有する凍結既定値を返す。"""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data": {
            "random_seed": 20260807,
            "n_folds": 5,
            "start_outer_fold": 0,
            "end_outer_fold": 4,
            "dataset_dir": None,
            "stage_to_local": True,
            "stage_root": "/dev/shm/vai-fracture-dataset",
            "stage_copy_workers": 8,
            "num_workers": 8,
        },
        "model": {
            "backbone": "tf_efficientnetv2_s",
            "pretrained": True,
            "n_planes": 15,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "head_dropout": 0.3,
            "lstm_hidden": 256,
            "lstm_layers": 2,
        },
        "training": {
            "gpu_id": 0,
            "natural_batch_size": 16,
            "annotated_batch_size": 1,
            "pos_weight": 2.0,
            "max_epochs": 75,
            "min_epoch": 1,
            "early_stopping_patience": 20,
            "early_stopping_metric": "val_bce",
            "weight_decay": 1e-4,
            "gradient_clip_norm": None,
            "amp_dtype": "bfloat16",
            "freeze_backbone_epochs": 0,
            "warmup_epochs": 0,
            "warmup_start_factor": 1.0,
            "backbone_learning_rate": 2.3e-4,
            "head_learning_rate": 2.3e-4,
            "backbone_min_learning_rate": 2.3e-5,
            "head_min_learning_rate": 2.3e-5,
            "lr_scheduler": "cosine_annealing",
            "mixup_probability": 0.2,
        },
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "affine_probability": 0.7,
            "shift_limit": 0.3,
            "scale_lower": 0.7,
            "scale_upper": 1.3,
            "rotate_limit": 45.0,
            "border_mode": 4,
            "brightness_limit": 0.1,
            "contrast_limit": 0.0,
            "intensity_probability": 0.7,
            "blur_noise_probability": 0.5,
            "noise_variance_lower": 3.0,
            "noise_variance_upper": 9.0,
            "distortion_probability": 0.5,
            "cutout_probability": 0.05,
            "cutout_ratio": 0.5,
        },
        "parallel": {
            "mode": "single",
            "gpu_ids": [0],
            "max_concurrent_folds": 1,
        },
        "calibration": {
            "loss_weights_path": "fracture_detection/calibration/outputs/loss_weights.json",
            "lambda_artifact_path": "fracture_detection/calibration/outputs/lambda_calibration.json",
            "beta_artifact_path": "fracture_detection/calibration/outputs/beta_calibration.json",
        },
        "freeze": {
            "manifest_path": "fracture_detection/experiments/frozen_experiment_manifest.json"
        },
        "wandb": {"enabled": True, "project": None, "run_name": None},
    }


def load_config(path: Path) -> dict[str, Any]:
    """minimal arm YAMLを完全な実効configへ展開する。"""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config最上位はmappingである必要があります")
    resolved = _deep_merge(default_config(), loaded)
    validate_config(resolved)
    return resolved


def apply_overrides(
    config: dict[str, Any],
    *,
    outer_fold: int | None = None,
    gpu_id: int | None = None,
    start_outer_fold: int | None = None,
    end_outer_fold: int | None = None,
) -> dict[str, Any]:
    """許可されたruntime上書きをcopyへ適用する。"""
    resolved = copy.deepcopy(config)
    if outer_fold is not None:
        assignment = resolve_nested_folds(outer_fold)
        resolved["runtime"] = {
            "outer_fold": assignment.outer_fold,
            "inner_fold": assignment.inner_fold,
            "train_folds": list(assignment.train_folds),
        }
    if gpu_id is not None:
        resolved["training"]["gpu_id"] = gpu_id
    if start_outer_fold is not None:
        resolved["data"]["start_outer_fold"] = start_outer_fold
    if end_outer_fold is not None:
        resolved["data"]["end_outer_fold"] = end_outer_fold
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """構成間比較を壊すconfig driftを拒否する。"""
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_versionは{PROTOCOL_VERSION}が必要です")
    arm = _mapping(config, "arm")
    name = arm.get("name")
    if name not in ARM_CONTRACTS:
        raise ValueError(f"未登録armです: {name}")
    expected_arm = {"name": name, **ARM_CONTRACTS[cast(str, name)]}
    if arm != expected_arm:
        raise ValueError(f"arm契約が不正です: expected={expected_arm}, actual={arm}")
    experiment = _mapping(config, "experiment")
    for key in ("phase", "name"):
        value = experiment.get(key)
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise ValueError(f"experiment.{key}が不正です")
        if "/" in value or "\\" in value:
            raise ValueError(f"experiment.{key}にpath区切りは使えません")
    data = _mapping(config, "data")
    if data["random_seed"] != 20260807 or data["n_folds"] != 5:
        raise ValueError("seed/fold数は凍結値が必要です")
    start, end = data["start_outer_fold"], data["end_outer_fold"]
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start <= end < 5
    ):
        raise ValueError("outer fold範囲が不正です")
    model = _mapping(config, "model")
    frozen_model = default_config()["model"]
    if model != frozen_model:
        raise ValueError("model設定が凍結値と一致しません")
    training = _mapping(config, "training")
    frozen_training = default_config()["training"]
    if {key: value for key, value in training.items() if key != "gpu_id"} != {
        key: value for key, value in frozen_training.items() if key != "gpu_id"
    }:
        raise ValueError("training設定が凍結値と一致しません")
    if not isinstance(training["gpu_id"], int) or training["gpu_id"] < 0:
        raise ValueError("training.gpu_idが不正です")
    if _mapping(config, "augmentation") != default_config()["augmentation"]:
        raise ValueError("augmentation設定が凍結値と一致しません")
    parallel = _mapping(config, "parallel")
    if parallel.get("mode") not in {"single", "fold"}:
        raise ValueError("parallel.modeはsingleまたはfoldが必要です")
    gpu_ids = parallel.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("parallel.gpu_idsは重複なしの非空listが必要です")
    if not all(isinstance(value, int) and value >= 0 for value in gpu_ids):
        raise ValueError("parallel.gpu_idsが不正です")
    concurrency = parallel.get("max_concurrent_folds")
    if not isinstance(concurrency, int) or not 1 <= concurrency <= len(gpu_ids):
        raise ValueError("max_concurrent_foldsが不正です")
    runtime = config.get("runtime")
    if runtime is not None:
        assignment = resolve_nested_folds(int(runtime["outer_fold"]))
        if runtime != {
            "outer_fold": assignment.outer_fold,
            "inner_fold": assignment.inner_fold,
            "train_folds": list(assignment.train_folds),
        }:
            raise ValueError("runtimeがnested fold契約と一致しません")


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key}はmappingが必要です")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
