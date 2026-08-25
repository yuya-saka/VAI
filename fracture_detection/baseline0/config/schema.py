"""Baseline 0のYAML設定を読み込み、凍結研究契約を検証する。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from fracture_detection.baseline0.data.splits import resolve_nested_folds

PROTOCOL_VERSION = "baseline0-nested-v8"
REQUIRED_SECTIONS = {
    "protocol_version",
    "experiment",
    "data",
    "model",
    "training",
    "augmentation",
    "wandb",
}
FORBIDDEN_CONFIG_KEYS = {
    "positive_weight",
    "focal",
    "focal_loss",
    "class_weight",
    "weighted_sampler",
    "balanced_sampler",
    "mixup",
    "ema",
    "matched",
    "plateau_factor",
    "plateau_patience",
    "plateau_threshold",
    "plateau_cooldown",
}
FROZEN_MODEL: dict[str, object] = {
    "backbone": "tf_efficientnetv2_s",
    "in_chans": 6,
    "n_planes": 15,
    "drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "head_dropout": 0.3,
    "lstm_hidden": 256,
    "lstm_layers": 2,
}
FROZEN_TRAINING: dict[str, object] = {
    "natural_batch_size": 16,
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
}
FROZEN_AUGMENTATION: dict[str, object] = {
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
}


def load_config(path: Path) -> dict[str, Any]:
    """YAMLを読み込み、Baseline 0の契約を検証して返す。"""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("configの最上位はmappingである必要があります")
    validate_config(loaded)
    return loaded


def apply_cli_overrides(
    config: dict[str, Any],
    outer_fold: int | None = None,
    gpu_id: int | None = None,
    start_outer_fold: int | None = None,
    end_outer_fold: int | None = None,
) -> dict[str, Any]:
    """許可したCLI上書きを適用した新しい設定を返す。"""
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
    """Baseline 0の凍結設定とnested選択契約を検証する。"""
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"configに必要なsectionがありません: {sorted(missing)}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"protocol_versionは{PROTOCOL_VERSION}である必要があります")
    _reject_forbidden_keys(config)

    experiment = _section(config, "experiment")
    _validate_experiment_identifiers(experiment)

    data = _section(config, "data")
    if data.get("random_seed") != 20260807:
        raise ValueError("data.random_seedは凍結値20260807である必要があります")
    if data.get("n_folds") != 5:
        raise ValueError("data.n_foldsは5である必要があります")
    if data.get("stage_to_local") is not True:
        raise ValueError("data.stage_to_localはtrueである必要があります")
    stage_copy_workers = data.get("stage_copy_workers")
    if (
        not isinstance(stage_copy_workers, int)
        or isinstance(stage_copy_workers, bool)
        or not 1 <= stage_copy_workers <= 32
    ):
        raise ValueError("data.stage_copy_workersは1から32の整数である必要があります")
    if not isinstance(data.get("num_workers"), int) or data["num_workers"] < 0:
        raise ValueError("data.num_workersは0以上の整数である必要があります")
    _validate_outer_fold_range(data)

    model = _section(config, "model")
    _require_exact_values(model, FROZEN_MODEL, "model")
    if not isinstance(model.get("pretrained"), bool):
        raise ValueError("model.pretrainedはboolである必要があります")

    training = _section(config, "training")
    _require_exact_values(training, FROZEN_TRAINING, "training")
    if not isinstance(training.get("gpu_id"), int) or training["gpu_id"] < 0:
        raise ValueError("training.gpu_idは0以上の整数である必要があります")

    augmentation = _section(config, "augmentation")
    # horizontal flipだけは許可する。R2=right_transverse_foramenと
    # R3=left_transverse_foramenは左右対称の同種構造なので、鏡像化と同時に
    # ラベルを入れ替えれば意味論が保存される（common.dataset.flip_horizontal参照）。
    # vertical flipとtransposeはR1=vertebral_bodyとR4=posterior_elementsを
    # 入れ替えることになるが、この2つは鏡像関係にない別種の構造のため、
    # 正しい入れ替えが存在しない。恒久的に禁止する。
    prohibited = {
        "vertical_flip",
        "vertical_flip_probability",
        "transpose",
        "transpose_probability",
    }
    present = prohibited & set(augmentation)
    if present:
        raise ValueError(f"禁止augmentation設定があります: {sorted(present)}")
    _require_exact_values(augmentation, FROZEN_AUGMENTATION, "augmentation")

    wandb = _section(config, "wandb")
    if not isinstance(wandb.get("enabled"), bool):
        raise ValueError("wandb.enabledはboolである必要があります")

    runtime = config.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise ValueError("config.runtimeはmappingである必要があります")
        outer_fold = runtime.get("outer_fold")
        if not isinstance(outer_fold, int) or isinstance(outer_fold, bool):
            raise ValueError("runtime.outer_foldは整数である必要があります")
        if outer_fold not in range(5):
            raise ValueError("runtime.outer_foldは0から4である必要があります")
        assignment = resolve_nested_folds(outer_fold)
        if runtime.get("inner_fold") != assignment.inner_fold:
            raise ValueError("runtime.inner_foldがcyclic innerと一致しません")
        if runtime.get("train_folds") != list(assignment.train_folds):
            raise ValueError("runtime.train_foldsがnested契約と一致しません")


def _validate_outer_fold_range(data: dict[str, Any]) -> None:
    """学習対象outer foldの包含範囲を検証する。"""
    start_fold = data.get("start_outer_fold")
    end_fold = data.get("end_outer_fold")
    if (
        not isinstance(start_fold, int)
        or isinstance(start_fold, bool)
        or not isinstance(end_fold, int)
        or isinstance(end_fold, bool)
        or not 0 <= start_fold <= end_fold < 5
    ):
        raise ValueError("outer fold範囲は0 <= start <= end < 5が必要です")


def _validate_experiment_identifiers(experiment: dict[str, Any]) -> None:
    """Validate experiment identifiers as safe relative path components."""
    for key in ("phase", "name"):
        value = experiment.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"experiment.{key}は必須です")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"experiment.{key}にpath区切りは使えません")


def _require_exact_values(
    section: dict[str, Any], expected: dict[str, object], section_name: str
) -> None:
    """凍結値との不一致をまとめて報告する。"""
    mismatches = {
        key: (section.get(key), value)
        for key, value in expected.items()
        if section.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{section_name}の凍結設定が不正です: {mismatches}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """辞書である必須sectionを返す。"""
    section = config[name]
    if not isinstance(section, dict):
        raise ValueError(f"config.{name}はmappingである必要があります")
    return section


def _reject_forbidden_keys(value: object) -> None:
    """廃止済みの設定を再帰的に拒否する。"""
    if isinstance(value, dict):
        invalid = FORBIDDEN_CONFIG_KEYS & set(value)
        if invalid:
            raise ValueError(f"Baseline 0で禁止された設定があります: {sorted(invalid)}")
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)
