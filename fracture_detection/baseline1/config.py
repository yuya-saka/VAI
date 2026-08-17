"""Baseline 1のYAML設定を読み込み、研究契約を検証する。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

FORBIDDEN_CONFIG_KEYS = {
    "positive_weight",
    "focal",
    "focal_loss",
    "class_weight",
    "weighted_sampler",
    "balanced_sampler",
    "mixup",
    "ema",
}
REQUIRED_SECTIONS = {"experiment", "data", "model", "training", "augmentation", "wandb"}
MATCHED_OPTIMIZATION_SCHEDULE: dict[str, object] = {
    "max_epochs": 100,
    "min_epoch": 1,
    "early_stopping_patience": 15,
    "freeze_backbone_epochs": 0,
    "warmup_epochs": 2,
    "warmup_start_factor": 0.1,
    "backbone_learning_rate": 1e-4,
    "head_learning_rate": 3e-4,
    "backbone_min_learning_rate": 1e-6,
    "head_min_learning_rate": 3e-6,
    "lr_scheduler": "reduce_on_plateau",
    "plateau_factor": 0.5,
    "plateau_patience": 4,
    "plateau_threshold": 0.001,
    "plateau_cooldown": 1,
}
FULL_OPTIMIZATION_SCHEDULE: dict[str, object] = {
    "max_epochs": 75,
    "min_epoch": 1,
    "early_stopping_patience": 15,
    "freeze_backbone_epochs": 0,
    "warmup_epochs": 0,
    "warmup_start_factor": 1.0,
    "backbone_learning_rate": 2.3e-4,
    "head_learning_rate": 2.3e-4,
    "backbone_min_learning_rate": 2.3e-5,
    "head_min_learning_rate": 2.3e-5,
    "lr_scheduler": "reduce_on_plateau",
    "plateau_factor": 0.5,
    "plateau_patience": 4,
    "plateau_threshold": 0.001,
    "plateau_cooldown": 1,
}


def load_config(path: Path) -> dict[str, Any]:
    """YAMLを読み込み、Baseline 1の契約を検証して返す。"""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("configの最上位はmappingである必要があります")
    validate_config(loaded)
    return loaded


def apply_cli_overrides(
    config: dict[str, Any],
    fold: int | None = None,
    gpu_id: int | None = None,
    start_fold: int | None = None,
    end_fold: int | None = None,
) -> dict[str, Any]:
    """CLIの上書きを適用した新しい設定を返す。"""
    resolved = copy.deepcopy(config)
    if fold is not None:
        resolved.setdefault("runtime", {})["fold"] = fold
    if gpu_id is not None:
        resolved["training"]["gpu_id"] = gpu_id
    if start_fold is not None:
        resolved["data"]["start_fold"] = start_fold
    if end_fold is not None:
        resolved["data"]["end_fold"] = end_fold
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """確定済みのBaseline 1設定の契約を検証する。"""
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"configに必要なsectionがありません: {sorted(missing)}")
    _reject_forbidden_keys(config)

    experiment = _section(config, "experiment")
    if not isinstance(experiment.get("phase"), str) or not experiment["phase"]:
        raise ValueError("experiment.phaseは必須です")
    if not isinstance(experiment.get("name"), str) or not experiment["name"]:
        raise ValueError("experiment.nameは必須です")

    data = _section(config, "data")
    mode = data.get("mode")
    if mode not in {"matched", "full"}:
        raise ValueError("data.modeはmatchedまたはfullである必要があります")
    if data.get("random_seed") != 20260807:
        raise ValueError("data.random_seedは凍結値20260807である必要があります")
    if data.get("n_folds") != 5:
        raise ValueError("data.n_foldsは5である必要があります")
    _validate_fold_range(data)
    if bool(data.get("stage_to_local")) != (mode == "full"):
        raise ValueError("data.stage_to_localはfullのみtrueである必要があります")

    model = _section(config, "model")
    if model.get("in_chans") != 6 or model.get("n_planes") != 15:
        raise ValueError("model.in_chans=6かつmodel.n_planes=15が必要です")
    if model.get("lstm_hidden") != 256 or model.get("lstm_layers") != 2:
        raise ValueError("BiLSTMはhidden=256、layers=2に固定されています")
    if not isinstance(model.get("backbone"), str):
        raise ValueError("model.backboneは必須です")

    training = _section(config, "training")
    if training.get("batch_size") != 16:
        raise ValueError("training.batch_sizeは16に固定されています")
    if training.get("pos_weight") != 2.0:
        raise ValueError("training.pos_weightは2.0に固定されています")
    if training.get("weight_decay") != 1e-4:
        raise ValueError("training.weight_decayは1e-4に固定されています")
    if training.get("gradient_clip_norm") != 5.0:
        raise ValueError("training.gradient_clip_normは5.0に固定されています")
    if training.get("amp_dtype") != "bfloat16":
        raise ValueError("training.amp_dtypeはbfloat16に固定されています")
    _validate_mode_schedule(mode, model, training)

    augmentation = _section(config, "augmentation")
    prohibited_augmentation = {
        "horizontal_flip",
        "vertical_flip",
        "transpose",
        "distortion",
        "cutout",
    }
    present = prohibited_augmentation & set(augmentation)
    if present:
        raise ValueError(f"禁止augmentation設定があります: {sorted(present)}")

    wandb = _section(config, "wandb")
    if not isinstance(wandb.get("enabled"), bool):
        raise ValueError("wandb.enabledはboolである必要があります")


def _validate_mode_schedule(
    mode: str, model: dict[str, Any], training: dict[str, Any]
) -> None:
    """設定別のバックボーン・epoch・早期終了の契約を検証する。"""
    if mode == "matched":
        if model["backbone"] not in {"tf_efficientnetv2_b0", "tf_efficientnetv2_s"}:
            raise ValueError("matched backboneはtf_efficientnetv2_b0または_sです")
        _require_optimization_schedule(training, MATCHED_OPTIMIZATION_SCHEDULE)
        return
    if model["backbone"] != "tf_efficientnetv2_s":
        raise ValueError("full backboneはtf_efficientnetv2_sに固定されています")
    _require_optimization_schedule(training, FULL_OPTIMIZATION_SCHEDULE)


def _require_optimization_schedule(
    training: dict[str, Any], expected: dict[str, object]
) -> None:
    """確定したoptimization scheduleとの完全一致を検証する。"""
    mismatches = {
        key: (training.get(key), value)
        for key, value in expected.items()
        if training.get(key) != value
    }
    if mismatches:
        raise ValueError(f"training optimization scheduleが不正です: {mismatches}")


def _validate_fold_range(data: dict[str, Any]) -> None:
    """学習対象foldの両端が有効な包含範囲か検証する。"""
    n_folds = data["n_folds"]
    start_fold = data.get("start_fold")
    end_fold = data.get("end_fold")
    if (
        not isinstance(start_fold, int)
        or isinstance(start_fold, bool)
        or not isinstance(end_fold, int)
        or isinstance(end_fold, bool)
        or not 0 <= start_fold <= end_fold < n_folds
    ):
        raise ValueError(
            "dataのfold範囲は0 <= start_fold <= end_fold < n_foldsが必要です"
        )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """辞書である必須セクションを返す。"""
    section = config[name]
    if not isinstance(section, dict):
        raise ValueError(f"config.{name}はmappingである必要があります")
    return section


def _reject_forbidden_keys(value: object) -> None:
    """禁止した重み付け・サンプリング・EMA設定を再帰的に拒否する。"""
    if isinstance(value, dict):
        invalid = FORBIDDEN_CONFIG_KEYS & set(value)
        if invalid:
            raise ValueError(f"Baseline 1で禁止された設定があります: {sorted(invalid)}")
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)
