"""region_branchのYAML設定を読み込み、凍結研究契約を検証する。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.region_branch.data_pipeline.constants import N_REGIONS

PROTOCOL_VERSION = "region-branch-v8"
REQUIRED_SECTIONS = {
    "protocol_version",
    "experiment",
    "calibration",
    "data",
    "model",
    "region",
    "training",
    "augmentation",
    "parallel",
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
    "region_pos_weight",
    "cam_magnitude_weight",
    "alpha",
    "rank_temperature",
    "pseudo_coefficient",
    "pseudo_loss_weight",
    "confidence_weight",
    "ramp",
    "pseudo_ramp",
    "human_bags_per_batch",
    "negative_bags_per_batch",
    "pseudo_bags_per_batch",
}
PSEUDO_ARMS = ("no_pseudo", "cam_soft", "cam_soft_shuffled")
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
FROZEN_REGION: dict[str, object] = {
    "fpn_channels": 256,
    "fpn_stride": 4,
    "region_lstm_hidden": 256,
    "region_lstm_layers": 2,
    "region_head_dropout": 0.3,
    "diagnostic_subset_size": 256,
    "collapse_spearman_threshold": 0.95,
    "collapse_consecutive_epochs": 3,
}
FROZEN_TRAINING: dict[str, object] = {
    "natural_batch_size": 16,
    "pos_weight": 2.0,
    "max_epochs": 75,
    "min_epoch": 1,
    "early_stopping_patience": 20,
    "early_stopping_metric": "val_region_centered_loss",
    "weight_decay": 1e-4,
    "gradient_clip_norm": None,
    "amp_dtype": "bfloat16",
    "lr_scheduler": "cosine_annealing",
    "mixup_probability": 0.2,
}
INITIALIZATION_CONTRACTS: dict[str, dict[str, dict[str, object]]] = {
    "baseline0_fold_matched": {
        "model": {
            "pretrained": False,
            "baseline0_checkpoint_root": "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加",
        },
        "training": {
            "pretrained_learning_rate": 2.3e-5,
            "region_learning_rate": 2.3e-4,
            "pretrained_min_learning_rate": 2.3e-6,
            "region_min_learning_rate": 2.3e-5,
        },
    },
    "joint_from_start": {
        "model": {
            "pretrained": True,
            "baseline0_checkpoint_root": None,
        },
        "training": {
            "pretrained_learning_rate": 2.3e-4,
            "region_learning_rate": 2.3e-4,
            "pretrained_min_learning_rate": 2.3e-5,
            "region_min_learning_rate": 2.3e-5,
        },
    },
}
FROZEN_AUGMENTATION: dict[str, object] = {
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "transpose_probability": 0.5,
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
    """YAMLを読み込み、region_branchの契約を検証して返す。"""
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
    """region_branchの凍結設定とnested選択契約を検証する。"""
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"configに必要なsectionがありません: {sorted(missing)}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"protocol_versionは{PROTOCOL_VERSION}である必要があります")
    _reject_forbidden_keys(config)

    experiment = _section(config, "experiment")
    _validate_experiment_identifiers(experiment)

    calibration = _section(config, "calibration")
    _validate_calibration_version(calibration)

    data = _section(config, "data")
    if data.get("random_seed") != 20260807:
        raise ValueError("data.random_seedは凍結値20260807である必要があります")
    if data.get("n_folds") != 5:
        raise ValueError("data.n_foldsは5である必要があります")
    if not isinstance(data.get("num_workers"), int) or data["num_workers"] < 0:
        raise ValueError("data.num_workersは0以上の整数である必要があります")
    _validate_outer_fold_range(data)

    model = _section(config, "model")
    _require_exact_values(model, FROZEN_MODEL, "model")
    region = _section(config, "region")
    _require_exact_values(region, FROZEN_REGION, "region")
    _validate_active_regions(region)
    _validate_pseudo_arm(region)

    training = _section(config, "training")
    _require_exact_values(training, FROZEN_TRAINING, "training")
    _validate_initialization_contract(model, training)
    if not isinstance(training.get("gpu_id"), int) or training["gpu_id"] < 0:
        raise ValueError("training.gpu_idは0以上の整数である必要があります")

    augmentation = _section(config, "augmentation")
    _require_exact_values(augmentation, FROZEN_AUGMENTATION, "augmentation")

    _validate_parallel(_section(config, "parallel"))

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


def _validate_parallel(parallel: dict[str, Any]) -> None:
    """fold-process並列設定を検証する。"""
    if parallel.get("mode") not in {"single", "fold"}:
        raise ValueError("parallel.modeはsingleまたはfoldが必要です")
    gpu_ids = parallel.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(value, int) or value < 0 for value in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("parallel.gpu_idsは重複のない0以上の整数listが必要です")
    concurrency = parallel.get("max_concurrent_folds")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or not 1 <= concurrency <= len(gpu_ids)
    ):
        raise ValueError("parallel.max_concurrent_foldsは1以上GPU数以下が必要です")


def _validate_initialization_contract(
    model: dict[str, Any], training: dict[str, Any]
) -> None:
    """初期化方式と、それに対応するpretrained設定・学習率を検証する。"""
    initialization = model.get("initialization")
    contract = INITIALIZATION_CONTRACTS.get(initialization)
    if contract is None:
        raise ValueError(
            "model.initializationはbaseline0_fold_matchedまたは"
            "joint_from_startが必要です"
        )
    _require_exact_values(model, contract["model"], "model initialization")
    _require_exact_values(training, contract["training"], "training initialization")


def _validate_active_regions(region: dict[str, Any]) -> None:
    """active_regionsが0..N_REGIONS-1の非空・重複なし集合であることを検証する。"""
    active_regions = region.get("active_regions")
    if (
        not isinstance(active_regions, list)
        or not active_regions
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in active_regions
        )
    ):
        raise ValueError("region.active_regionsは整数の非空listである必要があります")
    if len(set(active_regions)) != len(active_regions):
        raise ValueError("region.active_regionsに重複があります")
    if not set(active_regions).issubset(range(N_REGIONS)):
        raise ValueError(f"region.active_regionsは0..{N_REGIONS - 1}の範囲が必要です")


def _validate_pseudo_arm(region: dict[str, Any]) -> None:
    """`region.pseudo_arm`が既知のアームであることを検証する。

    `pseudo_label_dir`は`null`のままでよい（呼び出し側が
    `DEFAULT_PSEUDO_LABEL_DIR`へfallbackする既存規約に従う）ため、
    ここでは値自体を要求しない。
    """
    pseudo_arm = region.get("pseudo_arm")
    if pseudo_arm not in PSEUDO_ARMS:
        raise ValueError(f"region.pseudo_armは{PSEUDO_ARMS}のいずれかが必要です")


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
    """experiment識別子が安全な相対path要素であることを検証する。"""
    for key in ("phase", "name"):
        value = experiment.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"experiment.{key}は必須です")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"experiment.{key}にpath区切りは使えません")


def _validate_calibration_version(calibration: dict[str, Any]) -> None:
    """校正versionを`v1`, `v2`, ...の安全なpath要素へ制限する。"""
    version = calibration.get("version")
    if not isinstance(version, str) or re.fullmatch(r"v[1-9][0-9]*", version) is None:
        raise ValueError("calibration.versionはv1, v2, ...の形式が必要です")


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
    """廃止済み・不採用の設定を再帰的に拒否する。"""
    if isinstance(value, dict):
        invalid = FORBIDDEN_CONFIG_KEYS & set(value)
        if invalid:
            raise ValueError(
                f"region_branchで禁止された設定があります: {sorted(invalid)}"
            )
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)
