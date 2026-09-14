"""Load and validate the weak-label region-MIL YAML configuration.

Mirrors ``region_branch/config/schema.py``'s "frozen research contract"
pattern (plain nested dicts, exact-value checks against frozen sub-dicts,
recursively rejected forbidden keys) adapted to this package's own sections.
Unlike ``region_branch``, there is only one initialization mode -- Baseline 0
encoder transfer is always required, never optional -- so there is no
``initialization`` switch to validate.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.weak.modeling.losses import (
    NOISY_OR_AGGREGATION,
    NORMALIZED_LSE_AGGREGATION,
    WHOLE_AGGREGATIONS,
)

PROTOCOL_VERSION = "weak-region-mil-v2"
LEGACY_PROTOCOL_VERSION = "weak-region-mil-v1"
REQUIRED_SECTIONS = {
    "protocol_version",
    "experiment",
    "data",
    "model",
    "loss",
    "sampling",
    "training",
    "augmentation",
    "parallel",
    "wandb",
}

# Settings this design explicitly rejected during review -- see
# fracture_detection/REGION_MIL_DESIGN.md sections 6-8. Reintroducing any of
# these would silently revert a deliberate design decision.
FORBIDDEN_CONFIG_KEYS = {
    "pos_weight",
    "lambda",
    "alpha",
    "ramp",
    "beta_ramp",
    "pseudo_arm",
    "pseudo_label_dir",
    "pseudo_coefficient",
    "cam_magnitude_weight",
    "calibration",
    "region_validity",
    "annotation_complete",
    "importance",
    "distribution_matching",
    "weighted_sampler",
    "balanced_sampler",
    "class_weight",
    "focal",
    "focal_loss",
    "detach",
    "ema",
    "temperature_scaling",
    "isotonic",
    "mixup_probability",
    "matched",
    "plateau_factor",
    "plateau_patience",
    "plateau_threshold",
    "plateau_cooldown",
}

FROZEN_MODEL: dict[str, object] = {
    "backbone": "tf_efficientnetv2_s",
    "pretrained": True,
    "in_chans": 6,
    "n_planes": 15,
    "drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "fpn_channels": 256,
    "fpn_stride": 4,
    "region_lstm_hidden": 128,
    "region_lstm_layers": 1,
    "region_head_dropout": 0.3,
    "freeze_encoder_bn_stats": True,
}
FROZEN_TRAINING: dict[str, object] = {
    "weight_decay": 1e-4,
    "gradient_clip_norm": None,
    "amp_dtype": "bfloat16",
    "lr_scheduler": "cosine_annealing",
    "transferred_learning_rate": 2.3e-5,
    "new_learning_rate": 2.3e-4,
    "transferred_min_learning_rate": 2.3e-6,
    "new_min_learning_rate": 2.3e-5,
    "max_gt_passes": 60,
    "min_gt_passes": 10,
    "selection_metric": "inner_region_macro_ap",
}
# baseline0's frozen augmentation recipe, used unchanged per the user's
# instruction ("baseline0でやったとこまで"). mixup_probability is NOT part of
# this section -- see FORBIDDEN_CONFIG_KEYS and weak/README.md for why MixUp
# cannot be applied to this model at all.
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
    """Load and validate the YAML config against this package's contract."""
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
    """Return a new config with the allowed CLI overrides applied."""
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
    """Validate the frozen contract and nested-fold runtime section."""
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"configに必要なsectionがありません: {sorted(missing)}")
    protocol_version = config["protocol_version"]
    if protocol_version not in {LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION}:
        raise ValueError(
            f"protocol_versionは{LEGACY_PROTOCOL_VERSION}または{PROTOCOL_VERSION}"
            "である必要があります"
        )
    _reject_forbidden_keys(config)

    _validate_experiment_identifiers(_section(config, "experiment"))

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
    if (
        not isinstance(model.get("baseline0_checkpoint_root"), str)
        or not model["baseline0_checkpoint_root"]
    ):
        raise ValueError(
            "model.baseline0_checkpoint_rootは非空文字列である必要があります"
        )
    loss = _section(config, "loss")
    _validate_loss(loss, str(protocol_version))
    _validate_sampling(_section(config, "sampling"), float(loss["beta"]))
    training = _section(config, "training")
    _require_exact_values(training, FROZEN_TRAINING, "training")
    patience = training.get("patience_gt_passes")
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("training.patience_gt_passesは1以上の整数である必要があります")
    if not isinstance(training.get("gpu_id"), int) or training["gpu_id"] < 0:
        raise ValueError("training.gpu_idは0以上の整数である必要があります")
    _require_exact_values(
        _section(config, "augmentation"), FROZEN_AUGMENTATION, "augmentation"
    )
    _validate_parallel(_section(config, "parallel"))

    wandb = _section(config, "wandb")
    if not isinstance(wandb.get("enabled"), bool):
        raise ValueError("wandb.enabledはboolである必要があります")

    _validate_runtime(config.get("runtime"))


def _validate_loss(loss: dict[str, Any], protocol_version: str) -> None:
    """Validate beta and the protocol-specific whole aggregation."""
    beta = loss.get("beta")
    if (
        not isinstance(beta, int | float)
        or isinstance(beta, bool)
        or not math.isfinite(beta)
        or beta < 0
    ):
        raise ValueError("loss.betaは0以上の数値である必要があります")
    if protocol_version == LEGACY_PROTOCOL_VERSION:
        if set(loss) != {"beta"}:
            raise ValueError("v1のloss設定はbetaのみである必要があります")
        return

    aggregation = loss.get("whole_aggregation")
    if aggregation not in WHOLE_AGGREGATIONS:
        raise ValueError(
            f"loss.whole_aggregationは{sorted(WHOLE_AGGREGATIONS)}が必要です"
        )
    lse_temperature = loss.get("lse_temperature")
    if aggregation == NOISY_OR_AGGREGATION:
        if lse_temperature is not None:
            raise ValueError("noisy_orではloss.lse_temperatureはnullが必要です")
        return
    if aggregation == NORMALIZED_LSE_AGGREGATION and (
        not isinstance(lse_temperature, int | float)
        or isinstance(lse_temperature, bool)
        or not math.isfinite(lse_temperature)
        or lse_temperature <= 0
    ):
        raise ValueError(
            "normalized_logsumexpではloss.lse_temperatureに0より大きい有限値が必要です"
        )


def _validate_sampling(sampling: dict[str, Any], beta: float) -> None:
    """N/A counts must be positive; U may be 0, but only when beta is 0."""
    for key in ("negative_bags_per_batch", "annotated_bags_per_batch"):
        value = sampling.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"sampling.{key}は1以上の整数である必要があります")
    weak = sampling.get("weak_bags_per_batch")
    if not isinstance(weak, int) or isinstance(weak, bool) or weak < 0:
        raise ValueError(
            "sampling.weak_bags_per_batchは0以上の整数である必要があります"
        )
    # With no U bag in any batch a positive beta would silently do nothing.
    if weak == 0 and beta != 0:
        raise ValueError(
            "sampling.weak_bags_per_batch=0はloss.beta=0の場合だけ指定できます"
        )


def _validate_parallel(parallel: dict[str, Any]) -> None:
    """Validate the fold-process parallel launch configuration."""
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


def _validate_outer_fold_range(data: dict[str, Any]) -> None:
    """Validate the inclusive outer-fold training range."""
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
    """Validate that experiment identifiers are safe relative path segments."""
    for key in ("phase", "name"):
        value = experiment.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"experiment.{key}は必須です")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"experiment.{key}にpath区切りは使えません")


def _validate_runtime(runtime: Any) -> None:
    """Validate the CLI-injected nested-fold runtime section, if present."""
    if runtime is None:
        return
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


def _require_exact_values(
    section: dict[str, Any], expected: dict[str, object], section_name: str
) -> None:
    """Report every frozen-value mismatch in one error."""
    mismatches = {
        key: (section.get(key), value)
        for key, value in expected.items()
        if section.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{section_name}の凍結設定が不正です: {mismatches}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required mapping section."""
    section = config[name]
    if not isinstance(section, dict):
        raise ValueError(f"config.{name}はmappingである必要があります")
    return section


def _reject_forbidden_keys(value: object) -> None:
    """Recursively reject settings this design explicitly rejected."""
    if isinstance(value, dict):
        invalid = FORBIDDEN_CONFIG_KEYS & set(value)
        if invalid:
            raise ValueError(f"weakで禁止された設定があります: {sorted(invalid)}")
        for child in value.values():
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)
