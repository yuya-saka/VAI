from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from fracture_detection.baseline0.config.schema import (
    apply_cli_overrides,
    validate_config,
)

CONFIG_PATH = Path("fracture_detection/baseline0/config/baseline0.yaml")


def _config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def test_validate_config_accepts_frozen_baseline0_contract() -> None:
    validate_config(_config())


def test_apply_cli_overrides_adds_nested_assignment() -> None:
    config = _config()

    resolved = apply_cli_overrides(
        config, outer_fold=4, start_outer_fold=2, end_outer_fold=4
    )

    assert resolved["runtime"] == {
        "outer_fold": 4,
        "inner_fold": 0,
        "train_folds": [1, 2, 3],
    }
    assert resolved["data"]["start_outer_fold"] == 2
    assert resolved["data"]["end_outer_fold"] == 4
    assert "runtime" not in config


@pytest.mark.parametrize(
    ("start_fold", "end_fold"),
    [(-1, 4), (0, 5), (3, 2), (True, 4)],
)
def test_validate_config_rejects_invalid_outer_range(
    start_fold: object, end_fold: object
) -> None:
    config = _config()
    config["data"]["start_outer_fold"] = start_fold
    config["data"]["end_outer_fold"] = end_fold

    with pytest.raises(ValueError, match="outer fold範囲"):
        validate_config(config)


def test_validate_config_rejects_rsna_contract_drift() -> None:
    config = _config()
    config["model"]["backbone"] = "tf_efficientnetv2_b0"

    with pytest.raises(ValueError, match="modelの凍結設定"):
        validate_config(config)


def test_validate_config_rejects_removed_matched_mode() -> None:
    config = _config()
    config["data"]["matched"] = True

    with pytest.raises(ValueError, match="禁止された設定"):
        validate_config(config)


def test_validate_config_rejects_vertical_flip_and_transpose() -> None:
    """R1/R4に正しい入れ替えが存在しない反転は恒久的に禁止する。"""
    for key in (
        "vertical_flip",
        "vertical_flip_probability",
        "transpose",
        "transpose_probability",
    ):
        config = _config()
        config["augmentation"][key] = 0.5

        with pytest.raises(ValueError, match="禁止augmentation"):
            validate_config(config)


def test_validate_config_accepts_horizontal_flip() -> None:
    """R2/R3の入れ替えで意味論が保存されるためhorizontal flipは許可する。"""
    config = _config()

    validate_config(config)

    assert config["augmentation"]["horizontal_flip_probability"] == 0.5


def test_validate_config_rejects_stage1_augmentation_drift() -> None:
    config = _config()
    config["augmentation"]["distortion_probability"] = 0.4

    with pytest.raises(ValueError, match="augmentationの凍結設定"):
        validate_config(config)


def test_validate_config_rejects_mixup_probability_drift() -> None:
    config = _config()
    config["training"]["mixup_probability"] = 0.0

    with pytest.raises(ValueError, match="trainingの凍結設定"):
        validate_config(config)


def test_validate_config_rejects_retired_plateau_scheduler() -> None:
    config = _config()
    config["training"]["plateau_patience"] = 4

    with pytest.raises(ValueError, match="禁止された設定"):
        validate_config(config)


def test_validate_config_rejects_auroc_early_stopping() -> None:
    config = _config()
    config["training"]["early_stopping_metric"] = "val_auroc"

    with pytest.raises(ValueError, match="trainingの凍結設定"):
        validate_config(config)


def test_validate_config_rejects_invalid_stage_copy_workers() -> None:
    config = _config()
    config["data"]["stage_copy_workers"] = 0

    with pytest.raises(ValueError, match="stage_copy_workers"):
        validate_config(config)
