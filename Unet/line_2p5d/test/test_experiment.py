"""phase・実験名による成果物path管理テスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from Unet.line_2p5d.src import trainer
from Unet.line_2p5d.src.data_utils import validate_config


def _minimum_config() -> dict:
    """設定検証に必要な最小configを返す。"""
    return {
        "experiment": {"phase": "line_2p5d_test", "name": "baseline"},
        "folds": {"start": 0, "end": 0},
        "data": {
            "annotation_root": "/annotation",
            "dense_root": "/dense",
            "context_offsets": [-2, -1, 0, 1, 2],
            "n_folds": 5,
        },
        "training": {"batch_size": 1, "num_workers": 0},
        "evaluation": {"metrics_frequency": 1},
    }


def test_validate_config_requires_experiment_phase() -> None:
    """phase欠落を実験管理設定の不備として拒否する。"""
    config = _minimum_config()
    del config["experiment"]["phase"]
    with pytest.raises(ValueError, match="experiment.phase"):
        validate_config(config)


def test_output_paths_use_phase_and_experiment_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成果物をUnet/outputs/phase/name以下へ配置する。"""
    fake_file = tmp_path / "Unet" / "line_2p5d" / "src" / "trainer.py"
    monkeypatch.setattr(trainer, "__file__", str(fake_file))

    paths = trainer._output_paths(_minimum_config(), fold=2)

    expected_root = tmp_path / "Unet" / "outputs" / "line_2p5d_test" / "baseline"
    assert paths["root"] == expected_root / "fold_2"
    assert paths["line_output"] == expected_root / "vis" / "fold2" / "test_lines"
