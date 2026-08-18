from __future__ import annotations

from pathlib import Path
from typing import Any

from fracture_detection.baseline0.training import experiment


def _config() -> dict[str, Any]:
    return {
        "experiment": {"phase": "baseline0", "name": "nested"},
        "wandb": {"enabled": False, "project": None, "run_name": None},
    }


def test_experiment_paths_and_effective_config_are_isolated(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(experiment, "BASELINE0_DIR", tmp_path)
    config = _config()

    fold_dir = experiment.resolve_fold_dir(config, 2)
    config_path = experiment.save_effective_config(config)
    fold_config_path = experiment.save_fold_effective_config(config, fold_dir)

    assert fold_dir == tmp_path / "outputs" / "baseline0" / "nested" / "outer2"
    assert "nested" in config_path.read_text(encoding="utf-8")
    assert fold_config_path.is_file()
    assert experiment.initialize_wandb(config, 2) is None


def test_log_wandb_epoch_uses_val_metric_names() -> None:
    class FakeWandb:
        def __init__(self) -> None:
            self.logged: dict[str, float | int] = {}

        def log(self, values: dict[str, float | int], step: int) -> None:
            assert step == 1
            self.logged = values

    wandb_module = FakeWandb()
    train_metrics = {
        "loss": 0.4,
        "grad_norm": 1.0,
        "clip_fraction": 0.0,
        "mixup_fraction": 0.2,
        "data_wait_seconds": 10.0,
        "compute_seconds": 20.0,
    }
    validation_metrics = {
        "loss": 0.5,
        "auroc": 0.8,
        "average_precision": 0.6,
        "negative_score_mean": 0.2,
        "positive_score_mean": 0.7,
        "score_gap": 0.5,
        "precision_at_0_5": 0.7,
        "recall_at_0_5": 0.8,
        "f1_at_0_5": 0.75,
        "f1_optimal_threshold": 0.4,
        "precision_at_f1_optimal": 0.6,
        "recall_at_f1_optimal": 0.9,
        "f1_optimal": 0.72,
    }

    experiment.log_wandb_epoch(
        wandb_module,
        epoch=1,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        backbone_lr=1e-4,
        head_lr=1e-4,
        elapsed_seconds=30.0,
        early_stopping_best_loss=0.5,
        early_stopping_bad_epochs=0,
    )

    assert wandb_module.logged["val_bce"] == 0.5
    assert wandb_module.logged["val_auroc"] == 0.8
    assert wandb_module.logged["val_prauc"] == 0.6
    assert wandb_module.logged["val_precision_at_0_5"] == 0.7
    assert wandb_module.logged["val_recall_at_0_5"] == 0.8
    assert wandb_module.logged["val_f1_at_0_5"] == 0.75
    assert wandb_module.logged["val_f1_optimal_threshold"] == 0.4
    assert wandb_module.logged["val_f1_optimal"] == 0.72
    assert wandb_module.logged["train_mixup_fraction"] == 0.2
    assert not any(key.startswith("inner_") for key in wandb_module.logged)
