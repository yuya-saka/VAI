from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

import fracture_detection.baseline0.training.trainer as trainer_module
from fracture_detection.baseline0.training.trainer import (
    _batch_tensors,
    _mixup_batch,
    set_seed,
    train_fold,
)
from fracture_detection.common.sampling import EpochShuffleSampler


class _TinyDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, fold: int) -> None:
        self.fold = fold
        self.targets = [0.0, 1.0, 0.0, 1.0]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        target = self.targets[index]
        inputs = torch.full((15, 6, 4, 4), target)
        return {
            "inputs": inputs,
            "vertebra_target": torch.tensor(target),
            "fold": torch.tensor(self.fold),
            "study_id": f"study-{self.fold}-{index}",
            "level": "C1",
        }


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs[:, :, 0, 0, 0].unsqueeze(-1)
        return self.head(self.encoder(values)).squeeze(-1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable


def _config() -> dict[str, Any]:
    return {
        "protocol_version": "baseline0-nested-v7",
        "experiment": {"phase": "baseline0", "name": "trainer"},
        "runtime": {"outer_fold": 0, "inner_fold": 1, "train_folds": [2, 3, 4]},
        "training": {
            "max_epochs": 2,
            "min_epoch": 1,
            "early_stopping_patience": 2,
            "early_stopping_metric": "val_bce",
            "pos_weight": 2.0,
            "mixup_probability": 0.2,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 5.0,
            "freeze_backbone_epochs": 0,
            "warmup_epochs": 0,
            "warmup_start_factor": 1.0,
            "backbone_learning_rate": 2.3e-4,
            "head_learning_rate": 2.3e-4,
            "backbone_min_learning_rate": 2.3e-5,
            "head_min_learning_rate": 2.3e-5,
            "lr_scheduler": "cosine_annealing",
        },
        "wandb": {"enabled": False},
    }


def _loaders() -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_dataset = _TinyDataset(fold=2)
    train = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=EpochShuffleSampler(train_dataset, seed=9),
    )
    inner = DataLoader(_TinyDataset(fold=1), batch_size=2, shuffle=False)
    outer = DataLoader(_TinyDataset(fold=0), batch_size=2, shuffle=False)
    return train, inner, outer


def test_mixup_batch_uses_shared_permutation_and_uniform_lambda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.arange(4, dtype=torch.float32).view(4, 1)
    targets = torch.arange(4, dtype=torch.float32)
    monkeypatch.setattr(trainer_module.np.random, "uniform", lambda _low, _high: 0.25)

    mixed, targets_a, targets_b, mixup_lambda = _mixup_batch(inputs, targets)

    assert mixup_lambda == 0.25
    assert torch.equal(targets_a, targets)
    assert torch.allclose(
        mixed.squeeze(1), mixup_lambda * targets_a + (1.0 - mixup_lambda) * targets_b
    )


def test_set_seed_enables_fixed_shape_cudnn_autotuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)

    set_seed(123, 0)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_batch_tensors_normalizes_uint8_after_device_transfer() -> None:
    inputs = torch.tensor([[[[[0, 255]]]]], dtype=torch.uint8)
    targets = torch.tensor([1.0])

    normalized, moved_targets = _batch_tensors(
        {"inputs": inputs, "vertebra_target": targets}, torch.device("cpu")
    )

    assert normalized.dtype == torch.float32
    assert normalized.flatten().tolist() == [0.0, 1.0]
    assert torch.equal(moved_targets, targets)


def test_train_fold_selects_on_inner_then_writes_outer_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_loader, inner_loader, outer_loader = _loaders()

    result = train_fold(
        _TinyModel(),
        train_loader,
        inner_loader,
        outer_loader,
        _config(),
        outer_fold=0,
        fold_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert result.best_epoch >= 1
    assert result.best_prauc_epoch >= 1
    assert len(result.outer_predictions) == 4
    assert len(result.outer_prauc_predictions) == 4
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "best_val_prauc_model.pt").is_file()
    assert (tmp_path / "last_checkpoint.pt").is_file()
    assert len(pd.read_csv(tmp_path / "history.csv")) == 2
    assert (tmp_path / "val_predictions.csv").is_file()
    assert set(pd.read_csv(tmp_path / "outer_predictions.csv")) == {
        "study_id",
        "level",
        "fold",
        "vertebra_target",
        "vertebra_score",
        "decision_threshold",
        "vertebra_prediction",
    }
    assert (tmp_path / "val_predictions_prauc_checkpoint.csv").is_file()
    assert (tmp_path / "outer_predictions_prauc_checkpoint.csv").is_file()
    fold_metrics = json.loads((tmp_path / "fold_metrics.json").read_text())
    assert fold_metrics["primary_checkpoint"] == "best_val_auroc"
    assert "f1_optimal" not in fold_metrics["auroc_checkpoint"]["outer"]
    assert fold_metrics["auroc_checkpoint"]["outer"]["threshold"] == pytest.approx(
        pd.read_csv(tmp_path / "outer_predictions.csv")["decision_threshold"].iloc[0]
    )
    history = pd.read_csv(tmp_path / "history.csv")
    assert "val_precision_at_0_5" in history
    assert "val_recall_at_0_5" in history
    assert "val_f1_at_0_5" in history
    assert "val_f1_optimal_threshold" in history
    assert "val_f1_optimal" in history
    captured = capsys.readouterr()
    assert "[outer 0] 学習を初期化しています" in captured.out
    assert "val=4 bag" in captured.out
    assert "inner=" not in captured.out
    assert "[outer 0] outerを1回だけ推論しています" in captured.out


def test_train_fold_rejects_resume_with_different_config(tmp_path: Path) -> None:
    train_loader, inner_loader, outer_loader = _loaders()
    config = _config()
    train_fold(
        _TinyModel(),
        train_loader,
        inner_loader,
        outer_loader,
        config,
        outer_fold=0,
        fold_dir=tmp_path,
        device=torch.device("cpu"),
    )
    changed = _config()
    changed["training"]["max_epochs"] = 3

    with pytest.raises(ValueError, match="実効config"):
        train_fold(
            _TinyModel(),
            *_loaders(),
            changed,
            outer_fold=0,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
            resume=True,
        )


def test_train_fold_stops_on_inner_bce_while_auroc_improves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["training"]["max_epochs"] = 4
    metric_sequence = iter(
        [
            (0.50, 0.60, 0.40),
            (0.60, 0.70, 0.90),
            (0.70, 0.80, 0.80),
            (0.70, 0.80, 0.80),
            (0.70, 0.80, 0.80),
            (0.70, 0.80, 0.80),
            (0.70, 0.80, 0.80),
        ]
    )

    def fake_evaluate(
        model: nn.Module,
        loader: DataLoader[Any],
        device: torch.device,
        pos_weight: float,
        progress_description: str,
        include_f1_optimal: bool = True,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        del model, loader, device, pos_weight, progress_description
        loss, auroc, prauc = next(metric_sequence)
        metrics = {
            "loss": loss,
            "auroc": auroc,
            "average_precision": prauc,
            "negative_score_mean": 0.2,
            "positive_score_mean": 0.8,
            "score_gap": 0.6,
            "precision_at_0_5": 1.0,
            "recall_at_0_5": 1.0,
            "f1_at_0_5": 1.0,
            "f1_optimal_threshold": 0.8,
            "precision_at_f1_optimal": 1.0,
            "recall_at_f1_optimal": 1.0,
            "f1_optimal": 1.0,
        }
        if not include_f1_optimal:
            metrics = {
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "f1_optimal_threshold",
                    "precision_at_f1_optimal",
                    "recall_at_f1_optimal",
                    "f1_optimal",
                }
            }
        predictions = pd.DataFrame(
            {
                "study_id": ["negative", "positive"],
                "level": ["C1", "C1"],
                "fold": [0, 0],
                "vertebra_target": [0, 1],
                "vertebra_score": [0.2, 0.8],
            }
        )
        return metrics, predictions

    monkeypatch.setattr(trainer_module, "evaluate", fake_evaluate)

    result = train_fold(
        _TinyModel(),
        *_loaders(),
        config,
        outer_fold=0,
        fold_dir=tmp_path,
        device=torch.device("cpu"),
    )

    history = pd.read_csv(tmp_path / "history.csv")
    assert result.stopped_epoch == 3
    assert result.best_epoch == 3
    assert result.best_prauc_epoch == 2
    assert len(history) == 3
    assert "val_bce" in history
    assert "inner_bce" not in history
    assert history["is_best_val_auroc"].tolist() == [True, True, True]
    assert history["is_best_val_prauc"].tolist() == [True, True, False]
    assert history["early_stopping_bad_epochs"].tolist() == [0, 1, 2]
    assert history["early_stopping_improved"].tolist() == [True, False, False]
    prauc_checkpoint = torch.load(
        tmp_path / "best_val_prauc_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert prauc_checkpoint["epoch"] == 2
    assert prauc_checkpoint["best_prauc_epoch"] == 2
    assert prauc_checkpoint["best_prauc_metrics"]["average_precision"] == 0.9


def test_seed_worker_limits_opencv_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_threads: list[int] = []
    monkeypatch.setattr(
        trainer_module.cv2,
        "setNumThreads",
        configured_threads.append,
    )

    trainer_module.seed_worker(worker_id=0)

    assert configured_threads == [1]
