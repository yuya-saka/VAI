"""Weights & Biases実験ログ管理テスト。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from Unet.line_2p5d.src.experiment import (
    finish_wandb,
    initialize_wandb,
    log_wandb_epoch,
    update_best_summary,
)


class FakeWandb:
    """外部通信せずwandb呼び出しを記録するfake。"""

    def __init__(self) -> None:
        self.run = SimpleNamespace(summary={})
        self.init_kwargs: dict[str, Any] = {}
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def init(self, **kwargs: Any) -> None:
        """初期化引数を記録する。"""
        self.init_kwargs = kwargs

    def log(self, values: dict[str, float], step: int) -> None:
        """epochログを記録する。"""
        self.logs.append((values, step))

    def finish(self) -> None:
        """終了呼び出しを記録する。"""
        self.finished = True


def _config() -> dict[str, Any]:
    """wandb有効configを返す。"""
    return {
        "experiment": {"phase": "line_2p5d_test", "name": "baseline"},
        "wandb": {"enabled": True, "project": None, "run_name": None},
    }


def test_initialize_wandb_derives_project_and_fold_name() -> None:
    """phase・name・foldから既定project/run名を生成する。"""
    fake = FakeWandb()

    enabled, module = initialize_wandb(_config(), fold=3, wandb_module=fake)

    assert enabled is True
    assert module is fake
    assert fake.init_kwargs["project"] == "unet-line_2p5d_test-baseline"
    assert fake.init_kwargs["name"] == "fold3"
    assert fake.init_kwargs["reinit"] is True


def test_epoch_best_and_test_metrics_are_logged() -> None:
    """有限scalarだけをepoch・best・test情報として記録する。"""
    fake = FakeWandb()
    log_wandb_epoch(
        fake,
        epoch=2,
        metrics={
            "epoch": 2,
            "train_loss": 0.2,
            "line_angle_error_deg": 3.0,
            "invalid": float("nan"),
            "per_line": {"line_1": {}},
        },
    )
    update_best_summary(
        fake,
        epoch=2,
        selection_metric="line_combined_error_px",
        selection_value=4.0,
        validation_metrics={"val_heatmap_mse": 0.1, "per_line": {}},
    )
    finish_wandb(
        fake,
        test_metrics={"line_angle_error_deg": 2.5, "per_line": {}},
        line_summary={"n_samples": 12, "line_extend_ratio": 1.0},
    )

    assert fake.logs == [({"train_loss": 0.2, "line_angle_error_deg": 3.0}, 2)]
    assert fake.run.summary["best_epoch"] == 2
    assert fake.run.summary["best_selection_metric"] == "line_combined_error_px"
    assert fake.run.summary["test_line_angle_error_deg"] == 2.5
    assert fake.run.summary["test_output_samples"] == 12
    assert fake.finished is True
