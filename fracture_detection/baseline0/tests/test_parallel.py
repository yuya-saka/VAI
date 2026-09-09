from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fracture_detection.baseline0.training.parallel import (
    _build_worker_command,
    _write_execution_plan,
    build_fold_to_gpu,
)


def test_build_fold_to_gpu_assigns_round_robin() -> None:
    assert build_fold_to_gpu([0, 1, 2, 3, 4], [0, 1]) == {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
        4: 0,
    }


def test_build_fold_to_gpu_rejects_duplicate_gpu() -> None:
    with pytest.raises(ValueError, match="重複"):
        build_fold_to_gpu([0, 1], [0, 0])


def test_build_worker_command_targets_one_fold() -> None:
    command = _build_worker_command(
        Path("config.yaml"),
        "fracture_detection.baseline0.cli.train",
        outer_fold=3,
        gpu_id=1,
        resume=True,
    )

    assert command == [
        sys.executable,
        "-m",
        "fracture_detection.baseline0.cli.train",
        "--config",
        "config.yaml",
        "--outer-fold",
        "3",
        "--gpu-id",
        "1",
        "--resume",
    ]


def test_write_execution_plan_is_idempotent_and_rejects_remap(
    tmp_path: Path,
) -> None:
    signature = {
        "name": "GPU",
        "compute_capability": "8.6",
        "total_memory": [1, 1],
    }
    path = _write_execution_plan(tmp_path, {0: 0, 1: 1}, signature)
    repeated = _write_execution_plan(tmp_path, {0: 0, 1: 1}, signature)

    assert repeated == path
    assert json.loads(path.read_text(encoding="utf-8"))["fold_to_gpu"] == {
        "0": 0,
        "1": 1,
    }
    with pytest.raises(FileExistsError, match="異なるfold-to-GPU割当"):
        _write_execution_plan(tmp_path, {0: 1, 1: 0}, signature)
