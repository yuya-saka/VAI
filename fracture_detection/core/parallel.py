"""1 foldを1 GPU processへ固定する並列launcher。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fracture_detection.core.experiment import resolve_experiment_root


@dataclass(frozen=True)
class FoldProcessResult:
    """子processの終了状態。"""

    outer_fold: int
    gpu_id: int
    return_code: int


def build_fold_to_gpu(folds: list[int], gpu_ids: list[int]) -> dict[int, int]:
    """outer昇順をGPUへround-robin固定割当する。"""
    if not folds or not gpu_ids:
        raise ValueError("foldsとgpu_idsは非空が必要です")
    if len(set(folds)) != len(folds) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("fold/gpu idに重複は許可されません")
    return {
        outer_fold: gpu_ids[index % len(gpu_ids)]
        for index, outer_fold in enumerate(sorted(folds))
    }


def validate_homogeneous_gpus(gpu_ids: list[int]) -> dict[str, object]:
    """正式比較用GPUのmodelとcompute capability一致を検証する。"""
    if not torch.cuda.is_available():
        raise RuntimeError("fold並列にはCUDAが必要です")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError("存在しないGPU idが指定されました")
    properties = [torch.cuda.get_device_properties(gpu_id) for gpu_id in gpu_ids]
    signatures = {
        (value.name, int(value.major), int(value.minor)) for value in properties
    }
    if len(signatures) != 1:
        raise RuntimeError(f"異種GPU混在を拒否しました: {sorted(signatures)}")
    name, major, minor = next(iter(signatures))
    return {
        "name": name,
        "compute_capability": f"{major}.{minor}",
        "total_memory": [int(value.total_memory) for value in properties],
    }


def launch_fold_processes(
    config_path: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    smoke_steps: int | None,
) -> list[FoldProcessResult]:
    """固定割当で子processを最大指定数まで並列実行する。"""
    parallel = config["parallel"]
    gpu_ids = [int(value) for value in parallel["gpu_ids"]]
    gpu_signature = validate_homogeneous_gpus(gpu_ids)
    folds = list(
        range(
            int(config["data"]["start_outer_fold"]),
            int(config["data"]["end_outer_fold"]) + 1,
        )
    )
    mapping = build_fold_to_gpu(folds, gpu_ids)
    plan_config = config
    if smoke_steps is not None:
        import copy

        plan_config = copy.deepcopy(config)
        plan_config["experiment"]["name"] += "_smoke"
    _write_execution_plan(plan_config, mapping, gpu_signature)
    concurrency = int(parallel["max_concurrent_folds"])
    pending = list(sorted(mapping))
    running: dict[int, tuple[int, subprocess.Popen[bytes]]] = {}
    results: list[FoldProcessResult] = []
    while pending or running:
        used_gpus = {gpu_id for gpu_id, _ in running.values()}
        while pending and len(running) < concurrency:
            candidate_index = next(
                (
                    index
                    for index, outer_fold in enumerate(pending)
                    if mapping[outer_fold] not in used_gpus
                ),
                None,
            )
            if candidate_index is None:
                break
            outer_fold = pending.pop(candidate_index)
            gpu_id = mapping[outer_fold]
            command = [
                sys.executable,
                "-m",
                "fracture_detection.cli.train",
                "--config",
                str(config_path),
                "--outer-fold",
                str(outer_fold),
                "--gpu-id",
                str(gpu_id),
            ]
            if resume:
                command.append("--resume")
            if smoke_steps is not None:
                command.extend(["--smoke-steps", str(smoke_steps)])
            running[outer_fold] = (gpu_id, subprocess.Popen(command))
            used_gpus.add(gpu_id)
        finished: list[int] = []
        for outer_fold, (gpu_id, process) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            results.append(FoldProcessResult(outer_fold, gpu_id, return_code))
            finished.append(outer_fold)
        for outer_fold in finished:
            del running[outer_fold]
        if running and not finished:
            time.sleep(0.2)
    failures = [result for result in results if result.return_code != 0]
    if failures:
        summary = [(value.outer_fold, value.return_code) for value in failures]
        raise RuntimeError(f"fold processが失敗しました: {summary}")
    return sorted(results, key=lambda value: value.outer_fold)


def _write_execution_plan(
    config: dict[str, Any],
    mapping: dict[int, int],
    gpu_signature: dict[str, object],
) -> Path:
    """異なる割当での上書きを拒否して実行計画を保存する。"""
    path = resolve_experiment_root(config) / "fold_execution_plan.json"
    payload = {
        "mode": "fold",
        "fold_to_gpu": {str(key): value for key, value in mapping.items()},
        "gpu_signature": gpu_signature,
    }
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なるfold-to-GPU割当が既に存在します: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return path
