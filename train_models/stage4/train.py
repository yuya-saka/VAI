"""Stage4 fixed-epoch multi-GPU cross-validation entry point."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_models.stage3.src.staging import (
    cleanup_stage,
    stage_dataset,
    sweep_stale_stages,
)
from train_models.stage4.src.data_utils import (
    collect_items,
    load_config,
    load_stage4_fold_map,
    save_effective_config,
    set_seed,
)
from train_models.stage4.src.experiment import (
    reject_unresumed_reuse,
    resolve_output_base,
    validate_resume_config,
)
from train_models.stage4.src.trainer import train_one_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-fold", type=int, default=None)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def apply_overrides(
    config: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Apply runtime fold and seed overrides without changing fixed epochs."""
    data = {**config.get("data", {})}
    model = {**config.get("model", {})}
    if arguments.start_fold is not None:
        data["start_fold"] = arguments.start_fold
    if arguments.end_fold is not None:
        data["end_fold"] = arguments.end_fold
    if arguments.seed is not None:
        data["random_seed"] = arguments.seed
        model["scramble_seed"] = arguments.seed
    return {**config, "data": data, "model": model}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("", 0))
        return int(handle.getsockname()[1])


def _configure_temp(base_dir: Path = Path("/tmp")) -> Path:
    temp_dir = base_dir / f"vai-stage4-{os.getuid()}"
    temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    return temp_dir


def _raise_on_sigterm(signum: int, frame: Any) -> None:
    del frame
    raise SystemExit(f"terminated by signal {signum}")


def _validate_gpu_config(config: dict[str, Any]) -> tuple[int, list[int]]:
    training = config.get("training", {})
    world_size = int(training.get("n_gpu", 1))
    gpu_ids = [int(value) for value in training.get("gpu_ids", [0])]
    if world_size < 1:
        raise ValueError("training.n_gpu must be at least 1")
    if len(gpu_ids) < world_size:
        raise ValueError("training.gpu_ids must contain training.n_gpu IDs")
    selected = gpu_ids[:world_size]
    if len(set(selected)) != len(selected):
        raise ValueError("training.gpu_ids must not contain duplicates")
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("multi-GPU training requires CUDA")
    if torch.cuda.is_available():
        invalid = [gpu_id for gpu_id in selected if gpu_id >= torch.cuda.device_count()]
        if invalid:
            raise ValueError(f"unavailable GPU IDs: {invalid}")
    return world_size, gpu_ids


def _resolve_fold_range(data: dict[str, Any]) -> tuple[int, int]:
    start_fold = int(data["start_fold"])
    end_fold = int(data["end_fold"])
    n_folds = int(data["n_folds"])
    if not 0 <= start_fold <= end_fold < n_folds:
        raise ValueError("invalid Stage4 fold range")
    return start_fold, end_fold


def _build_split_manifest(
    items: list[dict[str, Any]],
    fold_map: dict[str, int],
) -> pd.DataFrame:
    rows = [
        {
            "study_uid": str(item["study_uid"]),
            "vertebra": str(item["vertebra"]),
            "label": int(item["label"]),
            "region_supervision": str(item["region_supervision"]),
            "fold": fold_map[str(item["study_uid"])],
        }
        for item in items
    ]
    return (
        pd.DataFrame(rows).sort_values(["study_uid", "vertebra"]).reset_index(drop=True)
    )


def _validate_or_save_split_manifest(
    output_dir: Path,
    manifest: pd.DataFrame,
    resume: bool,
) -> None:
    path = output_dir / "split_manifest.csv"
    if resume:
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing, manifest, check_dtype=False)
        return
    manifest.to_csv(path, index=False)


def _run(
    local_rank: int,
    world_size: int,
    config: dict[str, Any],
    resume: bool,
    port: int,
    dataset_root: Path,
) -> None:
    if world_size > 1:
        gpu_ids = [
            int(value) for value in config.get("training", {}).get("gpu_ids", [0])
        ]
        torch.cuda.set_device(gpu_ids[local_rank])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{gpu_ids[local_rank]}"),
        )
    try:
        _run_training(local_rank, world_size, config, resume, dataset_root)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_training(
    local_rank: int,
    world_size: int,
    config: dict[str, Any],
    resume: bool,
    dataset_root: Path,
) -> None:
    training = config.get("training", {})
    gpu_ids = [int(value) for value in training.get("gpu_ids", [0])]
    gpu_id = (
        gpu_ids[local_rank]
        if world_size > 1
        else int(training.get("gpu_id", gpu_ids[0]))
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = config.get("data", {})
    set_seed(int(data.get("random_seed", 42)) + local_rank)
    excluded_studies = data.get("excluded_studies_path")
    excluded_levels = data.get("excluded_levels_path")
    items = collect_items(
        dataset_root,
        ROOT / str(data["csv_path"]),
        ROOT / str(data["region_labels_path"]),
        ROOT / str(excluded_studies) if excluded_studies else None,
        ROOT / str(excluded_levels) if excluded_levels else None,
    )
    fold_map = load_stage4_fold_map(ROOT / str(data["folds_path"]))
    output_dir = resolve_output_base(config, ROOT)
    if local_rank == 0:
        reject_unresumed_reuse(output_dir, resume)
        saved_config_path = output_dir / "config.yaml"
        if resume:
            validate_resume_config(
                load_config(saved_config_path),
                config,
                next_epoch=0,
            )
        else:
            save_effective_config(config, output_dir)
        manifest = _build_split_manifest(items, fold_map)
        _validate_or_save_split_manifest(output_dir, manifest, resume)
        print(
            f"[SPLIT] bags={len(items)} studies={manifest['study_uid'].nunique()}",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    start_fold, end_fold = _resolve_fold_range(data)
    for fold in range(start_fold, end_fold + 1):
        metrics, predictions = train_one_fold(
            config,
            fold,
            items,
            fold_map,
            ROOT,
            device,
            resume,
        )
        if local_rank == 0:
            fold_dir = output_dir / f"fold{fold}"
            pd.DataFrame([{**record, "fold": fold} for record in predictions]).to_csv(
                fold_dir / "oof_predictions.csv", index=False
            )
            with (fold_dir / "metrics.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metrics,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=True,
                )
    if local_rank != 0:
        return
    fold_paths = [
        output_dir / f"fold{fold}" / "oof_predictions.csv"
        for fold in range(int(data["n_folds"]))
    ]
    if all(path.exists() for path in fold_paths):
        pooled = pd.concat(
            [pd.read_csv(path) for path in fold_paths],
            ignore_index=True,
        )
        pooled.to_csv(output_dir / "oof_predictions.csv", index=False)


def main() -> None:
    _configure_temp()
    arguments = parse_args()
    config = apply_overrides(load_config(arguments.config), arguments)
    world_size, _ = _validate_gpu_config(config)
    data = config.get("data", {})
    dataset_root = ROOT / str(data["dataset_dir"])
    stage_dir: Path | None = None
    if bool(data.get("stage_to_local", False)):
        stage_root = Path(str(data.get("stage_root", "/dev/shm")))
        sweep_stale_stages(stage_root)
        stage_dir = stage_dataset(
            dataset_root,
            stage_root,
            max_workers=int(data.get("stage_workers", 32)),
        )
        dataset_root = stage_dir
        signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        if world_size > 1:
            mp.spawn(
                _run,
                args=(
                    world_size,
                    config,
                    arguments.resume,
                    _free_port(),
                    dataset_root,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            _run(
                0,
                1,
                config,
                arguments.resume,
                _free_port(),
                dataset_root,
            )
    finally:
        cleanup_stage(stage_dir)


if __name__ == "__main__":
    main()
