"""Stage2 anatomy-region MIL training entry point."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_models.stage2.src.data_utils import (  # noqa: E402
    collect_items,
    load_config,
    save_effective_config,
    set_seed,
    split_test_holdout,
)
from train_models.stage2.src.evaluation import compute_metrics  # noqa: E402
from train_models.stage2.src.experiment import (  # noqa: E402
    reject_unresumed_reuse,
    resolve_fold_paths,
    resolve_output_base,
)
from train_models.stage2.src.staging import (  # noqa: E402
    cleanup_stage,
    stage_dataset,
    sweep_stale_stages,
)
from train_models.stage2.src.trainer import (  # noqa: E402
    predict_ensemble,
    train_one_fold,
)


def parse_args() -> argparse.Namespace:
    """Parse training CLI options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-fold", type=int, default=None)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _primary_and_region_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Score a predictions frame on both the primary and region-MIL heads.

    Both probability columns are always present (region masks are always
    passed through the model), so ablation conditions that train only one
    head still report a fair side-by-side comparison against the other.
    """
    labels = frame["label"].to_numpy()
    study_uids = frame["study_uid"].to_numpy()
    vertebrae = frame["vertebra"].to_numpy()
    return {
        "primary": compute_metrics(
            labels, frame["pred_prob"].to_numpy(), study_uids, vertebrae
        ),
        "region": compute_metrics(
            labels, frame["region_pred_prob"].to_numpy(), study_uids, vertebrae
        ),
    }


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply supported CLI overrides without mutating the loaded dictionary."""
    updated = {
        **config,
        "data": {**config.get("data", {})},
        "training": {**config.get("training", {})},
    }
    if args.start_fold is not None:
        updated["data"]["start_fold"] = args.start_fold
    if args.end_fold is not None:
        updated["data"]["end_fold"] = args.end_fold
    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    return updated


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("", 0))
        return int(socket_handle.getsockname()[1])


def _configure_temp(base_dir: Path = Path("/tmp")) -> Path:
    temp_dir = base_dir / f"vai-stage2-{os.getuid()}"
    temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    return temp_dir


def _run(
    local_rank: int,
    world_size: int,
    config: dict[str, Any],
    resume: bool,
    port: int,
    dataset_root: Path,
) -> None:
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
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
    training_config = config.get("training", {})
    gpu_ids = list(training_config.get("gpu_ids", [0]))
    gpu_id = (
        int(gpu_ids[local_rank])
        if world_size > 1
        else int(training_config.get("gpu_id", 0))
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(int(config.get("data", {}).get("random_seed", 42)) + local_rank)

    data_config = config.get("data", {})
    excluded_studies_path = data_config.get("excluded_studies_path")
    excluded_levels_path = data_config.get("excluded_levels_path")
    collection_started = time.perf_counter()
    items = collect_items(
        dataset_root,
        ROOT / str(data_config["csv_path"]),
        excluded_studies_path=ROOT / excluded_studies_path
        if excluded_studies_path
        else None,
        excluded_levels_path=ROOT / excluded_levels_path
        if excluded_levels_path
        else None,
    )
    if local_rank == 0:
        print(
            f"[INFO] Stage2 item collection completed in "
            f"{time.perf_counter() - collection_started:.1f}s",
            flush=True,
        )
    train_val_items, test_items = split_test_holdout(
        items, seed=int(data_config.get("random_seed", 42))
    )
    start_fold = int(data_config.get("start_fold", 0))
    end_fold = int(data_config.get("end_fold", 4))
    output_dir = resolve_output_base(config, ROOT)
    reject_unresumed_reuse(output_dir, resume)
    if local_rank == 0 and not (resume and (output_dir / "config.yaml").exists()):
        save_effective_config(config, output_dir)

    all_oof: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(start_fold, end_fold + 1):
        metrics, predictions = train_one_fold(
            config, fold, train_val_items, ROOT, device, resume=resume
        )
        if local_rank == 0:
            fold_metrics.append(metrics)
            all_oof.extend({**record, "fold": fold} for record in predictions)

    if local_rank == 0:
        oof_frame = pd.DataFrame(all_oof)
        oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
        oof_metrics = _primary_and_region_metrics(oof_frame)
        model_paths = [
            resolve_fold_paths(config, fold, ROOT)[0]
            for fold in range(start_fold, end_fold + 1)
        ]
        test_predictions = predict_ensemble(config, test_items, model_paths, device)
        test_frame = pd.DataFrame(test_predictions)
        test_frame.to_csv(output_dir / "test_predictions.csv", index=False)
        test_metrics = _primary_and_region_metrics(test_frame)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "fold_metrics": fold_metrics,
                    "oof": oof_metrics,
                    "test": test_metrics,
                },
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
            )


def _raise_on_sigterm(signum: int, frame: Any) -> None:
    """Convert SIGTERM into a normal Python exception so `finally` blocks still run."""
    del frame
    raise SystemExit(f"terminated by signal {signum}")


def main() -> None:
    """Launch single-process or multi-GPU Stage2 training."""
    _configure_temp()
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    world_size = int(config.get("training", {}).get("n_gpu", 1))
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("multi-GPU training requires CUDA")

    data_config = config.get("data", {})
    dataset_root = ROOT / str(data_config["dataset_dir"])
    stage_dir: Path | None = None
    if bool(data_config.get("stage_to_local", False)):
        stage_root = Path(str(data_config.get("stage_root", "/dev/shm")))
        sweep_stale_stages(stage_root)
        stage_dir = stage_dataset(dataset_root, stage_root)
        dataset_root = stage_dir
        signal.signal(signal.SIGTERM, _raise_on_sigterm)

    port = _free_port()
    try:
        if world_size > 1:
            mp.spawn(
                _run,
                args=(world_size, config, args.resume, port, dataset_root),
                nprocs=world_size,
                join=True,
            )
        else:
            _run(0, 1, config, args.resume, port, dataset_root)
    finally:
        cleanup_stage(stage_dir)


if __name__ == "__main__":
    main()
