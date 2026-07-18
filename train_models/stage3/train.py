"""Stage3 multi-GPU cross-validation training entry point."""
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

from train_models.stage3.src.data_utils import (
    collect_items,
    load_config,
    save_effective_config,
    set_seed,
    split_items_cv,
    split_test_holdout,
)  # noqa: E402
from train_models.stage3.src.evaluation import (  # noqa: E402
    compute_evidence_diagnostics,
    compute_prediction_metrics,
    concatenate_evidence,
    save_evidence,
)
from train_models.stage3.src.experiment import (  # noqa: E402
    reject_unresumed_reuse,
    resolve_output_base,
    validate_resume_config,
)
from train_models.stage3.src.staging import (  # noqa: E402
    cleanup_stage,
    stage_dataset,
    sweep_stale_stages,
)
from train_models.stage3.src.trainer import (  # noqa: E402
    predict_ensemble,
    train_one_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-fold", type=int, default=None)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = {**config.get("data", {})}
    training = {**config.get("training", {})}
    if args.start_fold is not None:
        data["start_fold"] = args.start_fold
    if args.end_fold is not None:
        data["end_fold"] = args.end_fold
    if args.epochs is not None:
        training["epochs"] = args.epochs
    return {**config, "data": data, "training": training}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("", 0))
        return int(handle.getsockname()[1])


def _configure_temp(base_dir: Path = Path("/tmp")) -> Path:
    temp_dir = base_dir / f"vai-stage3-{os.getuid()}"
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
        raise ValueError("training.gpu_ids must contain at least training.n_gpu IDs")
    selected_ids = gpu_ids[:world_size]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("training.gpu_ids must not contain duplicates")
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("multi-GPU training requires CUDA")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        invalid_ids = [gpu_id for gpu_id in selected_ids if gpu_id >= device_count]
        if invalid_ids:
            raise ValueError(
                f"training.gpu_ids contains unavailable IDs {invalid_ids}; "
                f"visible CUDA device count is {device_count}"
            )
    return world_size, gpu_ids


def _resolve_fold_range(data: dict[str, Any]) -> tuple[int, int]:
    try:
        start_fold = int(data["start_fold"])
        end_fold = int(data["end_fold"])
        n_folds = int(data["n_folds"])
    except KeyError as error:
        raise ValueError(f"missing required data config: {error.args[0]}") from error
    if not 0 <= start_fold <= end_fold < n_folds:
        raise ValueError(
            "fold range must satisfy "
            f"0 <= start_fold <= end_fold < n_folds; got "
            f"start_fold={start_fold}, end_fold={end_fold}, n_folds={n_folds}"
        )
    return start_fold, end_fold


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item["study_uid"]), str(item["vertebra"])


def _build_split_manifest(
    items: list[dict[str, Any]], data: dict[str, Any]
) -> pd.DataFrame:
    train_val_items, test_items = split_test_holdout(
        items,
        test_size=float(data.get("test_size", 0.2)),
        seed=int(data.get("random_seed", 42)),
    )
    all_keys = {_item_key(item) for item in items}
    train_val_keys = {_item_key(item) for item in train_val_items}
    test_keys = {_item_key(item) for item in test_items}
    if len(all_keys) != len(items):
        raise ValueError("duplicate Stage3 study/vertebra items")
    train_studies = {study_uid for study_uid, _ in train_val_keys}
    test_studies = {study_uid for study_uid, _ in test_keys}
    if train_studies & test_studies:
        raise ValueError("Stage3 train_val/test study overlap")
    if train_val_keys & test_keys or train_val_keys | test_keys != all_keys:
        raise ValueError("Stage3 holdout partition is incomplete or overlapping")

    fold_by_key: dict[tuple[str, str], int] = {}
    n_folds = int(data["n_folds"])
    seed = int(data.get("random_seed", 42))
    for fold in range(n_folds):
        _, valid_items = split_items_cv(
            train_val_items, n_splits=n_folds, val_fold=fold, seed=seed
        )
        for item in valid_items:
            key = _item_key(item)
            if key in fold_by_key:
                raise ValueError(f"Stage3 item assigned to multiple folds: {key}")
            fold_by_key[key] = fold
    if set(fold_by_key) != train_val_keys:
        raise ValueError("Stage3 CV folds do not cover train_val exactly once")

    rows = [
        {
            "study_uid": str(item["study_uid"]),
            "vertebra": str(item["vertebra"]),
            "label": int(item["label"]),
            "partition": "test" if _item_key(item) in test_keys else "train_val",
            "fold": fold_by_key.get(_item_key(item)),
        }
        for item in items
    ]
    return (
        pd.DataFrame(rows).sort_values(["study_uid", "vertebra"]).reset_index(drop=True)
    )


def _validate_or_save_split_manifest(
    output_dir: Path, manifest: pd.DataFrame, resume: bool
) -> None:
    path = output_dir / "split_manifest.csv"
    if resume:
        if not path.exists():
            raise ValueError(
                "existing Stage3 output predates the fixed holdout protocol; "
                "retraining is required"
            )
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing, manifest, check_dtype=False)
        return
    manifest.to_csv(path, index=False)


def _save_fold_artifacts(
    output_dir: Path,
    fold: int,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> None:
    fold_dir = output_dir / f"fold{fold}"
    fold_predictions = [{**record, "fold": fold} for record in predictions]
    fold_evidence = [{**record, "fold": fold} for record in evidence]
    pd.DataFrame(fold_predictions).to_csv(fold_dir / "oof_predictions.csv", index=False)
    save_evidence(fold_dir / "oof_evidence.npz", fold_evidence)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=True)


def _load_completed_fold_artifacts(
    output_dir: Path, n_folds: int
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    completed_folds: list[int] = []
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    for fold in range(n_folds):
        fold_dir = output_dir / f"fold{fold}"
        metrics_path = fold_dir / "metrics.json"
        predictions_path = fold_dir / "oof_predictions.csv"
        evidence_path = fold_dir / "oof_evidence.npz"
        best_path = fold_dir / "best_model.pt"
        if not all(
            path.exists()
            for path in (metrics_path, predictions_path, evidence_path, best_path)
        ):
            continue
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        completed_folds.append(fold)
        fold_metrics.append({"fold": fold, **metrics})
        predictions.extend(pd.read_csv(predictions_path).to_dict(orient="records"))
        evidence_paths.append(evidence_path)
    return completed_folds, fold_metrics, predictions, evidence_paths


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
        ROOT / str(excluded_studies) if excluded_studies else None,
        ROOT / str(excluded_levels) if excluded_levels else None,
    )
    train_val_items, test_items = split_test_holdout(
        items,
        test_size=float(data.get("test_size", 0.2)),
        seed=int(data.get("random_seed", 42)),
    )
    output_dir = resolve_output_base(config, ROOT)
    if local_rank == 0:
        reject_unresumed_reuse(output_dir, resume)
        saved_config_path = output_dir / "config.yaml"
        if resume:
            if not saved_config_path.exists():
                raise ValueError(f"resume config does not exist: {saved_config_path}")
            validate_resume_config(load_config(saved_config_path), config, next_epoch=0)
        if not (resume and (output_dir / "config.yaml").exists()):
            save_effective_config(config, output_dir)
        manifest = _build_split_manifest(items, data)
        _validate_or_save_split_manifest(output_dir, manifest, resume)
        print(
            f"[SPLIT] train_val={len(train_val_items)} test={len(test_items)} "
            f"studies={manifest['study_uid'].nunique()}",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
    start_fold, end_fold = _resolve_fold_range(data)
    for fold in range(start_fold, end_fold + 1):
        metrics, predictions, evidence = train_one_fold(
            config, fold, train_val_items, ROOT, device, resume
        )
        if local_rank == 0:
            _save_fold_artifacts(output_dir, fold, metrics, predictions, evidence)
    if local_rank != 0:
        return
    completed_folds, fold_metrics, all_predictions, evidence_paths = (
        _load_completed_fold_artifacts(output_dir, int(data["n_folds"]))
    )
    if not completed_folds:
        raise RuntimeError("no completed Stage3 folds were found")
    pd.DataFrame(all_predictions).to_csv(
        output_dir / "oof_predictions.csv", index=False
    )
    oof_evidence_path = output_dir / "oof_evidence.npz"
    concatenate_evidence(oof_evidence_path, evidence_paths)
    model_paths = [
        (fold, resolve_output_base(config, ROOT) / f"fold{fold}" / "best_model.pt")
        for fold in completed_folds
    ]
    class_prior = sum(int(item["label"]) for item in train_val_items) / max(
        len(train_val_items), 1
    )
    test_predictions, test_predictions_per_fold, test_evidence = predict_ensemble(
        config, test_items, model_paths, device, class_prior
    )
    pd.DataFrame(test_predictions).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame(test_predictions_per_fold).to_csv(
        output_dir / "test_predictions_per_fold.csv", index=False
    )
    test_evidence_path = output_dir / "test_evidence_per_fold.npz"
    save_evidence(test_evidence_path, test_evidence)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "fold_metrics": fold_metrics,
                "oof": compute_prediction_metrics(all_predictions),
                "oof_evidence_diagnostics": compute_evidence_diagnostics(
                    oof_evidence_path
                ),
                "test": compute_prediction_metrics(test_predictions),
                "test_evidence_diagnostics_per_fold": compute_evidence_diagnostics(
                    test_evidence_path
                ),
                "n_train_val": len(train_val_items),
                "n_test": len(test_items),
                "trained_folds": completed_folds,
            },
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )


def main() -> None:
    _configure_temp()
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
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
                args=(world_size, config, args.resume, _free_port(), dataset_root),
                nprocs=world_size,
                join=True,
            )
        else:
            _run(0, 1, config, args.resume, _free_port(), dataset_root)
    finally:
        cleanup_stage(stage_dir)


if __name__ == "__main__":
    main()
