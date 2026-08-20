"""6構成共通のnested fold学習CLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.staging import manifest_sha256, stage_dataset
from fracture_detection.common.augmentation import build_canonical_augmentation
from fracture_detection.common.canonical_dataset import CanonicalFractureDataset
from fracture_detection.common.constants import DATASET_DIR, INPUT_MANIFEST_CSV
from fracture_detection.common.sampling import (
    AnnotatedCycleSampler,
    EpochShuffleSampler,
)
from fracture_detection.common.splits import split_nested_manifest
from fracture_detection.config.schema import apply_overrides, load_config
from fracture_detection.core.artifacts import sha256_file, verify_frozen_manifest
from fracture_detection.core.contracts import LossWeights
from fracture_detection.core.experiment import resolve_fold_dir, save_effective_config
from fracture_detection.core.factory import build_adapter, build_model
from fracture_detection.core.parallel import launch_fold_processes
from fracture_detection.core.trainer import (
    create_data_loader,
    set_seed,
    train_fold,
    write_fold_summary,
)


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """multiprocessing一時ファイルをNFS外へ置く。"""
    path = base_dir / f"vai-fracture-{os.getuid()}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(path)
    tempfile.tempdir = str(path)
    return path


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="骨折検出6構成の共通nested学習")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="凍結前の構造検証として各epochのstep数を制限する",
    )
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """指定CUDAまたはCPUを返す。"""
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def run_training(
    config: dict[str, Any],
    *,
    resume: bool,
    smoke_steps: int | None,
) -> None:
    """設定範囲のouter foldを順に実行する。"""
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError("smoke_stepsは1以上が必要です")
    if smoke_steps is None:
        manifest_value = config["freeze"].get("manifest_path")
        weights_value = config["calibration"].get("loss_weights_path")
        if not isinstance(manifest_value, str) or not isinstance(weights_value, str):
            raise RuntimeError("正式学習にはfrozen manifestとloss weightsが必要です")
        manifest_path = Path(manifest_value)
        verify_frozen_manifest(config, manifest_path, Path(weights_value))
        config = copy.deepcopy(config)
        config["frozen_manifest_sha256"] = sha256_file(manifest_path)
    configure_local_temp_dir()
    manifest = load_manifest()
    source_dir = Path(config["data"].get("dataset_dir") or DATASET_DIR)
    dataset_dir = source_dir
    if config["data"]["stage_to_local"]:
        dataset_dir = stage_dataset(
            manifest,
            manifest_sha256(INPUT_MANIFEST_CSV),
            source_dir=source_dir,
            stage_root=Path(config["data"]["stage_root"]),
            copy_workers=int(config["data"]["stage_copy_workers"]),
        )
    start = int(config["data"]["start_outer_fold"])
    stop = int(config["data"]["end_outer_fold"])
    if "runtime" in config:
        start = stop = int(config["runtime"]["outer_fold"])
    for outer_fold in range(start, stop + 1):
        fold_config = apply_overrides(config, outer_fold=outer_fold)
        if smoke_steps is not None:
            fold_config = copy.deepcopy(fold_config)
            fold_config["experiment"]["name"] += "_smoke"
            fold_config["training"]["max_epochs"] = 1
            fold_config["training"]["early_stopping_patience"] = 1
        _run_fold(
            fold_config,
            manifest,
            dataset_dir,
            resume=resume,
            smoke_steps=smoke_steps,
        )


def _run_fold(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    dataset_dir: Path,
    *,
    resume: bool,
    smoke_steps: int | None,
) -> None:
    outer_fold = int(config["runtime"]["outer_fold"])
    device = resolve_device(int(config["training"]["gpu_id"]))
    set_seed(int(config["data"]["random_seed"]), outer_fold)
    train_manifest, validation_manifest, outer_manifest = split_nested_manifest(
        manifest, outer_fold
    )
    annotated_manifest = train_manifest[
        train_manifest["has_region_target"].astype(bool)
    ].reset_index(drop=True)
    if smoke_steps is not None:
        train_manifest = _balanced_smoke_manifest(train_manifest, 32)
        validation_manifest = _balanced_smoke_manifest(validation_manifest, 32)
        outer_manifest = _balanced_smoke_manifest(outer_manifest, 32)
        annotated_manifest = annotated_manifest.head(8).reset_index(drop=True)
    augmentation = build_canonical_augmentation(config["augmentation"])
    natural_dataset = CanonicalFractureDataset(
        train_manifest,
        dataset_dir,
        augmentation,
        base_seed=int(config["data"]["random_seed"]),
        outer_fold=outer_fold,
        stream="natural",
    )
    validation_dataset = CanonicalFractureDataset(validation_manifest, dataset_dir)
    outer_dataset = CanonicalFractureDataset(outer_manifest, dataset_dir)
    batch_size = int(config["training"]["natural_batch_size"])
    workers = int(config["data"]["num_workers"])
    if smoke_steps is not None:
        workers = 0
        batch_size = min(batch_size, len(natural_dataset))
    stream_seed = int(config["data"]["random_seed"]) + outer_fold
    natural_loader = create_data_loader(
        natural_dataset,
        batch_size,
        workers,
        stream_seed,
        device,
        EpochShuffleSampler(natural_dataset, seed=stream_seed, include_metadata=True),
    )
    annotated_loader = None
    if bool(config["arm"]["region_enabled"]):
        if annotated_manifest.empty:
            raise ValueError("train splitにannotated bagがありません")
        annotated_dataset = CanonicalFractureDataset(
            annotated_manifest,
            dataset_dir,
            build_canonical_augmentation(config["augmentation"]),
            base_seed=int(config["data"]["random_seed"]),
            outer_fold=outer_fold,
            stream="annotated",
        )
        annotated_loader = create_data_loader(
            annotated_dataset,
            int(config["training"]["annotated_batch_size"]),
            workers,
            stream_seed + 10_000,
            device,
            AnnotatedCycleSampler(
                len(annotated_dataset),
                samples_per_epoch=len(natural_loader),
                seed=stream_seed + 10_000,
                include_metadata=True,
            ),
        )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size,
        workers,
        stream_seed + 20_000,
        device,
    )
    outer_loader = create_data_loader(
        outer_dataset,
        batch_size,
        workers,
        stream_seed + 30_000,
        device,
    )
    fold_dir = resolve_fold_dir(config, outer_fold)
    completed = (
        fold_dir / "outer_predictions.csv",
        fold_dir / "outer_predictions_prauc_checkpoint.csv",
    )
    if resume and all(path.is_file() for path in completed):
        print(f"[outer {outer_fold}] outer推論済みのためskipします", flush=True)
        return
    if resume and any(path.is_file() for path in completed):
        raise RuntimeError("outer成果物が片方だけ存在するためresumeを拒否しました")
    if any(fold_dir.iterdir()) and not resume:
        raise FileExistsError(f"既存fold成果物があります: {fold_dir}")
    save_effective_config(config, fold_dir / "effective_config.yaml")
    weights = _resolve_loss_weights(config, outer_fold, smoke=smoke_steps is not None)
    result = train_fold(
        build_model(config),
        build_adapter(config),
        natural_loader,
        annotated_loader,
        validation_loader,
        outer_loader,
        config,
        outer_fold,
        fold_dir,
        device,
        weights,
        resume=resume,
        max_steps_per_epoch=smoke_steps,
        run_outer_inference=smoke_steps is None,
    )
    write_fold_summary(result, fold_dir / "summary.json")


def _resolve_loss_weights(
    config: dict[str, Any], outer_fold: int, *, smoke: bool
) -> LossWeights:
    """smoke仮値または凍結artifactからfold係数を返す。"""
    if smoke:
        return LossWeights(
            region=1.0 if config["arm"]["region_enabled"] else 0.0,
            attention=(0.0 if config["arm"]["beta_mode"] == "zero" else 1.0),
        )
    path_value = config["calibration"].get("loss_weights_path")
    if not isinstance(path_value, str):
        raise RuntimeError("正式学習には凍結loss_weights_pathが必要です")
    import json

    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    fold = payload["outer_folds"][str(outer_fold)]
    beta = 0.0 if config["arm"]["beta_mode"] == "zero" else float(fold["beta"])
    return LossWeights(
        region=(float(fold["lambda"]) if config["arm"]["region_enabled"] else 0.0),
        attention=beta,
    )


def _balanced_smoke_manifest(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    """whole陽性・陰性を含む小規模smoke subsetを返す。"""
    per_class = max(rows // 2, 1)
    parts = [
        frame[frame["vertebra_target"] == target].head(per_class) for target in (0, 1)
    ]
    result = pd.concat(parts, ignore_index=True)
    if result["vertebra_target"].nunique() != 2:
        raise ValueError("smoke subsetにwhole両classが必要です")
    return result


def run_cli(
    config_path: Path,
    *,
    outer_fold: int | None = None,
    gpu_id: int | None = None,
    resume: bool = False,
    smoke_steps: int | None = None,
) -> None:
    """config pathからfold並列起動または単一fold学習へ振り分ける。"""
    config = apply_overrides(
        load_config(config_path),
        outer_fold=outer_fold,
        gpu_id=gpu_id,
    )
    if config["parallel"]["mode"] == "fold" and outer_fold is None:
        launch_fold_processes(
            config_path,
            config,
            resume=resume,
            smoke_steps=smoke_steps,
        )
        return
    run_training(config, resume=resume, smoke_steps=smoke_steps)


def main() -> None:
    """CLI entry point。"""
    args = parse_args()
    run_cli(
        args.config,
        outer_fold=args.outer_fold,
        gpu_id=args.gpu_id,
        resume=args.resume,
        smoke_steps=args.smoke_steps,
    )


if __name__ == "__main__":
    main()
