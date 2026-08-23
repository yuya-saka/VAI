"""mtl_type2（control_type2 / baseline1_type2）のnested fold学習CLI。

正式パイプラインの`cli/train.py`と違い、fold並列launcher
（`core.parallel`）は使わない。この探索projectはouter foldごとに
手動で1プロセス起動する運用を想定する（複数GPUで並行させたい場合は
`--gpu-id`を変えて別々に手動起動する）。
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from fracture_detection.core.contracts import LossWeights
from fracture_detection.core.steps import ArmAdapter
from fracture_detection.core.trainer import create_data_loader, set_seed
from fracture_detection.mtl_type2.config.schema import apply_overrides, load_config
from fracture_detection.mtl_type2.modeling.model import BranchedMtlModel
from fracture_detection.mtl_type2.training.experiment import (
    resolve_fold_dir,
    save_effective_config,
)
from fracture_detection.mtl_type2.training.trainer import train_fold, write_fold_summary


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """multiprocessing一時ファイルをNFS外へ置く。"""
    path = base_dir / f"vai-fracture-mtl-type2-{os.getuid()}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(path)
    tempfile.tempdir = str(path)
    return path


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="branch分離MTL（Type2型）の探索学習")
    parser.add_argument(
        "--arm", choices=["control_type2", "baseline1_type2"], required=True
    )
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--start-outer-fold", type=int, default=None)
    parser.add_argument("--end-outer-fold", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """指定CUDAまたはCPUを返す。"""
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def build_model(config: dict[str, Any]) -> BranchedMtlModel:
    """configからbranch分離MTLモデルを作る。"""
    model = config["model"]
    return BranchedMtlModel(
        backbone_name=str(model["backbone"]),
        in_chans=int(config["arm"]["input_channels"]),
        pretrained=bool(model["pretrained"]),
        drop_rate=float(model["drop_rate"]),
        drop_path_rate=float(model["drop_path_rate"]),
        head_dropout=float(model["head_dropout"]),
        lstm_hidden=int(model["lstm_hidden"]),
        lstm_layers=int(model["lstm_layers"]),
        n_planes=int(model["n_planes"]),
    )


def run_training(config: dict[str, Any], *, resume: bool) -> None:
    """設定範囲のouter foldを1プロセスで順に実行する。"""
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
        _run_fold(fold_config, manifest, dataset_dir, resume=resume)


def _run_fold(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    dataset_dir: Path,
    *,
    resume: bool,
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
    if annotated_manifest.empty:
        raise ValueError("train splitにannotated bagがありません")
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
    # augmentationなしで学習中のannotated精度を見る診断専用datasetと分ける
    # （train_stream付きのannotated_datasetはaugmentation込みで学習に使う）。
    annotated_eval_dataset = CanonicalFractureDataset(annotated_manifest, dataset_dir)
    batch_size = int(config["training"]["natural_batch_size"])
    workers = int(config["data"]["num_workers"])
    stream_seed = int(config["data"]["random_seed"]) + outer_fold
    natural_loader = create_data_loader(
        natural_dataset,
        batch_size,
        workers,
        stream_seed,
        device,
        EpochShuffleSampler(natural_dataset, seed=stream_seed, include_metadata=True),
    )
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
        # RSNA修正方針§4: 1 epoch = annotated datasetをちょうど1周させる
        # （毎step混入だった旧方式との違い。region step scheduleが
        # natural step列への配置を担う）。
        AnnotatedCycleSampler(
            len(annotated_dataset),
            samples_per_epoch=len(annotated_dataset),
            seed=stream_seed + 10_000,
            include_metadata=True,
        ),
    )
    annotated_train_eval_loader = create_data_loader(
        annotated_eval_dataset,
        batch_size,
        workers,
        stream_seed + 15_000,
        device,
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
    completed = fold_dir / "outer_predictions.csv"
    if resume and completed.is_file():
        print(f"[outer {outer_fold}] outer推論済みのためskipします", flush=True)
        return
    if any(fold_dir.iterdir()) and not resume:
        raise FileExistsError(f"既存fold成果物があります: {fold_dir}")
    save_effective_config(config, fold_dir / "effective_config.yaml")
    weights = LossWeights(region=float(config["training"]["region_lambda"]))
    adapter = ArmAdapter(
        input_channels=int(config["arm"]["input_channels"]),
        region_enabled=True,
        attention_enabled=False,
    )
    result = train_fold(
        build_model(config),
        adapter,
        natural_loader,
        annotated_loader,
        annotated_train_eval_loader,
        validation_loader,
        outer_loader,
        config,
        outer_fold,
        fold_dir,
        device,
        weights,
        resume=resume,
    )
    write_fold_summary(result, fold_dir / "summary.json")


def main() -> None:
    """CLI entry point。"""
    args = parse_args()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    config = apply_overrides(
        load_config(config_dir / f"{args.arm}.yaml"),
        outer_fold=args.outer_fold,
        gpu_id=args.gpu_id,
        start_outer_fold=args.start_outer_fold,
        end_outer_fold=args.end_outer_fold,
    )
    run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
