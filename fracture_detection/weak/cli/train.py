"""nested 5-fold training CLI for the weak-label region-MIL model.

Every outer fold's CNN trunk is always transferred from the fold-matched
Baseline 0 checkpoint (``fracture_detection/REGION_MIL_DESIGN.md`` section 7)
-- there is no ImageNet-only alternative here, unlike ``region_branch``'s
``joint_from_start`` option.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.constants import DATASET_DIR, INPUT_MANIFEST_CSV
from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.staging import manifest_sha256, stage_dataset
from fracture_detection.baseline0.training.parallel import launch_fold_processes
from fracture_detection.weak.config.schema import apply_cli_overrides, load_config
from fracture_detection.weak.data_pipeline.augmentation import augment_from_config
from fracture_detection.weak.data_pipeline.groups import build_group_inventory
from fracture_detection.weak.data_pipeline.loaders import build_outer_fold_loaders
from fracture_detection.weak.modeling.initialization import (
    load_baseline0_encoder,
    save_initialization_report,
)
from fracture_detection.weak.modeling.model import build_model
from fracture_detection.weak.training.experiment import (
    resolve_experiment_root,
    resolve_fold_dir,
    save_effective_config,
    save_fold_effective_config,
)
from fracture_detection.weak.training.trainer import set_seed, train_fold


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """Point multiprocessing's temp dir off NFS, matching baseline0's CLI."""
    local_temp_dir = base_dir / f"vai-weak-{os.getuid()}"
    local_temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(local_temp_dir)
    tempfile.tempdir = str(local_temp_dir)
    return local_temp_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="weak region-MIL nested 5-fold学習")
    parser.add_argument(
        "--config", type=Path, default=Path("fracture_detection/weak/config/weak.yaml")
    )
    parser.add_argument("--start-outer-fold", type=int, default=None)
    parser.add_argument("--end-outer-fold", type=int, default=None)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(gpu_id: int) -> torch.device:
    """Return CUDA if the requested GPU is available, else CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def run_training(config: dict[str, Any], resume: bool) -> None:
    """Train the configured outer-fold range under nested selection."""
    local_temp_dir = configure_local_temp_dir()
    print(f"multiprocessing一時領域: {local_temp_dir}", flush=True)
    data = config["data"]
    start_outer_fold = int(data["start_outer_fold"])
    end_outer_fold = int(data["end_outer_fold"])
    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        start_outer_fold = end_outer_fold = int(runtime["outer_fold"])
    print("凍結fullマニフェストを読み込んでいます", flush=True)
    manifest = load_manifest()
    print(f"マニフェストを読み込みました: {len(manifest):,} bag", flush=True)
    source_dir = Path(data.get("dataset_dir") or DATASET_DIR)
    dataset_dir = source_dir
    if data["stage_to_local"]:
        print(
            f"共有ローカルキャッシュを準備しています: {data['stage_root']}", flush=True
        )
        dataset_dir = stage_dataset(
            manifest,
            manifest_sha256(INPUT_MANIFEST_CSV),
            source_dir=source_dir,
            stage_root=Path(data["stage_root"]),
            copy_workers=int(data["stage_copy_workers"]),
        )
        print(f"共有ローカルキャッシュを使用します: {dataset_dir}", flush=True)

    device = resolve_device(int(config["training"]["gpu_id"]))
    print(
        f"学習対象outer fold: {start_outer_fold}〜{end_outer_fold}, device={device}",
        flush=True,
    )
    for outer_fold in range(start_outer_fold, end_outer_fold + 1):
        _train_one_fold(config, manifest, dataset_dir, outer_fold, device, resume)


def _train_one_fold(
    config: dict[str, Any],
    manifest: Any,
    dataset_dir: Path,
    outer_fold: int,
    device: torch.device,
    resume: bool,
) -> None:
    fold_config = apply_cli_overrides(config, outer_fold=outer_fold)
    data = fold_config["data"]
    sampling = fold_config["sampling"]
    runtime = fold_config["runtime"]
    inner_fold = int(runtime["inner_fold"])
    train_folds = tuple(int(value) for value in runtime["train_folds"])
    print(
        f"[outer {outer_fold}] train={train_folds}, val={inner_fold}を準備します",
        flush=True,
    )

    fold_dir = resolve_fold_dir(fold_config, outer_fold)
    if any(fold_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"既存のouter成果物があります: {fold_dir}。再開する場合は--resumeを指定してください"
        )
    outer_prediction_path = fold_dir / "outer_predictions.csv"
    if resume and outer_prediction_path.is_file():
        print(f"[outer {outer_fold}] outer推論済みのためskipします", flush=True)
        return
    save_fold_effective_config(fold_config, fold_dir)
    set_seed(int(data["random_seed"]), outer_fold)

    train_manifest = manifest[manifest["fold"].isin(train_folds)]
    inventory = build_group_inventory(train_manifest)
    print(
        f"[outer {outer_fold}] train inventory: "
        f"N={inventory.n_negative:,} A={inventory.n_annotated:,} U={inventory.n_weak:,} "
        f"annotated_cells={inventory.n_annotated_cells:,} "
        f"positive_cells={inventory.n_annotated_positive_cells:,}",
        flush=True,
    )

    stream_seed = int(data["random_seed"]) + outer_fold
    loaders = build_outer_fold_loaders(
        manifest,
        outer_fold=outer_fold,
        dataset_dir=dataset_dir,
        negative_per_batch=int(sampling["negative_bags_per_batch"]),
        annotated_per_batch=int(sampling["annotated_bags_per_batch"]),
        weak_per_batch=int(sampling["weak_bags_per_batch"]),
        num_workers=int(data["num_workers"]),
        seed=stream_seed,
        device=device,
        train_transform=augment_from_config(fold_config["augmentation"]),
        eval_batch_size=int(sampling["negative_bags_per_batch"])
        + int(sampling["annotated_bags_per_batch"])
        + int(sampling["weak_bags_per_batch"]),
    )
    print(
        f"[outer {outer_fold}] DataLoader作成完了: "
        f"train_steps_per_pass={len(loaders.train_sampler)}, "
        f"inner={len(loaders.inner_loader)}, outer={len(loaders.outer_loader)}",
        flush=True,
    )

    model = build_model(fold_config)
    checkpoint_root = Path(fold_config["model"]["baseline0_checkpoint_root"])
    checkpoint_path = checkpoint_root / f"outer{outer_fold}" / "best_model.pt"
    report = load_baseline0_encoder(model, checkpoint_path, dict(runtime))
    save_initialization_report(report, fold_dir / "initialization.json")
    print(
        f"[outer {outer_fold}] Baseline 0 encoderを転送しました: "
        f"loaded={report.loaded_key_count}, random={report.random_key_count}",
        flush=True,
    )

    result = train_fold(
        model, loaders, fold_config, outer_fold, fold_dir, device, resume=resume
    )
    print(
        f"outer={outer_fold} completed: best_gt_pass={result.best_gt_pass} "
        f"region_macro_ap={result.best_metrics['region_macro_ap']:.6f} "
        f"outer_rows={len(result.outer_predictions):,}"
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = apply_cli_overrides(
        load_config(args.config),
        outer_fold=args.outer_fold,
        gpu_id=args.gpu_id,
        start_outer_fold=args.start_outer_fold,
        end_outer_fold=args.end_outer_fold,
    )
    if (
        config["parallel"]["mode"] == "fold"
        and args.outer_fold is None
        and args.gpu_id is None
    ):
        config_path = save_effective_config(config)
        print(f"実効configを保存しました: {config_path}")
        launch_fold_processes(
            args.config,
            config,
            module_name="fracture_detection.weak.cli.train",
            experiment_root=resolve_experiment_root(config),
            resume=args.resume,
        )
        return
    if args.outer_fold is None:
        config_path = save_effective_config(config)
        print(f"実効configを保存しました: {config_path}")
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
