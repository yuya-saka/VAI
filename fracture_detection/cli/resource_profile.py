"""full training前の20-step resource profile CLI。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.staging import manifest_sha256, stage_dataset
from fracture_detection.common.constants import DATASET_DIR, INPUT_MANIFEST_CSV
from fracture_detection.config.schema import load_config
from fracture_detection.core.artifacts import write_new_json
from fracture_detection.profiling.runner import profile_arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="骨折検出armの20-step resource profile"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=10)
    return parser.parse_args()


def run_resource_profile(
    config_path: Path,
    *,
    output: Path,
    gpu_id: int = 0,
    outer_fold: int = 0,
    steps: int = 20,
    warmup_steps: int = 10,
) -> Path:
    """1 armのresource profileを測定してartifactを保存する。"""
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("resource profileにはCUDAが必要です")
    device = torch.device(f"cuda:{gpu_id}")
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
    payload = profile_arm(
        config,
        manifest,
        dataset_dir,
        device,
        outer_fold=outer_fold,
        steps=steps,
        warmup_steps=warmup_steps,
    )
    write_new_json(output, payload)
    print(f"resource profileを保存しました: {output}")
    return output


def main() -> None:
    args = parse_args()
    run_resource_profile(
        args.config,
        output=args.output,
        gpu_id=args.gpu_id,
        outer_fold=args.outer_fold,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
