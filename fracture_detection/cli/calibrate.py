"""outer fold別λ/β校正CLI。"""
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
from fracture_detection.calibration.runner import (
    calibrate_outer_fold,
    calibration_record,
)
from fracture_detection.common.constants import DATASET_DIR, INPUT_MANIFEST_CSV
from fracture_detection.config.schema import load_config
from fracture_detection.core.artifacts import write_calibration_artifact

REFERENCE_ARMS = {"lambda": "baseline1_b", "beta": "proposed_b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="λ/βの5-fold gradient校正")
    parser.add_argument("--kind", choices=tuple(REFERENCE_ARMS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--expected-batches", type=int, default=64)
    return parser.parse_args()


def run_calibration(
    config_path: Path,
    *,
    kind: str,
    output: Path | None = None,
    gpu_id: int = 0,
    expected_batches: int = 64,
) -> Path:
    """5 fold分のλまたはβを校正してartifactを保存する。"""
    if kind not in REFERENCE_ARMS:
        raise ValueError(f"校正kindはlambdaまたはbetaが必要です: {kind}")
    config = load_config(config_path)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
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
    records: dict[int, dict[str, object]] = {}
    hashes: dict[int, str] = {}
    for outer_fold in range(5):
        result, config_hash = calibrate_outer_fold(
            config,
            manifest,
            dataset_dir,
            outer_fold,
            kind,
            device,
            expected_batches=expected_batches,
        )
        records[outer_fold] = calibration_record(result)
        hashes[outer_fold] = config_hash
    resolved_output = output or Path(config["calibration"][f"{kind}_artifact_path"])
    write_calibration_artifact(
        resolved_output,
        kind=kind,
        reference_arm=REFERENCE_ARMS[kind],
        outer_folds=records,
        reference_config_hashes=hashes,
    )
    print(f"校正artifactを保存しました: {resolved_output}")
    return resolved_output


def main() -> None:
    args = parse_args()
    run_calibration(
        args.config,
        kind=args.kind,
        output=args.output,
        gpu_id=args.gpu_id,
        expected_batches=args.expected_batches,
    )


if __name__ == "__main__":
    main()
