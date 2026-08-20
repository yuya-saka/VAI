"""凍結済み6構成のpooled OOF解析CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.common.metrics import evaluate_prediction_frame
from fracture_detection.core.artifacts import sha256_file, write_new_json
from fracture_detection.evaluation.analysis import (
    collect_oof_predictions,
    fixed_sequence_whole_tests,
    region_ap_sensitivity,
    region_floor_gate,
    region_pair_differences,
)

ARMS = (
    "baseline0",
    "control_b",
    "baseline1_b",
    "proposed_b",
    "proposed_max",
    "proposed_max_beta0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="骨折検出6構成の正式pooled OOF解析")
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-root", type=Path, required=True)
    parser.add_argument("--floor-predictions", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_hash = sha256_file(args.frozen_manifest)
    output_paths = [
        args.output,
        *[args.output.with_name(f"{args.output.stem}-{arm}-oof.csv") for arm in ARMS],
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"解析成果物は上書きできません: {existing}")
    manifest = load_manifest()
    predictions = {
        arm: collect_oof_predictions(
            getattr(args, f"{arm}_root"),
            manifest,
            expected_frozen_manifest_sha256=manifest_hash,
        )
        for arm in ARMS
    }
    floor = pd.read_csv(args.floor_predictions, dtype={"study_id": str, "level": str})
    metrics = {
        arm: (
            evaluate_prediction_frame(frame, n_bootstrap=args.n_bootstrap)
            if arm != "baseline0"
            else {
                "vertebra_auroc": safe_vertebra(frame, "auroc"),
                "vertebra_ap": safe_vertebra(frame, "ap"),
            }
        )
        for arm, frame in predictions.items()
    }
    payload = {
        "protocol_version": "fracture-oof-analysis-v1",
        "frozen_manifest_sha256": manifest_hash,
        "metrics": metrics,
        "fixed_sequence": fixed_sequence_whole_tests(
            predictions, n_bootstrap=args.n_bootstrap
        ),
        "region_floor_gate": region_floor_gate(
            predictions["proposed_b"], floor, n_bootstrap=args.n_bootstrap
        ),
        "control_vs_baseline1_region_ap": region_pair_differences(
            predictions["baseline1_b"],
            predictions["control_b"],
            n_bootstrap=args.n_bootstrap,
        ),
        "proposed_b_sensitivity": region_ap_sensitivity(
            predictions["proposed_b"], n_bootstrap=args.n_bootstrap
        ),
    }
    write_new_json(args.output, payload)
    for arm, frame in predictions.items():
        frame.to_csv(
            args.output.with_name(f"{args.output.stem}-{arm}-oof.csv"), index=False
        )
    print(json.dumps(payload["fixed_sequence"], ensure_ascii=False, indent=2))


def safe_vertebra(frame: pd.DataFrame, kind: str) -> float:
    from fracture_detection.common.metrics import safe_auroc, safe_average_precision

    metric = safe_auroc if kind == "auroc" else safe_average_precision
    return metric(
        frame["vertebra_target"].to_numpy(),
        frame["vertebra_score"].to_numpy(),
    )


if __name__ == "__main__":
    main()
