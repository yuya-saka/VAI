"""Generate region pseudo-label scores from fold-matched in-sample Baseline 0 teachers.

Per ``.claude/docs/codex/20260823-cam-audit-gate-interpretation.md``: memorization
audit found no in-sample inflation (<0.041 AUROC everywhere, all CIs cross zero),
so the student for outer fold ``k`` is labelled by ``Teacher_k`` (the Baseline 0
run whose outer fold is ``k``) on exactly ``Teacher_k``'s own training set. That
is precisely the population `Student_k` trains on, so pairs are always compared
within one teacher's score scale.

This writes only the raw region density and each fold's frozen temperature; it
does not build pairs, does not serialize the 268 human region labels, and trains
nothing. Temperatures are computed from fracture-positive bags only because CAM
ranking is a conditional localization signal. The logical-zero comparison arm
will add exact negative supervision separately rather than rank CAMs from
whole-negative bags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from fracture_detection.baseline0.data.constants import DATASET_DIR, REGION_COLUMNS
from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.splits import (
    resolve_nested_folds,
    split_nested_manifest,
)
from fracture_detection.baseline0.data.staging import manifest_sha256
from fracture_detection.baseline0.pseudo_labeling.cam_audit import (
    MaskPerturbation,
    region_density_enrichment,
)
from fracture_detection.baseline0.pseudo_labeling.gradcam import (
    compute_gradcam,
    load_bag_arrays,
    load_baseline0_checkpoint,
    prepare_inputs,
)
from fracture_detection.baseline0.pseudo_labeling.scoring import (
    DEFAULT_TEMPERATURE_PAIRS,
    DEFAULT_TEMPERATURE_SEED,
    region_temperature,
)

N_FOLDS = 5
IDENTITY = MaskPerturbation("identity", "identity")
DEFAULT_EXPERIMENT_DIR = Path(
    "fracture_detection/baseline0/outputs/08_19/baseline0_shared_core"
)
SCORES_CSV = "pseudo_label_scores.csv"
TEMPERATURES_CSV = "pseudo_label_temperatures.csv"
METADATA_JSON = "pseudo_label_generation_metadata.json"


def run_generation(args: argparse.Namespace) -> Path:
    """Score every training-fold bag with its fold-matched in-sample teacher."""
    experiment_dir = cast(Path, args.experiment_dir).resolve()
    dataset_dir = cast(Path, args.dataset_dir).resolve()
    output_dir = cast(Path, args.output_dir).resolve()
    _guard_output(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    device = _resolve_device(args.device)
    print(f"Pseudo-label generation: {N_FOLDS} teachers, device={device}", flush=True)

    all_rows: list[dict[str, Any]] = []
    teacher_assignments: list[dict[str, Any]] = []
    for outer_fold in range(N_FOLDS):
        assignment = resolve_nested_folds(outer_fold, N_FOLDS)
        train, _inner, _outer = split_nested_manifest(manifest, outer_fold, N_FOLDS)
        if args.limit_bags is not None:
            train = train.head(args.limit_bags)
        checkpoint = experiment_dir / f"outer{outer_fold}" / args.checkpoint_name
        checkpoint_hash = manifest_sha256(checkpoint)
        teacher_id = f"baseline0_outer{outer_fold}"
        teacher_assignments.append(
            {
                "student_outer_fold": outer_fold,
                "teacher_id": teacher_id,
                "teacher_train_folds": list(assignment.train_folds),
                "teacher_checkpoint": str(checkpoint),
                "teacher_checkpoint_sha256": checkpoint_hash,
            }
        )
        model = load_baseline0_checkpoint(checkpoint, device)
        records = train.to_dict("records")
        print(
            f"  outer{outer_fold} teacher scoring {len(records)} training bags",
            flush=True,
        )
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            all_rows.extend(
                _score_batch(
                    model,
                    device,
                    dataset_dir,
                    batch,
                    outer_fold,
                    teacher_id,
                    assignment.train_folds,
                    checkpoint_hash,
                )
            )
            done = min(start + args.batch_size, len(records))
            if done % 400 < args.batch_size or done == len(records):
                print(f"    {done}/{len(records)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(all_rows)
    scores_path = output_dir / SCORES_CSV
    scores.to_csv(scores_path, index=False)

    temperatures = _compute_temperatures(scores, args.temperature_pairs, args.seed)
    temperatures_path = output_dir / TEMPERATURES_CSV
    temperatures.to_csv(temperatures_path, index=False)

    metadata = {
        "n_rows": int(len(scores)),
        "n_bags": int(scores[["study_id", "level"]].drop_duplicates().shape[0]),
        "checkpoint_name": str(args.checkpoint_name),
        "experiment_dir": str(experiment_dir),
        "temperature_pairs": int(args.temperature_pairs),
        "temperature_seed": int(args.seed),
        "temperature_population": "fracture_positive",
        "scores_sha256": manifest_sha256(scores_path),
        "temperatures_sha256": manifest_sha256(temperatures_path),
        "teacher_assignments": teacher_assignments,
    }
    (output_dir / METADATA_JSON).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n== temperatures (label-free, per outer-fold teacher x region) ==")
    print(temperatures.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nwrote {len(scores)} rows for {metadata['n_bags']} bags to {output_dir}")
    return output_dir


def _score_batch(
    model: Any,
    device: torch.device,
    dataset_dir: Path,
    batch: list[dict[str, Any]],
    outer_fold: int,
    teacher_id: str,
    teacher_train_folds: tuple[int, ...],
    checkpoint_hash: str,
) -> list[dict[str, Any]]:
    loaded = [
        load_bag_arrays(dataset_dir, str(record["study_id"]), str(record["level"]))
        for record in batch
    ]
    inputs = torch.stack(
        [prepare_inputs(ct, whole_mask) for ct, whole_mask, _ in loaded]
    )
    result = compute_gradcam(model, inputs, device)

    rows: list[dict[str, Any]] = []
    for index, (record, arrays) in enumerate(zip(batch, loaded, strict=True)):
        _, whole_mask, region_mask = arrays
        cams = result.cams[index]
        scores = region_density_enrichment(cams, whole_mask, region_mask, IDENTITY)
        row: dict[str, Any] = {
            "study_id": str(record["study_id"]),
            "level": str(record["level"]),
            "fold": int(record["fold"]),
            "student_outer_fold": outer_fold,
            "teacher_outer_fold": outer_fold,
            "teacher_id": teacher_id,
            "teacher_train_folds": json.dumps(teacher_train_folds),
            "teacher_checkpoint_sha256": checkpoint_hash,
            "vertebra_target": int(record["vertebra_target"]),
            "bag_probability": float(result.bag_probabilities[index]),
        }
        for region_index, region_column in enumerate(REGION_COLUMNS):
            row[f"{region_column}_score"] = float(scores[region_index])
        rows.append(row)
    return rows


def _compute_temperatures(
    scores: pd.DataFrame, n_pairs: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold in sorted(scores["teacher_outer_fold"].unique()):
        subset = scores[
            (scores["teacher_outer_fold"] == outer_fold)
            & scores["vertebra_target"].eq(1)
        ]
        for region_column in REGION_COLUMNS:
            values = subset[f"{region_column}_score"].to_numpy(dtype=np.float64)
            defined = np.isfinite(values) & (values > 0)
            temperature = region_temperature(values, n_pairs=n_pairs, seed=seed)
            rows.append(
                {
                    "teacher_outer_fold": int(outer_fold),
                    "region": region_column,
                    "population": "fracture_positive",
                    "n_bags": int(len(values)),
                    "n_defined": int(defined.sum()),
                    "temperature": temperature,
                }
            )
    return pd.DataFrame(rows)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _guard_output(output_dir: Path, overwrite: bool) -> None:
    expected = (SCORES_CSV, TEMPERATURES_CSV, METADATA_JSON)
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Pseudo-label output already exists; pass --overwrite: {existing[0]}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="fold-matched in-sample teacherで疑似ラベル用CAMスコアを生成する"
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fracture_detection/baseline0/outputs/08_19/pseudo_labels"),
    )
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--temperature-pairs", type=int, default=DEFAULT_TEMPERATURE_PAIRS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TEMPERATURE_SEED)
    parser.add_argument("--limit-bags", type=int, help="smoke run上限（fold毎）")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run_generation(parse_args())


if __name__ == "__main__":
    main()
