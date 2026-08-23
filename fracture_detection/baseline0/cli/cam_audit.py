"""Generation-stage audit of the Baseline 0 Grad-CAM region teacher signal.

Scores every annotated bag with all five Baseline 0 checkpoints so that the same
bag is seen once as in-sample (``train``), once as checkpoint-selection
(``inner``) and once as fully held out (``outer``). Each CAM is aggregated under
the frozen four-region mask perturbation grid and under a laterality-safe
horizontal flip, which yields the teacher-memorization and mask-sensitivity
tables the pre-registered kill criteria are read from.

No training happens here and no pseudo-label is written; this only decides
whether pseudo-label generation is worth doing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from fracture_detection.baseline0.analysis.cam_audit import (
    MaskPerturbation,
    default_perturbations,
    flip_planes_horizontally,
    gate_perturbation_names,
    region_density_enrichment,
    teacher_role,
)
from fracture_detection.baseline0.analysis.cam_audit_report import (
    HFLIP_TTA,
    NO_TTA,
    audit_verdict,
    laterality_summary,
    localization_table,
    memorization_table,
    perturbation_table,
    tta_table,
)
from fracture_detection.baseline0.analysis.gradcam import (
    compute_gradcam,
    load_bag_arrays,
    load_baseline0_checkpoint,
    load_oof_predictions,
    prepare_inputs,
)
from fracture_detection.baseline0.cli.attention import (
    DEFAULT_EXPERIMENT_DIR,
    attach_annotation_validity,
    load_annotation_coverage,
)
from fracture_detection.common.constants import DATASET_DIR, REGION_COLUMNS

N_FOLDS = 5
OUTPUT_NAME = "cam_generation_audit"
SCORES_CSV = "cam_audit_scores.csv"
LOCALIZATION_CSV = "cam_audit_localization.csv"
MEMORIZATION_CSV = "cam_audit_memorization.csv"
PERTURBATION_CSV = "cam_audit_mask_perturbation.csv"
TTA_CSV = "cam_audit_tta.csv"
VERDICT_JSON = "cam_audit_verdict.json"


def run_audit(args: argparse.Namespace) -> Path:
    """Score annotated bags with every teacher and write the audit tables."""
    experiment_dir = cast(Path, args.experiment_dir).resolve()
    dataset_dir = cast(Path, args.dataset_dir).resolve()
    output_dir = (
        cast(Path, args.output_dir).resolve()
        if args.output_dir is not None
        else experiment_dir / OUTPUT_NAME
    )
    _guard_output(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    bags = _select_annotated_bags(experiment_dir, args.limit_bags)
    perturbations = default_perturbations()
    tta_modes = (NO_TTA,) if args.no_tta else (NO_TTA, HFLIP_TTA)
    device = _resolve_device(args.device)
    print(
        f"CAM audit: {len(bags)} bags x {N_FOLDS} teachers x {len(tta_modes)} TTA "
        f"x {len(perturbations)} mask variants, device={device}",
        flush=True,
    )

    records = bags.to_dict("records")
    arrays = _preload_bags(dataset_dir, records)
    rows: list[dict[str, Any]] = []
    for teacher in range(N_FOLDS):
        checkpoint = experiment_dir / f"outer{teacher}" / args.checkpoint_name
        model = load_baseline0_checkpoint(checkpoint, device)
        for tta in tta_modes:
            # The flip probe only feeds the descriptive TTA table, which is read
            # on the held-out teacher, so it is not run for the other four.
            scope = (
                [r for r in records if int(r["fold"]) == teacher]
                if tta == HFLIP_TTA
                else records
            )
            done = 0
            for start in range(0, len(scope), args.batch_size):
                batch = scope[start : start + args.batch_size]
                rows.extend(
                    _score_batch(
                        model=model,
                        device=device,
                        arrays=arrays,
                        records=batch,
                        teacher=teacher,
                        tta=tta,
                        perturbations=perturbations,
                    )
                )
                done += len(batch)
                if done % 50 < args.batch_size or done == len(scope):
                    print(f"  teacher{teacher} {tta}: {done}/{len(scope)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / SCORES_CSV, index=False)
    localization = localization_table(scores)
    localization.to_csv(output_dir / LOCALIZATION_CSV, index=False)
    memorization = memorization_table(scores)
    memorization.to_csv(output_dir / MEMORIZATION_CSV, index=False)
    perturbation = perturbation_table(scores)
    perturbation.to_csv(output_dir / PERTURBATION_CSV, index=False)
    verdict: dict[str, Any] = dict(audit_verdict(memorization, perturbation))
    if HFLIP_TTA in tta_modes:
        flip = tta_table(scores)
        flip.to_csv(output_dir / TTA_CSV, index=False)
        verdict["tta_min_spearman"] = float(flip["spearman_vs_identity"].min())
        verdict["tta_argmax_change_rate"] = float(flip["argmax_change_rate"].iloc[0])
    verdict["laterality"] = laterality_summary(scores)
    verdict["n_bags"] = int(len(bags))
    verdict["n_studies"] = int(bags["study_id"].nunique())
    verdict["gate_perturbations"] = list(gate_perturbation_names())
    verdict["checkpoint_name"] = str(args.checkpoint_name)
    verdict["experiment_dir"] = str(experiment_dir)
    (output_dir / VERDICT_JSON).write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_report(localization, memorization, perturbation, verdict)
    return output_dir


def _select_annotated_bags(experiment_dir: Path, limit: int | None) -> pd.DataFrame:
    predictions = attach_annotation_validity(
        load_oof_predictions(experiment_dir), load_annotation_coverage()
    )
    annotated = predictions[predictions["has_region_target"].astype(bool)]
    annotated = annotated.sort_values(["study_id", "level"]).reset_index(drop=True)
    if annotated.empty:
        raise ValueError("No annotated bags were found in the OOF predictions")
    if limit is not None:
        annotated = annotated.head(limit).reset_index(drop=True)
    return annotated


def _preload_bags(
    dataset_dir: Path, records: list[dict[str, Any]]
) -> dict[tuple[str, str], tuple[Any, Any, Any]]:
    """Read every bag once so the teacher/TTA loops stay off the filesystem."""
    cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    for record in records:
        key = (str(record["study_id"]), str(record["level"]))
        if key not in cache:
            cache[key] = load_bag_arrays(dataset_dir, key[0], key[1])
    return cache


def _score_batch(
    model: Any,
    device: torch.device,
    arrays: dict[tuple[str, str], tuple[Any, Any, Any]],
    records: list[dict[str, Any]],
    teacher: int,
    tta: str,
    perturbations: tuple[MaskPerturbation, ...],
) -> list[dict[str, Any]]:
    """Grad-CAM one batch of bags and aggregate it under every mask variant."""
    loaded = [arrays[(str(r["study_id"]), str(r["level"]))] for r in records]
    if tta == HFLIP_TTA:
        inputs = torch.stack(
            [
                prepare_inputs(
                    flip_planes_horizontally(ct), flip_planes_horizontally(whole)
                )
                for ct, whole, _ in loaded
            ]
        )
    else:
        inputs = torch.stack([prepare_inputs(ct, whole) for ct, whole, _ in loaded])
    result = compute_gradcam(model, inputs, device)

    rows: list[dict[str, Any]] = []
    for index, (record, bag_arrays) in enumerate(zip(records, loaded, strict=True)):
        _, whole_mask, region_mask = bag_arrays
        cams = result.cams[index]
        if tta == HFLIP_TTA:
            # Back into the original frame, so the unflipped masks stay valid.
            cams = flip_planes_horizontally(cams)
        role = teacher_role(int(record["fold"]), teacher, N_FOLDS)
        cam_total = float(cams.sum(dtype=np.float64))
        for perturbation in perturbations:
            scores = region_density_enrichment(
                cams, whole_mask, region_mask, perturbation
            )
            row: dict[str, Any] = {
                "study_id": str(record["study_id"]),
                "level": str(record["level"]),
                "fold": int(record["fold"]),
                "teacher": teacher,
                "role": role,
                "tta": tta,
                "perturbation": perturbation.name,
                "bag_probability": float(result.bag_probabilities[index]),
                "cam_total": cam_total,
            }
            for region_index, region_column in enumerate(REGION_COLUMNS):
                row[f"{region_column}_score"] = float(scores[region_index])
                row[region_column] = int(record[region_column])
                row[f"{region_column}_target_valid"] = bool(
                    record[f"{region_column}_target_valid"]
                )
            rows.append(row)
    return rows


def _print_report(
    localization: pd.DataFrame,
    memorization: pd.DataFrame,
    perturbation: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    formatter = "{:.4f}".format
    print("\n== held-out CAM localization (teacher signal) ==")
    print(localization.to_string(index=False, float_format=formatter))
    print("\n== teacher memorization (in-sample train vs held-out outer) ==")
    print(
        memorization[
            [
                "region_name",
                "n_positive",
                "n_negative",
                "auroc_train",
                "auroc_inner",
                "auroc_outer",
                "auroc_difference",
                "difference_ci_low",
                "difference_ci_high",
                "smd_positive",
                "smd_negative",
            ]
        ].to_string(index=False, float_format=formatter)
    )
    gate = perturbation[perturbation["variant"].isin(gate_perturbation_names())]
    print("\n== mask perturbation at the gate magnitude ==")
    print(
        gate[
            [
                "variant",
                "region_name",
                "spearman_vs_identity",
                "argmax_change_rate",
                "auroc",
                "auroc_identity",
            ]
        ].to_string(index=False, float_format=formatter)
    )
    print("\n== verdict ==")
    for key, value in verdict.items():
        print(f"  {key}: {value}")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _guard_output(output_dir: Path, overwrite: bool) -> None:
    expected = (
        SCORES_CSV,
        LOCALIZATION_CSV,
        MEMORIZATION_CSV,
        PERTURBATION_CSV,
        TTA_CSV,
        VERDICT_JSON,
    )
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Audit output already exists; pass --overwrite: {existing[0]}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline 0のGrad-CAM疑似ラベル教師信号を生成前に監査する"
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-bags", type=int, help="smoke run上限")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run_audit(parse_args())


if __name__ == "__main__":
    main()
