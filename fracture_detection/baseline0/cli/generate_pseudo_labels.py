"""Generate train and leakage-safe inner CAM pseudo-region targets.

Per ``.claude/docs/REGION_MODEL_DESIGN_JA.md`` Section 5 and
``.claude/docs/work-logs/2026-09/2026-09-01-cam-soft-bce-implementation-plan.md``:
the earlier cross-case pairwise-ranking pseudo label (``pseudo_labeling/scoring.py``)
is retired. Each whole-positive bag now gets an independent per-region soft
probability ``q`` from a fold-matched Baseline 0 teacher's four-view
test-time-augmented Grad-CAM, converted to a probability by one shared
(region-agnostic) logit-share Platt map fit on that student outer fold's
complete, fully human-annotated whole-positive bags.

This writes three new artifacts (``pseudo_region_targets.csv``,
``pseudo_target_calibration.csv``, ``pseudo_target_generation_metadata.json``)
without touching the retired ``pseudo_label_scores.csv`` /
``pseudo_label_temperatures.csv`` files from the pairwise-ranking pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from fracture_detection.baseline0.data.constants import (
    DATASET_DIR,
    REGION_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)
from fracture_detection.baseline0.data.dataset import load_manifest
from fracture_detection.baseline0.data.splits import (
    resolve_nested_folds,
    split_nested_manifest,
)
from fracture_detection.baseline0.data.staging import manifest_sha256
from fracture_detection.baseline0.pseudo_labeling.calibration import (
    N_REGIONS,
    REGULARIZATION_C,
    SHARE_FLOOR,
    SharedLogitShareCalibration,
    apply_shared_logit_share_calibration,
    average_view_shares,
    cross_validated_calibration_quality,
    enrichment_to_share,
    fit_shared_logit_share_calibration,
    project_sum_at_least_one,
    summarize_probability_distribution,
)
from fracture_detection.baseline0.pseudo_labeling.cam_audit import (
    MaskPerturbation,
    region_density_enrichment,
)
from fracture_detection.baseline0.pseudo_labeling.gradcam import (
    DEFAULT_TTA_VIEWS,
    apply_tta_view_to_inputs,
    compute_gradcam,
    invert_tta_view_on_cam,
    load_bag_arrays,
    load_baseline0_checkpoint,
    prepare_inputs,
)

N_FOLDS = 5
IDENTITY = MaskPerturbation("identity", "identity")
DEFAULT_EXPERIMENT_DIR = Path(
    "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加"
)
REGION_TARGETS_CSV = "pseudo_region_targets.csv"
CALIBRATION_CSV = "pseudo_target_calibration.csv"
METADATA_JSON = "pseudo_target_generation_metadata.json"
CALIBRATION_FORMULA = (
    "s_vr=e_vr/sum_j(e_vj); s_r=mean_v(s_vr); "
    "x_r=logit(clip(s_r,share_floor,1-share_floor)); "
    "q*_r=sigmoid(a_k*x_r+b_k); "
    "q_r=q*_r if sum_j(q*_j)>=1 else q*_r/sum_j(q*_j)"
)
MIN_FIT_POPULATION = 2


def run_generation(args: argparse.Namespace) -> Path:
    """Score student train bags and leakage-safe inner bags for every outer fold."""
    experiment_dir = cast(Path, args.experiment_dir).resolve()
    dataset_dir = cast(Path, args.dataset_dir).resolve()
    output_dir = cast(Path, args.output_dir).resolve()
    _guard_output(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    device = _resolve_device(args.device)
    smoke_only = args.limit_bags is not None
    print(
        f"Pseudo-target generation: {N_FOLDS} teachers, device={device}, "
        f"views={len(DEFAULT_TTA_VIEWS)}, smoke_only={smoke_only}",
        flush=True,
    )

    all_target_rows: list[dict[str, Any]] = []
    all_calibration_rows: list[dict[str, Any]] = []
    teacher_assignments: list[dict[str, Any]] = []
    q_pool_for_metadata: list[np.ndarray] = []

    folds = (
        [args.only_fold] if args.only_fold is not None else list(range(N_FOLDS))
    )
    for outer_fold in folds:
        assignment = resolve_nested_folds(outer_fold, N_FOLDS)
        train, _inner, outer = split_nested_manifest(manifest, outer_fold, N_FOLDS)
        if args.include_negatives:
            # 陰性bagのregion周辺確率は真ラベルから厳密に0だが、CAM shareを実測して
            # artifactへ残すため採点対象に含める。校正fit母集団は陽性のみのまま。
            whole_positive = train.reset_index(drop=True)
            oof_positive = outer.reset_index(drop=True)
        else:
            whole_positive = train[train["vertebra_target"].eq(1)].reset_index(drop=True)
            oof_positive = outer[outer["vertebra_target"].eq(1)].reset_index(drop=True)
        if args.limit_bags is not None:
            whole_positive = whole_positive.head(args.limit_bags)
            oof_positive = oof_positive.head(args.limit_bags)

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
        records = whole_positive.to_dict("records")
        print(
            f"  outer{outer_fold} teacher scoring {len(records)} train and "
            f"{len(oof_positive)} held-out whole-positive bags",
            flush=True,
        )

        shares_by_row: list[np.ndarray] = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            shares_by_row.extend(_score_batch_shares(model, device, dataset_dir, batch))
            done = min(start + args.batch_size, len(records))
            if done % 400 < args.batch_size or done == len(records):
                print(f"    {done}/{len(records)}", flush=True)
        averaged_shares = (
            np.stack(shares_by_row)
            if shares_by_row
            else np.zeros((0, N_REGIONS), dtype=np.float64)
        )
        q, calibration_row = _calibrate_fold(
            outer_fold, whole_positive, averaged_shares, args
        )
        all_calibration_rows.append(calibration_row)
        if not np.isnan(q).all() and q.size:
            q_pool_for_metadata.append(q)

        all_target_rows.extend(
            _target_rows(
                outer_fold,
                outer_fold,
                records,
                averaged_shares,
                q,
                checkpoint_hash,
            )
        )

        oof_records = oof_positive.to_dict("records")
        oof_shares_by_row = _score_records(
            model, device, dataset_dir, oof_records, args.batch_size
        )
        oof_shares = (
            np.stack(oof_shares_by_row)
            if oof_shares_by_row
            else np.zeros((0, N_REGIONS), dtype=np.float64)
        )
        oof_q = _apply_calibration_row(oof_shares, calibration_row)
        if not np.isnan(oof_q).all() and oof_q.size:
            q_pool_for_metadata.append(oof_q)
        inner_student_fold = (outer_fold - 1) % N_FOLDS
        all_target_rows.extend(
            _target_rows(
                inner_student_fold,
                outer_fold,
                oof_records,
                oof_shares,
                oof_q,
                checkpoint_hash,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    targets = pd.DataFrame(all_target_rows)
    if targets.duplicated(["student_outer_fold", "study_id", "level"]).any():
        raise ValueError("generated pseudo targets contain a duplicate key")
    targets_path = output_dir / REGION_TARGETS_CSV
    _atomic_write_csv(targets, targets_path)

    calibration_frame = pd.DataFrame(all_calibration_rows)
    calibration_path = output_dir / CALIBRATION_CSV
    _atomic_write_csv(calibration_frame, calibration_path)

    metadata: dict[str, Any] = {
        "views": [view.name for view in DEFAULT_TTA_VIEWS],
        "share_floor": args.share_floor,
        "regularization_c": args.regularization_c,
        "formula": CALIBRATION_FORMULA,
        "n_rows": int(len(targets)),
        "n_bags": int(targets[["study_id", "level"]].drop_duplicates().shape[0]),
        "checkpoint_name": str(args.checkpoint_name),
        "experiment_dir": str(experiment_dir),
        "smoke_only": smoke_only,
        "targets_sha256": manifest_sha256(targets_path),
        "calibration_sha256": manifest_sha256(calibration_path),
        "teacher_assignments": teacher_assignments,
    }
    if q_pool_for_metadata:
        pooled = np.concatenate(q_pool_for_metadata, axis=0)
        metadata["probability_distribution"] = summarize_probability_distribution(
            pooled
        )
    _atomic_write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        output_dir / METADATA_JSON,
    )

    print("\n== calibration (student_outer_fold x shared slope/intercept) ==")
    print(calibration_frame.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nwrote {len(targets)} rows for {metadata['n_bags']} bags to {output_dir}")
    return output_dir


def _score_records(
    model: Any,
    device: torch.device,
    dataset_dir: Path,
    records: list[dict[str, Any]],
    batch_size: int,
) -> list[np.ndarray]:
    """任意record列をbatch分割して4-view CAM shareへ変換する。"""
    shares: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        shares.extend(
            _score_batch_shares(
                model, device, dataset_dir, records[start : start + batch_size]
            )
        )
    return shares


def _apply_calibration_row(
    shares: np.ndarray, calibration_row: dict[str, Any]
) -> np.ndarray:
    """fold校正係数をheld-out shareへ適用する。"""
    slope = float(calibration_row["slope"])
    intercept = float(calibration_row["intercept"])
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.full_like(shares, np.nan, dtype=np.float64)
    calibration = SharedLogitShareCalibration(
        student_outer_fold=int(calibration_row["student_outer_fold"]),
        slope=slope,
        intercept=intercept,
        n_fit_bags=int(calibration_row["n_fit_bags"]),
        n_fit_studies=int(calibration_row["n_fit_studies"]),
        share_floor=float(calibration_row["share_floor"]),
    )
    return project_sum_at_least_one(
        apply_shared_logit_share_calibration(shares, calibration)
    )


def _target_rows(
    student_outer_fold: int,
    teacher_outer_fold: int,
    records: list[dict[str, Any]],
    shares: np.ndarray,
    targets: np.ndarray,
    checkpoint_hash: str,
) -> list[dict[str, Any]]:
    """CAM shareとsoft targetをartifact行へ変換する。"""
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "student_outer_fold": student_outer_fold,
            "study_id": str(record["study_id"]),
            "level": str(record["level"]),
            "teacher_outer_fold": teacher_outer_fold,
            "teacher_checkpoint_sha256": checkpoint_hash,
            "vertebra_target": int(record["vertebra_target"]),
        }
        is_negative = int(record["vertebra_target"]) == 0
        row["target_source"] = "hard_negative" if is_negative else "cam_pseudo"
        for region_index, region_column in enumerate(REGION_COLUMNS):
            row[f"{region_column}_cam_share"] = float(shares[index, region_index])
            # whole陰性bagの周辺確率は真ラベルから厳密に0。CAM shareはbag内で和が1に
            # 正規化されるため陰性でも各領域へ配分されるが、それを教師にはしない。
            row[f"{region_column}_pseudo_target"] = (
                0.0 if is_negative else float(targets[index, region_index])
            )
        rows.append(row)
    return rows


def _calibrate_fold(
    outer_fold: int,
    whole_positive: pd.DataFrame,
    averaged_shares: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit (if possible) and apply one outer fold's shared calibration map.

    The fit population is restricted to whole-positive bags whose four human
    region targets are all valid, independent of ``--limit-bags``: a smoke
    run is only marked ``smoke_only`` in the artifact metadata, it is not
    given a separate code path.
    """
    fit_mask = (
        whole_positive["has_region_target"].astype(bool)
        & whole_positive.loc[:, list(REGION_TARGET_VALID_COLUMNS)].all(axis=1)
    ).to_numpy()
    n_fit = int(fit_mask.sum())
    n_fit_studies = int(whole_positive.loc[fit_mask, "study_id"].nunique())

    if n_fit < MIN_FIT_POPULATION:
        q = np.full((len(whole_positive), N_REGIONS), np.nan)
        calibration_row: dict[str, Any] = {
            "student_outer_fold": outer_fold,
            "slope": float("nan"),
            "intercept": float("nan"),
            "n_fit_bags": n_fit,
            "n_fit_studies": n_fit_studies,
            "share_floor": args.share_floor,
            "oof_available": 0.0,
        }
        return q, calibration_row

    fit_shares = averaged_shares[fit_mask]
    fit_targets = whole_positive.loc[fit_mask, list(REGION_COLUMNS)].to_numpy(
        dtype=np.int64
    )
    fit_studies = whole_positive.loc[fit_mask, "study_id"].to_numpy()
    calibration = fit_shared_logit_share_calibration(
        fit_shares,
        fit_targets,
        fit_studies,
        outer_fold,
        share_floor=args.share_floor,
        regularization_c=args.regularization_c,
    )
    quality = cross_validated_calibration_quality(
        fit_shares,
        fit_targets,
        fit_studies,
        outer_fold,
        share_floor=args.share_floor,
        regularization_c=args.regularization_c,
    )
    q_star = apply_shared_logit_share_calibration(averaged_shares, calibration)
    q = project_sum_at_least_one(q_star)
    calibration_row = {
        "student_outer_fold": outer_fold,
        "slope": calibration.slope,
        "intercept": calibration.intercept,
        "n_fit_bags": calibration.n_fit_bags,
        "n_fit_studies": calibration.n_fit_studies,
        "share_floor": calibration.share_floor,
        **quality,
    }
    return q, calibration_row


def _score_batch_shares(
    model: Any,
    device: torch.device,
    dataset_dir: Path,
    batch: list[dict[str, Any]],
) -> list[np.ndarray]:
    """Four-view TTA-averaged region share for every bag in one batch.

    Each view transforms the model input, recomputes Grad-CAM, and maps the
    CAM back into the native frame before aggregating under the untouched
    (native) region mask — see ``gradcam.TTAView`` for why the region mask
    itself never needs to move.
    """
    loaded = [
        load_bag_arrays(dataset_dir, str(record["study_id"]), str(record["level"]))
        for record in batch
    ]
    view_shares: list[np.ndarray] = []
    for view in DEFAULT_TTA_VIEWS:
        transformed = [
            apply_tta_view_to_inputs(ct, whole_mask, view)
            for ct, whole_mask, _ in loaded
        ]
        inputs = torch.stack(
            [
                prepare_inputs(view_ct, view_whole_mask)
                for view_ct, view_whole_mask in transformed
            ]
        )
        result = compute_gradcam(model, inputs, device)
        enrichments = np.zeros((len(batch), N_REGIONS), dtype=np.float64)
        for index, (_, whole_mask, region_mask) in enumerate(loaded):
            native_cam = invert_tta_view_on_cam(result.cams[index], view)
            enrichments[index] = region_density_enrichment(
                native_cam, whole_mask, region_mask, IDENTITY
            )
        view_shares.append(enrichment_to_share(enrichments))
    averaged = average_view_shares(np.stack(view_shares, axis=0))
    return [averaged[index] for index in range(len(batch))]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _guard_output(output_dir: Path, overwrite: bool) -> None:
    expected = (REGION_TARGETS_CSV, CALIBRATION_CSV, METADATA_JSON)
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Pseudo-target output already exists; pass --overwrite: {existing[0]}"
        )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("fold-matched teacherの4-view TTA CAMから疑似区域確率を生成する")
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fracture_detection/baseline0/outputs/09_04/pseudo_labels"),
    )
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--share-floor", type=float, default=SHARE_FLOOR)
    parser.add_argument("--regularization-c", type=float, default=REGULARIZATION_C)
    parser.add_argument("--limit-bags", type=int, help="smoke run上限（fold毎）")
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="whole陰性bagも採点しartifactへ含める（pseudo_targetは厳密0）",
    )
    parser.add_argument(
        "--only-fold", type=int, help="この teacher fold だけ実行する（GPU分割用）"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run_generation(parse_args())


if __name__ == "__main__":
    main()
