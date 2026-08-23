"""Generate OOF Grad-CAM visualizations and anatomical summaries."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import matplotlib_fontja  # type: ignore[import-untyped]  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)

from fracture_detection.baseline0.analysis.gradcam import (
    GradCamBatch,
    anatomical_attention_metrics,
    compute_gradcam,
    load_bag_arrays,
    load_baseline0_checkpoint,
    load_oof_predictions,
    prepare_inputs,
    select_stratified_high_scores,
)
from fracture_detection.common.constants import (
    DATASET_DIR,
    REGION_COLUMNS,
    REGION_NAMES,
)

DEFAULT_EXPERIMENT_DIR = Path(
    "fracture_detection/baseline0/outputs/08_19/baseline0_shared_core"
)
REGION_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d627b4")
REGION_DISPLAY_NAMES = ("椎体", "右横突孔", "左横突孔", "後方要素")
LOCALIZATION_BOOTSTRAP_SAMPLES = 2_000
LOCALIZATION_BOOTSTRAP_SEED = 20_260_821


def run_analysis(args: argparse.Namespace) -> Path:
    """Run checkpoint-matched OOF attention analysis."""
    experiment_dir = cast(Path, args.experiment_dir).resolve()
    dataset_dir = cast(Path, args.dataset_dir).resolve()
    output_name = (
        "gradcam_annotated" if args.selection == "annotated" else "gradcam_attention"
    )
    output_dir = (
        cast(Path, args.output_dir).resolve()
        if args.output_dir is not None
        else experiment_dir / output_name
    )
    _guard_output(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "cases"
    figure_dir.mkdir(exist_ok=True)

    annotation_coverage = load_annotation_coverage()
    annotation_coverage.to_csv(output_dir / "annotation_coverage.csv", index=False)
    predictions = _attach_annotation_validity(
        load_oof_predictions(experiment_dir), annotation_coverage
    )
    if args.selection == "annotated":
        selected = predictions[
            predictions["has_region_target"].astype(bool)
        ].sort_values(["fold", "study_id", "level"])
        selection_description = "all bags with at least one annotated fracture run"
    else:
        categories = tuple(args.categories)
        selected = select_stratified_high_scores(
            predictions,
            categories,
            args.samples_per_stratum,
        )
        selection_description = (
            "highest score within each category/fold/vertebral-level stratum"
        )
    if selected.empty:
        raise ValueError("No OOF samples matched the requested categories")
    selected.to_csv(output_dir / "selected_samples.csv", index=False)
    figure_keys = _select_figure_keys(selected, args.figures_per_category)

    device = _resolve_device(args.device)
    records: list[dict[str, Any]] = []
    print(
        f"Grad-CAM: {len(selected)} bags, device={device}, output={output_dir}",
        flush=True,
    )
    for fold, fold_frame in selected.groupby("fold", sort=True):
        checkpoint_path = experiment_dir / f"outer{int(fold)}" / args.checkpoint_name
        model = load_baseline0_checkpoint(checkpoint_path, device)
        rows = list(fold_frame.to_dict("records"))
        for batch_rows in _batches(rows, args.batch_size):
            loaded = [
                load_bag_arrays(dataset_dir, str(row["study_id"]), str(row["level"]))
                for row in batch_rows
            ]
            inputs = torch.stack(
                [prepare_inputs(ct, whole_mask) for ct, whole_mask, _ in loaded]
            )
            result = compute_gradcam(model, inputs, device)
            for index, (row, arrays) in enumerate(zip(batch_rows, loaded, strict=True)):
                ct, whole_mask, region_mask = arrays
                record = _build_record(row, result, index, whole_mask, region_mask)
                records.append(record)
                key = (str(row["study_id"]), str(row["level"]))
                if key in figure_keys:
                    figure = build_case_figure(
                        row,
                        ct,
                        whole_mask,
                        region_mask,
                        result.cams[index],
                        result.plane_probabilities[index],
                        record,
                    )
                    filename = _figure_filename(row)
                    figure.savefig(
                        figure_dir / filename,
                        dpi=140,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    plt.close(figure)
            print(
                f"  outer{int(fold)}: {min(len(records), len(selected))}/{len(selected)}",
                flush=True,
            )
        del model

    metrics = pd.DataFrame(records).sort_values(
        ["category", "fold", "level", "vertebra_score"],
        ascending=[True, True, True, False],
    )
    metrics.to_csv(output_dir / "attention_metrics.csv", index=False)
    summary = summarize_attention(metrics)
    summary.to_csv(output_dir / "region_summary.csv", index=False)
    annotated_summary = summarize_annotated_targets(metrics)
    annotated_summary.to_csv(output_dir / "annotated_target_summary.csv", index=False)
    localization_summary = summarize_annotated_localization(metrics)
    localization_summary.to_csv(
        output_dir / "annotated_localization_metrics.csv", index=False
    )
    overall = summarize_overall(metrics, selection_description)
    (output_dir / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_figure = build_summary_figure(summary, overall)
    summary_figure.savefig(
        output_dir / "region_summary.png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(summary_figure)
    print(f"Saved Grad-CAM analysis: {output_dir}", flush=True)
    return output_dir


def _build_record(
    row: dict[str, Any],
    result: GradCamBatch,
    index: int,
    whole_mask: NDArray[np.uint8],
    region_mask: NDArray[np.uint8],
) -> dict[str, Any]:
    metrics = anatomical_attention_metrics(result.cams[index], whole_mask, region_mask)
    plane_probabilities = result.plane_probabilities[index]
    return {
        **row,
        "recomputed_fp32_score": float(result.bag_probabilities[index]),
        "score_absolute_difference": abs(
            float(result.bag_probabilities[index]) - float(row["vertebra_score"])
        ),
        **metrics,
        **{
            f"plane_{plane_index:02d}_probability": float(probability)
            for plane_index, probability in enumerate(plane_probabilities)
        },
    }


def summarize_attention(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a tidy region-level descriptive summary."""
    rows: list[dict[str, Any]] = []
    for category, category_frame in metrics.groupby("category", sort=True):
        for region_index, region_column in enumerate(REGION_COLUMNS):
            mass = category_frame[f"{region_column}_mass_fraction"]
            area = category_frame[f"{region_column}_area_fraction"]
            density = category_frame[f"{region_column}_density_enrichment"]
            rows.append(
                {
                    "category": str(category),
                    "region": region_column,
                    "region_name": REGION_NAMES[region_index],
                    "n": int(len(category_frame)),
                    "mass_fraction_mean": float(mass.mean()),
                    "mass_fraction_median": float(mass.median()),
                    "area_fraction_mean": float(area.mean()),
                    "density_enrichment_mean": float(density.mean()),
                    "density_enrichment_median": float(density.median()),
                }
            )
    return pd.DataFrame(rows)


def summarize_overall(
    metrics: pd.DataFrame,
    selection_description: str,
) -> dict[str, Any]:
    """Summarize coverage and score-recomputation diagnostics."""
    categories: dict[str, Any] = {}
    for category, frame in metrics.groupby("category", sort=True):
        categories[str(category)] = {
            "n": int(len(frame)),
            "mean_oof_score": float(frame["vertebra_score"].mean()),
            "mean_in_vertebra_mass_fraction": float(
                frame["in_vertebra_mass_fraction"].mean()
            ),
            "median_in_vertebra_mass_fraction": float(
                frame["in_vertebra_mass_fraction"].median()
            ),
            "mean_vertebra_area_fraction": float(
                frame["vertebra_area_fraction"].mean()
            ),
            "mean_vertebra_density_enrichment": float(
                frame["vertebra_density_enrichment"].mean()
            ),
            "mean_score_absolute_difference": float(
                frame["score_absolute_difference"].mean()
            ),
            "max_score_absolute_difference": float(
                frame["score_absolute_difference"].max()
            ),
            "zero_cam_count": int(frame["cam_zero"].sum()),
        }
    return {
        "method": "Grad-CAM on encoder.bn2 for mean(sigmoid(plane_logits))",
        "selection": selection_description,
        "categories": categories,
        "annotated_bags": int(metrics["has_region_target"].sum()),
        "complete_annotation_bags": int(
            metrics.loc[
                metrics["has_region_target"].astype(bool), "annotation_complete"
            ].sum()
        ),
    }


def summarize_annotated_targets(metrics: pd.DataFrame) -> pd.DataFrame:
    """Describe CAM density using only known positive and negative targets."""
    annotated = metrics[metrics["has_region_target"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for region_index, region_column in enumerate(REGION_COLUMNS):
        valid = annotated[_region_target_valid(annotated, region_column)]
        positives = valid[valid[region_column].eq(1)]
        negatives = valid[valid[region_column].eq(0)]
        positive_density = positives[f"{region_column}_density_enrichment"]
        negative_density = negatives[f"{region_column}_density_enrichment"]
        positive_mass = positives[f"{region_column}_mass_fraction"]
        negative_mass = negatives[f"{region_column}_mass_fraction"]
        positive_density_mean = float(positive_density.mean())
        negative_density_mean = float(negative_density.mean())
        rows.append(
            {
                "target_region": region_column,
                "target_region_name": REGION_NAMES[region_index],
                "n_positive": int(len(positives)),
                "n_negative": int(len(negatives)),
                "n_unknown": int(len(annotated) - len(valid)),
                "positive_mass_fraction_mean": float(positive_mass.mean()),
                "negative_mass_fraction_mean": float(negative_mass.mean()),
                "mass_fraction_mean_difference": float(
                    positive_mass.mean() - negative_mass.mean()
                ),
                "positive_density_enrichment_mean": positive_density_mean,
                "negative_density_enrichment_mean": negative_density_mean,
                "density_enrichment_mean_difference": (
                    positive_density_mean - negative_density_mean
                ),
                "positive_density_enrichment_median": float(positive_density.median()),
                "negative_density_enrichment_median": float(negative_density.median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_annotated_localization(
    metrics: pd.DataFrame,
    bootstrap_samples: int = LOCALIZATION_BOOTSTRAP_SAMPLES,
    seed: int = LOCALIZATION_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Evaluate CAM density against observed positives and complete negatives."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")
    annotated = metrics[metrics["has_region_target"].astype(bool)].reset_index(
        drop=True
    )
    if annotated.empty:
        return pd.DataFrame()
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for region_index, region_column in enumerate(REGION_COLUMNS):
        valid = annotated[_region_target_valid(annotated, region_column)].reset_index(
            drop=True
        )
        targets = valid[region_column].to_numpy(dtype=np.int8)
        density = valid[f"{region_column}_density_enrichment"].to_numpy(
            dtype=np.float64
        )
        within_level_rank = (
            valid.groupby("level")[f"{region_column}_density_enrichment"]
            .rank(pct=True, method="average")
            .to_numpy(dtype=np.float64)
        )
        study_groups = [
            group.index.to_numpy() for _, group in valid.groupby("study_id", sort=True)
        ]
        positive = density[targets == 1]
        negative = density[targets == 0]
        if len(positive) == 0 or len(negative) == 0:
            rows.append(
                _empty_localization_row(
                    region_index,
                    region_column,
                    len(positive),
                    len(negative),
                    len(annotated) - len(valid),
                )
            )
            continue
        bootstrap_differences: list[float] = []
        for _ in range(bootstrap_samples):
            sampled_indices = np.concatenate(
                [
                    study_groups[index]
                    for index in generator.integers(
                        0, len(study_groups), len(study_groups)
                    )
                ]
            )
            sampled_targets = targets[sampled_indices]
            if sampled_targets.min() == sampled_targets.max():
                continue
            sampled_density = density[sampled_indices]
            bootstrap_differences.append(
                float(
                    sampled_density[sampled_targets == 1].mean()
                    - sampled_density[sampled_targets == 0].mean()
                )
            )
        rows.append(
            {
                "region": region_column,
                "region_name": REGION_NAMES[region_index],
                "n_positive": int(len(positive)),
                "n_negative": int(len(negative)),
                "n_unknown": int(len(annotated) - len(valid)),
                "prevalence": float(targets.mean()),
                "density_mean_difference": float(positive.mean() - negative.mean()),
                "density_difference_ci_low": float(
                    np.percentile(bootstrap_differences, 2.5)
                ),
                "density_difference_ci_high": float(
                    np.percentile(bootstrap_differences, 97.5)
                ),
                "density_auroc": float(roc_auc_score(targets, density)),
                "density_average_precision": float(
                    average_precision_score(targets, density)
                ),
                "within_level_rank_auroc": float(
                    roc_auc_score(targets, within_level_rank)
                ),
                "within_level_rank_average_precision": float(
                    average_precision_score(targets, within_level_rank)
                ),
            }
        )
    return pd.DataFrame(rows)


def _empty_localization_row(
    region_index: int,
    region_column: str,
    positive_count: int,
    negative_count: int,
    unknown_count: int,
) -> dict[str, Any]:
    return {
        "region": region_column,
        "region_name": REGION_NAMES[region_index],
        "n_positive": positive_count,
        "n_negative": negative_count,
        "n_unknown": unknown_count,
        "prevalence": float("nan"),
        "density_mean_difference": float("nan"),
        "density_difference_ci_low": float("nan"),
        "density_difference_ci_high": float("nan"),
        "density_auroc": float("nan"),
        "density_average_precision": float("nan"),
        "within_level_rank_auroc": float("nan"),
        "within_level_rank_average_precision": float("nan"),
    }


def _region_target_valid(metrics: pd.DataFrame, region_column: str) -> pd.Series:
    """Mark observed positives and fully reviewed zero targets as valid."""
    validity_column = f"{region_column}_target_valid"
    if validity_column in metrics.columns:
        return metrics[validity_column].astype(bool)
    if "annotation_complete" not in metrics.columns:
        raise ValueError("annotation_complete is required for region evaluation")
    return metrics[region_column].eq(1) | metrics["annotation_complete"].astype(bool)


def load_annotation_coverage() -> pd.DataFrame:
    """Derive bag-level run completion from the annotation tool source inventory."""
    annotation_tool = importlib.import_module("Unet.dicom_bbox_annotation_tool.server")

    targets = annotation_tool.augment_with_missing_fractured_levels(
        annotation_tool.build_targets(
            annotation_tool.DEFAULT_BBOX_CSV,
            annotation_tool.DEFAULT_METADATA_DIR,
        ),
        annotation_tool.DEFAULT_BBOX_CSV,
        annotation_tool.DEFAULT_METADATA_DIR,
        annotation_tool.DEFAULT_TRAIN_CSV,
        annotation_tool.DEFAULT_TRAIN_IMAGES_DIR,
        annotation_tool.DEFAULT_SEGMENTATION_DIR,
        annotation_tool.DEFAULT_AUGMENT_CACHE,
    )
    label_keys = set(
        annotation_tool.LabelStore(annotation_tool.DEFAULT_LABEL_CSV).read()
    )
    target_keys = {target.label_key for target in targets}
    unmatched = label_keys - target_keys
    if unmatched:
        raise ValueError(
            f"Region labels do not match annotation targets: {len(unmatched)}"
        )

    annotated_bags = {(study_id, level) for study_id, level, _ in label_keys}
    rows: list[dict[str, Any]] = []
    for study_id, level in sorted(annotated_bags):
        bag_targets = [
            target
            for target in targets
            if target.study_id == study_id and target.level == level
        ]
        annotatable = [target for target in bag_targets if target.rows]
        annotated_runs = sum(target.label_key in label_keys for target in annotatable)
        bbox_missing_runs = sum(not target.rows for target in bag_targets)
        unannotated_runs = len(annotatable) - annotated_runs
        rows.append(
            {
                "study_id": study_id,
                "level": level,
                "expected_annotation_runs": len(annotatable),
                "annotated_runs": annotated_runs,
                "unannotated_runs": unannotated_runs,
                "bbox_missing_runs": bbox_missing_runs,
                "annotation_complete": (
                    unannotated_runs == 0 and bbox_missing_runs == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _attach_annotation_validity(
    predictions: pd.DataFrame,
    annotation_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-region validity without treating unreviewed zeros as negative."""
    merged = predictions.merge(
        annotation_coverage,
        on=["study_id", "level"],
        how="left",
        validate="one_to_one",
    )
    annotated = merged["has_region_target"].astype(bool)
    if merged.loc[annotated, "annotation_complete"].isna().any():
        raise ValueError("Annotation coverage is missing for an annotated bag")
    merged["annotation_complete"] = (
        merged["annotation_complete"].astype("boolean").fillna(False).astype(bool)
    )
    for region_column in REGION_COLUMNS:
        merged[f"{region_column}_target_valid"] = (
            merged[region_column].eq(1) | merged["annotation_complete"]
        ) & annotated
    return merged


def build_case_figure(
    row: dict[str, Any],
    ct: NDArray[np.uint8],
    whole_mask: NDArray[np.uint8],
    region_mask: NDArray[np.uint8],
    cams: NDArray[np.float32],
    plane_probabilities: NDArray[np.float32],
    metrics: dict[str, Any],
) -> Figure:
    """Render all 15 planes with CAM and anatomical boundaries."""
    normalized = cams / cams.max() if float(cams.max()) > 0 else cams
    plane_mass = cams.sum(axis=(1, 2), dtype=np.float64)
    plane_mass_fraction = (
        plane_mass / plane_mass.sum() if plane_mass.sum() > 0 else plane_mass
    )
    figure, axes = plt.subplots(3, 5, figsize=(16, 10))
    for plane_index, axis in enumerate(axes.ravel()):
        axis.imshow(ct[plane_index, 2], cmap="gray", vmin=0, vmax=255)
        axis.imshow(
            normalized[plane_index],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )
        axis.contour(
            whole_mask[plane_index] > 0, levels=[0.5], colors="white", linewidths=0.7
        )
        for region_value, color in enumerate(REGION_COLORS, start=1):
            region = region_mask[plane_index] == region_value
            if np.any(region):
                axis.contour(region, levels=[0.5], colors=color, linewidths=0.8)
        axis.set_title(
            f"面{plane_index:02d}  p={plane_probabilities[plane_index]:.2f}\n"
            f"CAM質量={plane_mass_fraction[plane_index]:.1%}",
            fontsize=8,
        )
        axis.axis("off")
    region_text = " / ".join(
        f"R{index + 1}={metrics[f'{column}_mass_fraction']:.1%}"
        for index, column in enumerate(REGION_COLUMNS)
    )
    figure.suptitle(
        f"{row['category']}  {row['study_id']} / {row['level']}  "
        f"OOF={float(row['vertebra_score']):.3f}  fold={int(row['fold'])}\n"
        f"椎体内CAM={metrics['in_vertebra_mass_fraction']:.1%} "
        f"(面積={metrics['vertebra_area_fraction']:.1%}, "
        f"密度={metrics['vertebra_density_enrichment']:.2f}倍)  {region_text}",
        fontsize=12,
    )
    legend = [
        Patch(color=color, label=f"R{index + 1}: {REGION_DISPLAY_NAMES[index]}")
        for index, color in enumerate(REGION_COLORS)
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    return figure


def build_summary_figure(
    summary: pd.DataFrame,
    overall: dict[str, Any],
) -> Figure:
    """Render category-level region mass and density comparisons."""
    categories = list(summary["category"].drop_duplicates())
    x_values = np.arange(len(REGION_COLUMNS), dtype=np.float32)
    width = 0.8 / max(len(categories), 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for category_index, category in enumerate(categories):
        values = summary[summary["category"].eq(category)].set_index("region")
        offset = (category_index - (len(categories) - 1) / 2) * width
        axes[0].bar(
            x_values + offset,
            values.loc[list(REGION_COLUMNS), "mass_fraction_mean"],
            width=width,
            label=category,
        )
        axes[1].bar(
            x_values + offset,
            values.loc[list(REGION_COLUMNS), "density_enrichment_mean"],
            width=width,
            label=category,
        )
    labels = [
        f"R{index + 1}\n{name}" for index, name in enumerate(REGION_DISPLAY_NAMES)
    ]
    axes[0].set_title("椎体内Grad-CAM質量の配分")
    axes[0].set_ylabel("CAM質量比")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("領域面積で補正したCAM密度")
    axes[1].set_ylabel("椎体内平均に対する密度比")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(x_values, labels)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    in_vertebra = ", ".join(
        f"{category}={values['mean_in_vertebra_mass_fraction']:.1%}/"
        f"{values['mean_vertebra_density_enrichment']:.2f}倍"
        for category, values in overall["categories"].items()
    )
    figure.suptitle(
        f"Baseline 0 OOF Grad-CAM（椎体内CAM質量/面積補正密度: {in_vertebra}）"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _select_figure_keys(
    selected: pd.DataFrame,
    figures_per_category: int,
) -> set[tuple[str, str]]:
    if figures_per_category < 0:
        raise ValueError("figures_per_category must not be negative")
    figures = (
        selected.sort_values(
            ["category", "vertebra_score", "level", "study_id"],
            ascending=[True, False, True, True],
        )
        .groupby("category", sort=True, as_index=False)
        .head(figures_per_category)
    )
    return {
        (str(row.study_id), str(row.level))
        for row in figures[["study_id", "level"]].itertuples(index=False)
    }


def _batches(
    rows: list[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _guard_output(output_dir: Path, overwrite: bool) -> None:
    expected = {
        "annotation_coverage.csv",
        "selected_samples.csv",
        "attention_metrics.csv",
        "region_summary.csv",
        "annotated_target_summary.csv",
        "annotated_localization_metrics.csv",
        "summary.json",
        "region_summary.png",
    }
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Analysis output already exists; pass --overwrite: {existing[0]}"
        )


def _figure_filename(row: dict[str, Any]) -> str:
    study_suffix = str(row["study_id"]).rsplit(".", maxsplit=1)[-1]
    return (
        f"{str(row['category']).lower()}_outer{int(row['fold'])}_"
        f"{row['level']}_{study_suffix}.png"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline 0の正式OOF checkpointでGrad-CAMを生成する"
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--selection",
        choices=("stratified-high-scores", "annotated"),
        default="stratified-high-scores",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=("TP", "FP", "FN", "TN"),
        default=("TP", "FP"),
    )
    parser.add_argument("--samples-per-stratum", type=int, default=1)
    parser.add_argument("--figures-per-category", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
