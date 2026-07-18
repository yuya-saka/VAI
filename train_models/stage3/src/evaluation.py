"""Stage3 vertebra metrics and contextual-evidence artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from train_models.stage1.utils.metrics import compute_level_metrics, compute_oof_metrics

REGION_NAMES = ("body", "right_foramen", "left_foramen", "posterior")


def compute_metrics(
    labels: NDArray, probabilities: NDArray, groups: NDArray, vertebrae: NDArray
) -> dict[str, Any]:
    return {
        **compute_oof_metrics(labels, probabilities, groups),
        "per_vertebra": compute_level_metrics(labels, probabilities, vertebrae),
    }


def compute_prediction_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics from valid metadata-bearing prediction records."""
    valid_records = [
        record for record in records if bool(record.get("vertebra_valid", True))
    ]
    if not valid_records:
        return {}
    return compute_metrics(
        np.asarray([record["label"] for record in valid_records]),
        np.asarray([record["pred_prob"] for record in valid_records]),
        np.asarray([record["study_uid"] for record in valid_records]),
        np.asarray([record["vertebra"] for record in valid_records]),
    )


def save_evidence(path: Path, records: list[dict[str, Any]]) -> None:
    """Persist contextual evidence without pickle-dependent object arrays."""
    if not records:
        return
    keys = (
        "study_uid",
        "vertebra",
        "label",
        "fold",
        "vertebra_logit",
        "probability",
        "instance",
        "region",
        "attention",
        "slice",
        "region_weights",
        "region_plane_valid",
        "region_valid",
        "plane_valid",
        "vertebra_valid",
    )
    arrays: dict[str, NDArray] = {
        key: np.asarray([record[key] for record in records]) for key in keys
    }
    np.savez_compressed(path, schema_version=np.asarray(1), **arrays)


def concatenate_evidence(path: Path, sources: list[Path]) -> None:
    """Concatenate compatible per-fold evidence artifacts."""
    if not sources:
        return
    collected: dict[str, list[NDArray]] = {}
    expected_keys: tuple[str, ...] | None = None
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError(f"unsupported evidence schema: {source}")
            keys = tuple(key for key in data.files if key != "schema_version")
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"evidence key mismatch: {source}")
            for key in keys:
                collected.setdefault(key, []).append(data[key].copy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    np.savez_compressed(path, schema_version=np.asarray(1), **arrays)


def _finite_mean(values: NDArray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_median(values: NDArray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def compute_evidence_diagnostics(path: Path) -> dict[str, Any]:
    """Compute label-free collapse diagnostics from a saved evidence artifact."""
    with np.load(path, allow_pickle=False) as data:
        labels = data["label"].astype(bool)
        vertebra_valid = data["vertebra_valid"].astype(bool)
        instance = data["instance"].astype(np.float64)
        region = data["region"].astype(np.float64)
        attention = data["attention"].astype(np.float64)
        slice_evidence = data["slice"].astype(np.float64)
        region_weights = data["region_weights"].astype(np.float64)
        region_plane_valid = data["region_plane_valid"].astype(bool)
        region_valid = data["region_valid"].astype(bool)
        plane_valid = data["plane_valid"].astype(bool)
        vertebra_logit = data["vertebra_logit"].astype(np.float64)

    effective_regions = np.divide(
        1.0,
        np.square(region_weights).sum(axis=1),
        out=np.full(len(labels), np.nan),
        where=vertebra_valid,
    )
    effective_slices = np.divide(
        1.0,
        np.square(attention).sum(axis=1),
        out=np.full(region_valid.shape, np.nan),
        where=region_valid,
    )
    valid_slice_counts = region_plane_valid.sum(axis=1)
    entropy = -(
        np.where(
            attention > 0,
            attention * np.log(np.clip(attention, 1e-12, None)),
            0.0,
        )
    ).sum(axis=1)
    entropy_denominator = np.log(np.maximum(valid_slice_counts, 2))
    normalized_entropy = np.divide(
        entropy,
        entropy_denominator,
        out=np.full(region_valid.shape, np.nan),
        where=region_valid,
    )

    def summarize(mask: NDArray) -> dict[str, Any]:
        selected = mask & vertebra_valid
        winner = np.argmax(
            np.where(region_valid[selected], region[selected], -np.inf), axis=1
        )
        winner_counts = np.bincount(winner, minlength=len(REGION_NAMES))
        winner_distribution = (
            winner_counts / winner_counts.sum()
            if winner_counts.sum()
            else np.zeros(len(REGION_NAMES))
        )
        return {
            "n": int(selected.sum()),
            "effective_regions_mean": _finite_mean(effective_regions[selected]),
            "effective_regions_median": _finite_median(effective_regions[selected]),
            "max_region_weight_mean": _finite_mean(
                region_weights[selected].max(axis=1)
            ),
            "effective_slices_mean": _finite_mean(effective_slices[selected]),
            "effective_slices_median": _finite_median(effective_slices[selected]),
            "attention_entropy_normalized_mean": _finite_mean(
                normalized_entropy[selected]
            ),
            "max_attention_mean": _finite_mean(attention[selected].max(axis=1)),
            "region_winner_distribution": {
                name: float(value)
                for name, value in zip(REGION_NAMES, winner_distribution, strict=True)
            },
            "vertebra_logit_mean": _finite_mean(vertebra_logit[selected]),
            "instance_logit_mean": _finite_mean(
                instance[selected][region_plane_valid[selected]]
            ),
            "region_logit_mean": _finite_mean(region[selected][region_valid[selected]]),
            "slice_logit_mean": _finite_mean(
                slice_evidence[selected][plane_valid[selected]]
            ),
        }

    return {
        "all": summarize(np.ones(len(labels), dtype=bool)),
        "negative": summarize(~labels),
        "positive": summarize(labels),
        "per_region": {
            name: {
                "valid_rate": float(region_valid[:, index].mean()),
                "effective_slices_mean": _finite_mean(effective_slices[:, index]),
                "attention_entropy_normalized_mean": _finite_mean(
                    normalized_entropy[:, index]
                ),
            }
            for index, name in enumerate(REGION_NAMES)
        },
    }
