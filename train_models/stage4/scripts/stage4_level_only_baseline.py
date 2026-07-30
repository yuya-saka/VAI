"""Compute the frozen-fold Stage4 level-only pooled OOF baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_models.stage4.src.region_labels import load_region_labels  # noqa: E402
from train_models.stage4.src.stage4_folds import load_stage4_fold_map  # noqa: E402

REGION_NAMES = ("R1", "R2", "R3", "R4")


def compute_level_only_oof(
    region_labels_path: Path,
    folds_path: Path,
) -> list[dict[str, Any]]:
    """Predict each annotated bag from training-fold level prevalences."""
    labels = load_region_labels(region_labels_path)
    fold_map = load_stage4_fold_map(folds_path)
    missing = sorted({study_id for study_id, _ in labels if study_id not in fold_map})
    if missing:
        raise ValueError(f"annotated studies absent from Stage4 folds: {missing[:5]}")

    records: list[dict[str, Any]] = []
    for fold in range(5):
        train_keys = [key for key in labels if fold_map[key[0]] != fold]
        valid_keys = [key for key in labels if fold_map[key[0]] == fold]
        level_rates: dict[str, np.ndarray] = {}
        for level in sorted({level for _, level in train_keys}):
            level_targets = np.stack(
                [labels[key] for key in train_keys if key[1] == level]
            )
            level_rates[level] = level_targets.mean(axis=0)
        for key in valid_keys:
            if key[1] not in level_rates:
                raise ValueError(f"fold {fold} has no training labels for {key[1]}")
            records.append(
                {
                    "study_id": key[0],
                    "level": key[1],
                    "fold": fold,
                    "target": labels[key].copy(),
                    "probability": level_rates[key[1]].copy(),
                }
            )
    if len(records) != len(labels):
        raise RuntimeError("level-only OOF did not cover every annotated bag")
    return records


def summarize_macro_ap(
    records: list[dict[str, Any]],
    exclude_c2: bool = False,
) -> dict[str, Any]:
    """Compute pooled OOF AP per region and their macro mean."""
    selected = [
        record for record in records if not (exclude_c2 and record["level"] == "C2")
    ]
    targets = np.stack([record["target"] for record in selected])
    probabilities = np.stack([record["probability"] for record in selected])
    per_region = {
        name: float(average_precision_score(targets[:, index], probabilities[:, index]))
        for index, name in enumerate(REGION_NAMES)
    }
    return {
        "n_bags": len(selected),
        "per_region_ap": per_region,
        "macro_ap": float(np.mean(list(per_region.values()))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region-labels",
        type=Path,
        default=ROOT / "data/rsna_data/fracture_region_labels_dicom.csv",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=ROOT / "data/rsna_data/stage4_folds.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    records = compute_level_only_oof(arguments.region_labels, arguments.folds)
    report = {
        "all_268": summarize_macro_ap(records),
        "excluding_c2_231": summarize_macro_ap(records, exclude_c2=True),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
