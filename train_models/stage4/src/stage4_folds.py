"""Immutable Stage4 study-fold loading and splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_STAGE4_FOLD_SHA256 = (
    "3f84f668070b952ce6b483cec1c288e5667d7fbcf53d550c71485369cd74e1c9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage4_fold_map(csv_path: Path) -> dict[str, int]:
    """Load the frozen study-to-fold map after verifying its exact hash."""
    actual_hash = _sha256(csv_path)
    print(f"[STAGE4 FOLDS] sha256={actual_hash}", flush=True)
    if actual_hash != EXPECTED_STAGE4_FOLD_SHA256:
        raise ValueError(
            "Stage4 fold manifest hash mismatch: "
            f"expected={EXPECTED_STAGE4_FOLD_SHA256} actual={actual_hash}"
        )

    frame = pd.read_csv(csv_path, dtype={"study_id": str})
    if list(frame.columns) != ["study_id", "fold"]:
        raise ValueError("Stage4 fold CSV must contain exactly study_id,fold")
    if frame["study_id"].duplicated().any():
        raise ValueError("Stage4 fold CSV contains duplicate study_id values")
    folds = frame["fold"].astype(int)
    if not folds.between(0, 4).all():
        raise ValueError("Stage4 folds must be integers in [0, 4]")
    return dict(zip(frame["study_id"], folds, strict=True))


def split_by_stage4_fold(
    items: list[dict[str, Any]],
    fold_map: dict[str, int],
    val_fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split bags by the frozen Stage4 study folds without a holdout."""
    if val_fold not in range(5):
        raise ValueError(f"val_fold must be in [0, 4], got {val_fold}")
    missing_studies = sorted(
        {
            str(item["study_uid"])
            for item in items
            if str(item["study_uid"]) not in fold_map
        }
    )
    if missing_studies:
        raise ValueError(
            f"{len(missing_studies)} item studies are absent from Stage4 folds: "
            f"{missing_studies[:5]}"
        )
    train_items = [
        item for item in items if fold_map[str(item["study_uid"])] != val_fold
    ]
    valid_items = [
        item for item in items if fold_map[str(item["study_uid"])] == val_fold
    ]
    return train_items, valid_items
