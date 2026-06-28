"""除外リスト（study / level）の読み込み。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .constants import RSNA_DATA_DIR


def load_excluded_studies(
    csv_path: Path | None = None,
) -> set[str]:
    """excluded_studies.csv から除外 study UID のセットを返す。"""
    if csv_path is None:
        csv_path = RSNA_DATA_DIR / "excluded_studies.csv"
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8") as f:
        return {row["study_uid"] for row in csv.DictReader(f)}


def load_excluded_levels(
    csv_path: Path | None = None,
) -> dict[str, set[str]]:
    """excluded_levels.csv から {study_uid: {vertebra, ...}} を返す。"""
    if csv_path is None:
        csv_path = RSNA_DATA_DIR / "excluded_levels.csv"
    if not csv_path.exists():
        return {}
    result: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["study_uid"]].add(row["vertebra"])
    return dict(result)
