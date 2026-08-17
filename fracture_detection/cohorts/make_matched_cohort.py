"""Baseline 1/2で共有する固定matchedコホートを生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fracture_detection.cohorts.constants import (
    ANNOTATED_ROLE,
    COHORT_ROLE_COLUMN,
    EXPECTED_ANNOTATED_ROWS,
    INPUT_MANIFEST_CSV,
    MATCHED_COHORT_CSV,
    MATCHED_COHORT_META_JSON,
    NEGATIVE_ROLE,
    OUTPUT_DIR,
    SEED,
    TRAIN_CSV,
)
from fracture_detection.common.constants import MANIFEST_COLUMNS

COHORT_COLUMNS = (*MANIFEST_COLUMNS, COHORT_ROLE_COLUMN)


def _as_bool(values: pd.Series) -> pd.Series:
    """boolまたはCSV由来のbool文字列を厳密にboolへ変換する。"""
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        invalid_values = sorted(normalized[invalid].unique().tolist())
        raise ValueError(f"has_region_targetに不正な値があります: {invalid_values}")
    return normalized.eq("true")


def _stable_rank(seed: int, fold: int, level: str, study_id: str) -> str:
    """同点を解消するための固定ハッシュ順位を返す。"""
    text = f"{seed}|{fold}|{level}|{study_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """必要列が不足していれば失敗する。"""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}に必要な列がありません: {missing}")


def _target_negative_rows(manifest: pd.DataFrame, annotated_rows: int) -> int:
    """fullの陽性率を再現するために必要な陰性bag数を返す。"""
    positive_rows = int(manifest["vertebra_target"].eq(1).sum())
    negative_rows = int(manifest["vertebra_target"].eq(0).sum())
    if positive_rows == 0 or negative_rows == 0:
        raise ValueError("input manifestには陽性bagと陰性bagの両方が必要です")
    return int(round(annotated_rows * negative_rows / positive_rows))


def _negative_requirements(
    manifest: pd.DataFrame,
    annotated_rows: int,
) -> pd.Series:
    """fullの陰性fold・level分布に比例した抽出件数を返す。"""
    negative = manifest[manifest["vertebra_target"].eq(0)]
    available = negative.groupby(["fold", "level"], sort=True).size()
    target_rows = _target_negative_rows(manifest, annotated_rows)
    if target_rows > int(available.sum()):
        raise ValueError("fullの陽性率を再現するための陰性bagが不足しています")

    exact = available.astype(float) * target_rows / int(available.sum())
    requirements = exact.astype(int)
    remaining = target_rows - int(requirements.sum())
    ranked_cells = sorted(
        available.index,
        key=lambda cell: (-(exact.loc[cell] - requirements.loc[cell]), cell),
    )
    for cell in ranked_cells[:remaining]:
        requirements.loc[cell] += 1
    if (requirements > available).any():
        raise ValueError("fold・level別の陰性bagが不足しています")
    return requirements.astype(int)


def _select_negative_bags(
    manifest: pd.DataFrame,
    requirements: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """各fold・levelから固定ハッシュ順で陰性bagを抽出する。"""
    candidates = manifest[manifest["vertebra_target"].eq(0)].copy()
    selected_indices: list[int] = []
    for (fold, level), required_rows in requirements.items():
        cell = candidates[
            candidates["fold"].eq(int(fold)) & candidates["level"].eq(str(level))
        ].copy()
        cell["_rank"] = [
            _stable_rank(seed, int(fold), str(level), str(study_id))
            for study_id in cell["study_id"]
        ]
        selected_indices.extend(
            cell.sort_values(["_rank", "study_id"]).head(int(required_rows)).index
        )
    return candidates.loc[selected_indices].copy()


def select_matched_cohort(
    manifest: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """アノテーション済み陽性bagとfull相当分布の陰性bagを選ぶ。"""
    _require_columns(manifest, MANIFEST_COLUMNS, "input manifest")

    source = manifest.copy()
    source["study_id"] = source["study_id"].astype(str)
    source["level"] = source["level"].astype(str)
    source["has_region_target"] = _as_bool(source["has_region_target"])
    if source.duplicated(["study_id", "level"]).any():
        raise ValueError("input manifestに重複したstudy_id・levelがあります")

    annotated = source[source["has_region_target"]].copy()
    if annotated.empty:
        raise ValueError("領域アノテーション済みbagがありません")
    if not annotated["vertebra_target"].eq(1).all():
        raise ValueError("アノテーション済みbagに椎体陰性が含まれています")

    requirements = _negative_requirements(source, len(annotated))
    negative = _select_negative_bags(source, requirements, seed)

    annotated_output = annotated.assign(**{COHORT_ROLE_COLUMN: ANNOTATED_ROLE})
    negative_output = negative.assign(**{COHORT_ROLE_COLUMN: NEGATIVE_ROLE})
    cohort = pd.concat([annotated_output, negative_output], ignore_index=True)
    cohort = cohort[list(COHORT_COLUMNS)].sort_values(
        ["study_id", "level", COHORT_ROLE_COLUMN]
    )
    return cohort.reset_index(drop=True)


def validate_matched_cohort(
    cohort: pd.DataFrame,
    source_manifest: pd.DataFrame,
    expected_annotated_rows: int | None = None,
) -> dict[str, int]:
    """matchedコホートがfull相当の分布契約を満たすことを検証する。"""
    _require_columns(cohort, COHORT_COLUMNS, "matched cohort")
    _require_columns(source_manifest, MANIFEST_COLUMNS, "input manifest")
    if cohort.duplicated(["study_id", "level"]).any():
        raise ValueError("matched cohortに重複したstudy_id・levelがあります")

    checked = cohort.copy()
    checked["study_id"] = checked["study_id"].astype(str)
    checked["level"] = checked["level"].astype(str)
    checked["has_region_target"] = _as_bool(checked["has_region_target"])
    roles = set(checked[COHORT_ROLE_COLUMN])
    if roles != {ANNOTATED_ROLE, NEGATIVE_ROLE}:
        raise ValueError(f"cohort_roleが不正です: {sorted(roles)}")

    annotated = checked[checked[COHORT_ROLE_COLUMN].eq(ANNOTATED_ROLE)]
    negative = checked[checked[COHORT_ROLE_COLUMN].eq(NEGATIVE_ROLE)]
    if (
        expected_annotated_rows is not None
        and len(annotated) != expected_annotated_rows
    ):
        raise ValueError(
            f"アノテーション済みbag数が想定外です: {len(annotated)} != {expected_annotated_rows}"
        )
    if (
        not annotated["has_region_target"].all()
        or not annotated["vertebra_target"].eq(1).all()
    ):
        raise ValueError("アノテーション済みbagのラベル契約に違反しています")
    if (
        negative["has_region_target"].any()
        or not negative["vertebra_target"].eq(0).all()
    ):
        raise ValueError("陰性bagのラベル契約に違反しています")
    expected_negative_rows = _target_negative_rows(source_manifest, len(annotated))
    if len(negative) != expected_negative_rows:
        raise ValueError(
            "陰性bag数がfullの陽性率に対応していません: "
            f"{len(negative)} != {expected_negative_rows}"
        )
    expected = _negative_requirements(source_manifest, len(annotated)).sort_index()
    actual = negative.groupby(["fold", "level"]).size().sort_index()
    if not expected.equals(actual):
        raise ValueError("陰性bagのfold・level分布がfullと一致しません")

    return {
        "rows": int(len(checked)),
        "annotated_rows": int(len(annotated)),
        "negative_rows": int(len(negative)),
        "studies": int(checked["study_id"].nunique()),
    }


def _sha256(path: Path) -> str:
    """ファイルのSHA256を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(
    cohort: pd.DataFrame,
    source_manifest: pd.DataFrame,
    manifest_path: Path,
    train_csv: Path,
    cohort_sha256: str,
) -> dict[str, float | int | str]:
    """コホートを再現・監査するためのメタデータを構築する。"""
    role_counts = Counter(cohort[COHORT_ROLE_COLUMN])
    return {
        "rows": int(len(cohort)),
        "studies": int(cohort["study_id"].nunique()),
        "annotated_rows": int(role_counts[ANNOTATED_ROLE]),
        "negative_rows": int(role_counts[NEGATIVE_ROLE]),
        "cohort_prevalence": float(cohort["vertebra_target"].mean()),
        "source_prevalence": float(source_manifest["vertebra_target"].mean()),
        "seed": SEED,
        "input_manifest_sha256": _sha256(manifest_path),
        "train_csv_sha256": _sha256(train_csv),
        "sha256": cohort_sha256,
    }


def write_frozen_cohort(
    cohort: pd.DataFrame,
    metadata: dict[str, float | int | str],
    output_dir: Path,
) -> tuple[Path, Path]:
    """コホートを固定出力し、異なる再生成を拒否する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / MATCHED_COHORT_CSV.name
    metadata_path = output_dir / MATCHED_COHORT_META_JSON.name
    cohort_bytes = cohort.to_csv(index=False).encode("utf-8")
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    existing = [path.exists() for path in (cohort_path, metadata_path)]
    if any(existing):
        if not all(existing):
            raise SystemExit(
                "matched cohortの凍結成果物が不完全です。手動で状態を確認してください"
            )
        if (
            cohort_path.read_bytes() != cohort_bytes
            or metadata_path.read_bytes() != metadata_bytes
        ):
            raise SystemExit(
                "内容の異なるmatched cohortがすでに存在します。cohortは凍結済みです。"
                "まだどの実験にも使用していない場合に限り、手動で状態を確認してください"
            )
        return cohort_path, metadata_path

    temporary_cohort = cohort_path.with_suffix(".csv.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_cohort.write_bytes(cohort_bytes)
    temporary_metadata.write_bytes(metadata_bytes)
    temporary_cohort.replace(cohort_path)
    temporary_metadata.replace(metadata_path)
    return cohort_path, metadata_path


def build_matched_cohort(
    manifest_path: Path = INPUT_MANIFEST_CSV,
    train_csv: Path = TRAIN_CSV,
    expected_annotated_rows: int | None = EXPECTED_ANNOTATED_ROWS,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """固定入力から検証済みmatchedコホートとメタデータを作る。"""
    manifest = pd.read_csv(manifest_path, dtype={"study_id": str, "level": str})
    cohort = select_matched_cohort(manifest)
    validate_matched_cohort(cohort, manifest, expected_annotated_rows)
    cohort_sha256 = hashlib.sha256(
        cohort.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    metadata = _metadata(cohort, manifest, manifest_path, train_csv, cohort_sha256)
    return cohort, metadata


def parse_args() -> argparse.Namespace:
    """CLI引数を解釈する。"""
    parser = argparse.ArgumentParser(description="固定matched cohortを生成します")
    parser.add_argument("--manifest", type=Path, default=INPUT_MANIFEST_CSV)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLIからmatchedコホートを生成または検証する。"""
    args = parse_args()
    cohort, metadata = build_matched_cohort(args.manifest, args.train_csv)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return
    cohort_path, metadata_path = write_frozen_cohort(cohort, metadata, args.output_dir)
    print(f"matched cohortを確認しました: {cohort_path}")
    print(f"メタデータを確認しました: {metadata_path}")


if __name__ == "__main__":
    main()
