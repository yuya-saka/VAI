"""Fold-matched teacher CAMスコアと領域温度の読み込み。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from fracture_detection.region_branch.data_pipeline.constants import (
    PSEUDO_SCORES_CSV,
    PSEUDO_TEMPERATURES_CSV,
    REGION_COLUMNS,
    REGION_SCORE_COLUMNS,
)


def load_pseudo_scores(pseudo_label_dir: Path, student_outer_fold: int) -> pd.DataFrame:
    """student_outer_fold==kの学習に使うTeacher_kのスコアだけを返す。

    teacher_outer_fold==kの行は、Teacher_kが自分の学習fold（=Student_kの学習fold
    と同一）を採点した結果であり、Student_kが参照してよい唯一のスコアである。
    """
    scores_path = pseudo_label_dir / PSEUDO_SCORES_CSV
    if not scores_path.is_file():
        raise FileNotFoundError(f"pseudo scoreがありません: {scores_path}")
    frame = pd.read_csv(scores_path, dtype={"study_id": str, "level": str})
    missing = {"study_id", "level", "teacher_outer_fold", *REGION_SCORE_COLUMNS} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"pseudo scoreに必要な列がありません: {sorted(missing)}")
    matched = frame[frame["teacher_outer_fold"].eq(student_outer_fold)].reset_index(
        drop=True
    )
    if matched.empty:
        raise ValueError(
            f"student_outer_fold={student_outer_fold}に対応するteacherスコアがありません"
        )
    if matched.duplicated(["study_id", "level"]).any():
        raise ValueError(
            f"student_outer_fold={student_outer_fold}のteacherスコアに重複があります"
        )
    return matched[["study_id", "level", *REGION_SCORE_COLUMNS]]


def load_pseudo_temperatures(pseudo_label_dir: Path, student_outer_fold: int) -> Tensor:
    """Teacher_kの領域別温度をREGION_COLUMNS順の4-vectorとして返す。"""
    temperatures_path = pseudo_label_dir / PSEUDO_TEMPERATURES_CSV
    if not temperatures_path.is_file():
        raise FileNotFoundError(f"pseudo温度がありません: {temperatures_path}")
    frame = pd.read_csv(temperatures_path)
    missing = {"teacher_outer_fold", "region", "population", "temperature"} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"pseudo温度に必要な列がありません: {sorted(missing)}")
    matched = frame[
        frame["teacher_outer_fold"].eq(student_outer_fold)
        & frame["population"].eq("fracture_positive")
    ]
    values: list[float] = []
    for region in REGION_COLUMNS:
        row = matched[matched["region"].eq(region)]
        if len(row) != 1:
            raise ValueError(
                f"student_outer_fold={student_outer_fold}, region={region}の"
                f"温度が一意に定まりません: {len(row)}件"
            )
        values.append(float(row["temperature"].iloc[0]))
    return torch.tensor(values, dtype=torch.float32)


def attach_teacher_scores(
    manifest: pd.DataFrame, pseudo_scores: pd.DataFrame
) -> pd.DataFrame:
    """manifestへteacherスコア列を左結合する。未スコアのbagはNaNのままにする。"""
    required = {"study_id", "level"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifestに必要な列がありません: {sorted(required)}")
    merged = manifest.merge(pseudo_scores, on=["study_id", "level"], how="left")
    if len(merged) != len(manifest):
        raise ValueError("teacherスコア結合でmanifestの行数が変化しました")
    return merged
