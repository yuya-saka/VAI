"""Fold一致teacherのCAM soft pseudo-region targetの読み込みと疑似対照生成。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fracture_detection.baseline0.data.splits import resolve_nested_folds
from fracture_detection.region_branch.data_pipeline.constants import (
    PSEUDO_REGION_TARGETS_CSV,
    REGION_PSEUDO_TARGET_COLUMNS,
    REGION_SHARE_COLUMNS,
)


def load_pseudo_region_targets(
    pseudo_label_dir: Path, student_outer_fold: int
) -> pd.DataFrame:
    """student_outer_fold==kのtrain/inner用CAM soft targetを返す。

    train bagはTeacher_k、inner bagはそのbagをheld outしたTeacher_innerが採点する。
    outer test bagのpseudo targetは含めない。
    """
    targets_path = pseudo_label_dir / PSEUDO_REGION_TARGETS_CSV
    if not targets_path.is_file():
        raise FileNotFoundError(f"pseudo targetがありません: {targets_path}")
    frame = pd.read_csv(targets_path, dtype={"study_id": str, "level": str})
    required = {
        "student_outer_fold",
        "study_id",
        "level",
        "teacher_outer_fold",
        *REGION_SHARE_COLUMNS,
        *REGION_PSEUDO_TARGET_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pseudo targetに必要な列がありません: {sorted(missing)}")

    matched = frame[frame["student_outer_fold"].eq(student_outer_fold)].reset_index(
        drop=True
    )
    if matched.empty:
        raise ValueError(
            f"student_outer_fold={student_outer_fold}に対応するpseudo targetがありません"
        )
    assignment = resolve_nested_folds(student_outer_fold)
    allowed_teachers = {student_outer_fold, assignment.inner_fold}
    observed_teachers = set(matched["teacher_outer_fold"].astype(int).unique())
    if not observed_teachers.issubset(allowed_teachers):
        raise ValueError(
            f"student_outer_fold={student_outer_fold}に不正なteacher foldが"
            f"含まれています: {sorted(observed_teachers - allowed_teachers)}"
        )
    if matched.duplicated(["study_id", "level"]).any():
        raise ValueError(
            f"student_outer_fold={student_outer_fold}のpseudo targetに重複があります"
        )
    return matched[
        [
            "study_id",
            "level",
            "teacher_outer_fold",
            *REGION_SHARE_COLUMNS,
            *REGION_PSEUDO_TARGET_COLUMNS,
        ]
    ]


def select_pseudo_targets_for_teacher(
    pseudo_targets: pd.DataFrame, teacher_outer_fold: int
) -> pd.DataFrame:
    """指定teacher foldが生成したtargetだけをbag key付きで返す。"""
    required = {
        "study_id",
        "level",
        "teacher_outer_fold",
        *REGION_SHARE_COLUMNS,
        *REGION_PSEUDO_TARGET_COLUMNS,
    }
    missing = required - set(pseudo_targets.columns)
    if missing:
        raise ValueError(f"pseudo targetに必要な列がありません: {sorted(missing)}")
    selected = pseudo_targets[
        pseudo_targets["teacher_outer_fold"].eq(teacher_outer_fold)
    ].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"teacher_outer_fold={teacher_outer_fold}のtargetがありません")
    if selected.duplicated(["study_id", "level"]).any():
        raise ValueError(
            f"teacher_outer_fold={teacher_outer_fold}のtargetに重複があります"
        )
    return selected[
        ["study_id", "level", *REGION_SHARE_COLUMNS, *REGION_PSEUDO_TARGET_COLUMNS]
    ]


def attach_pseudo_targets(
    manifest: pd.DataFrame, pseudo_targets: pd.DataFrame
) -> pd.DataFrame:
    """manifestへCAM share/pseudo target列を左結合する。

    未対象のbag（whole-negative、inner/outer fold等）はNaNのままにする。
    teacherはstudentの学習fold内whole-positive bagしか採点しないため、これは
    欠損ではなく想定される挙動である。
    """
    required = {"study_id", "level"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifestに必要な列がありません: {sorted(required)}")
    merged = manifest.merge(pseudo_targets, on=["study_id", "level"], how="left")
    if len(merged) != len(manifest):
        raise ValueError("pseudo target結合でmanifestの行数が変化しました")
    return merged


def shuffle_pseudo_target_associations(
    pseudo_targets: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """bag単位のCAM share/pseudo targetベクトル対応だけを固定seedで置換する。

    `cam_soft_shuffled`負対照が使う: target値の集合・有効セル母集団は保存し、
    どのbagがどのベクトルを受け取るかという対応関係だけをランダム化する。
    `study_id`/`level`と行順序は元のまま変えない。
    """
    columns = [*REGION_SHARE_COLUMNS, *REGION_PSEUDO_TARGET_COLUMNS]
    missing = set(columns) - set(pseudo_targets.columns)
    if missing:
        raise ValueError(f"pseudo targetに必要な列がありません: {sorted(missing)}")

    shuffled = pseudo_targets.reset_index(drop=True).copy()
    permutation = np.random.default_rng(seed).permutation(len(shuffled))
    shuffled.loc[:, columns] = (
        pseudo_targets.reset_index(drop=True)
        .loc[permutation, columns]
        .reset_index(drop=True)
    )
    return shuffled
