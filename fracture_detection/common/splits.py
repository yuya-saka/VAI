"""凍結5-foldをnested選択用の役割へ安全に割り当てる。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NestedFoldSplit:
    """1 outer runで使うfoldの役割。"""

    outer_fold: int
    inner_fold: int
    train_folds: tuple[int, ...]


def resolve_nested_folds(outer_fold: int, n_folds: int = 5) -> NestedFoldSplit:
    """outer、cyclic inner、残りの学習foldを返す。"""
    if n_folds != 5:
        raise ValueError("凍結済み研究契約ではn_folds=5が必要です")
    if outer_fold not in range(n_folds):
        raise ValueError(f"outer foldが不正です: {outer_fold}")
    inner_fold = (outer_fold + 1) % n_folds
    train_folds = tuple(
        fold for fold in range(n_folds) if fold not in {outer_fold, inner_fold}
    )
    return NestedFoldSplit(outer_fold, inner_fold, train_folds)


def split_nested_manifest(
    manifest: pd.DataFrame,
    outer_fold: int,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """manifestを3 training folds、inner、outerへ分割する。"""
    required = {"study_id", "level", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("manifestが空です")
    if manifest.duplicated(["study_id", "level"]).any():
        raise ValueError("manifestに重複したstudy_id・levelがあります")
    study_fold_counts = manifest.groupby("study_id")["fold"].nunique()
    if not study_fold_counts.eq(1).all():
        raise ValueError("同一studyが複数foldへ割り当てられています")
    observed_folds = set(manifest["fold"].astype(int).unique())
    if observed_folds != set(range(n_folds)):
        raise ValueError(f"manifestのfold集合が不正です: {sorted(observed_folds)}")

    assignment = resolve_nested_folds(outer_fold, n_folds)
    train = manifest[manifest["fold"].isin(assignment.train_folds)].copy()
    inner = manifest[manifest["fold"].eq(assignment.inner_fold)].copy()
    outer = manifest[manifest["fold"].eq(assignment.outer_fold)].copy()
    frames = [frame.reset_index(drop=True) for frame in (train, inner, outer)]
    if any(frame.empty for frame in frames):
        raise ValueError("train、inner、outerのいずれかが空です")
    if sum(len(frame) for frame in frames) != len(manifest):
        raise ValueError("nested分割でmanifestの行が欠落または重複しました")
    return frames[0], frames[1], frames[2]
