"""固定diagnostic subsetでのcollapse監視。

hard GTを使わない固定subsetで毎epoch測る領域間・領域-whole相関を追跡し、
3回連続でcollapse条件を満たしたら学習を停止させる（係数を調整して
再実行しない、事前定義した失敗として扱う）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fracture_detection.region_branch.data_pipeline.constants import (
    REGION_COLUMNS,
    REGION_TARGET_VALID_COLUMNS,
)

COLLAPSE_SPEARMAN_THRESHOLD = 0.95
COLLAPSE_CONSECUTIVE_EPOCHS = 3
STANDARD_DEVIATION_FLOOR = 1e-8


def select_diagnostic_subset(
    train_manifest: pd.DataFrame, size: int, seed: int
) -> pd.DataFrame:
    """train foldのwhole-positiveかつ人手未確定bagからseed固定で診断subsetを選ぶ。

    hard GTを使わない固定subsetという元の意図を、廃止された`SourcePools`に
    依存せず保つ: whole-positive、かつ4領域全てが人手確定済みではないbagだけを
    候補にする。
    """
    eligible = train_manifest[
        train_manifest["vertebra_target"].eq(1)
        & ~train_manifest[list(REGION_TARGET_VALID_COLUMNS)].all(axis=1)
    ]
    if len(eligible) < size:
        raise ValueError(
            f"診断subset用の候補がdiagnostic subset size未満です: "
            f"{len(eligible)} < {size}"
        )
    return eligible.sample(n=size, random_state=seed).reset_index(drop=True)


@dataclass(frozen=True)
class DiagnosticEpochRecord:
    """1 epochの固定subset診断値。"""

    epoch: int
    inter_region_spearman: dict[str, float]
    region_whole_spearman: dict[str, float]
    first_pc_explained_variance: float
    student_teacher_spearman: dict[str, float]

    def as_row(self) -> dict[str, Any]:
        """CSV1行分のflat dictへ変換する。"""
        row: dict[str, Any] = {
            "epoch": self.epoch,
            "first_pc_explained_variance": self.first_pc_explained_variance,
        }
        row.update(
            {
                f"spearman_{key}": value
                for key, value in self.inter_region_spearman.items()
            }
        )
        row.update(
            {
                f"spearman_{key}": value
                for key, value in self.region_whole_spearman.items()
            }
        )
        row.update(
            {
                f"student_teacher_spearman_{key}": value
                for key, value in self.student_teacher_spearman.items()
            }
        )
        return row


def compute_diagnostics(
    epoch: int,
    region_array: np.ndarray,
    whole_array: np.ndarray,
    pseudo_target_array: np.ndarray,
    pseudo_valid_array: np.ndarray,
    active_regions: tuple[int, ...],
) -> DiagnosticEpochRecord:
    """固定subsetのwhole/region bag logit（既に計算済み）からcollapse診断指標を計算する。

    Args:
        region_array: [N, |active_regions|] region bag logit。
        whole_array: [N] whole bag logit。
        pseudo_target_array: [N, |active_regions|] CAM soft pseudo target `q`。
        pseudo_valid_array: [N, |active_regions|] bool。そのcellにpseudo
            supervisionが実際に存在するか（`q`は0を正当な値として取り得るため、
            `q>0`のような値ベースの判定ではなく、この明示的なvalidity配列で
            除外セルを決める）。
        active_regions: `REGION_COLUMNS`への0-indexed参照。
    """
    if region_array.shape[0] != whole_array.shape[0]:
        raise ValueError("region_arrayとwhole_arrayの件数が一致しません")
    if region_array.shape[1] != len(active_regions):
        raise ValueError("region_arrayの列数がactive_regionsと一致しません")
    active_names = [REGION_COLUMNS[index] for index in active_regions]

    inter_region: dict[str, float] = {}
    for i in range(len(active_names)):
        for j in range(i + 1, len(active_names)):
            key = f"{active_names[i]}_vs_{active_names[j]}"
            inter_region[key] = float(
                spearmanr(region_array[:, i], region_array[:, j]).statistic
            )

    region_whole: dict[str, float] = {}
    for i, name in enumerate(active_names):
        region_whole[f"{name}_vs_whole"] = float(
            spearmanr(region_array[:, i], whole_array).statistic
        )

    standard_deviation = region_array.std(axis=0)
    standardized = (region_array - region_array.mean(axis=0)) / np.clip(
        standard_deviation, STANDARD_DEVIATION_FLOOR, None
    )
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigen_sum = float(eigenvalues.sum())
    explained_variance = (
        float(eigenvalues.max() / eigen_sum) if eigen_sum > 0 else float("nan")
    )

    student_teacher: dict[str, float] = {}
    for i, name in enumerate(active_names):
        column_scores = pseudo_target_array[:, i]
        # qは0を正当な値として取り得るため、`>0`ではなく明示的なvalidity配列で
        # 除外セルを決める（生CAM density時代のheuristicは新しいsoft targetに合わない）。
        finite = np.isfinite(column_scores) & pseudo_valid_array[:, i]
        if int(finite.sum()) >= 2:
            student_teacher[name] = float(
                spearmanr(region_array[finite, i], column_scores[finite]).statistic
            )
        else:
            student_teacher[name] = float("nan")

    return DiagnosticEpochRecord(
        epoch=epoch,
        inter_region_spearman=inter_region,
        region_whole_spearman=region_whole,
        first_pc_explained_variance=explained_variance,
        student_teacher_spearman=student_teacher,
    )


@dataclass
class CollapseMonitor:
    """領域間・領域-whole Spearman中央値が3epoch連続で閾値を超えたらalarmを出す。"""

    threshold: float = COLLAPSE_SPEARMAN_THRESHOLD
    required_consecutive: int = COLLAPSE_CONSECUTIVE_EPOCHS
    consecutive_collapsed_epochs: int = field(default=0, init=False)

    def update(self, record: DiagnosticEpochRecord) -> bool:
        """今epochの状態を反映し、alarmが確定したかを返す。"""
        inter_values = list(record.inter_region_spearman.values())
        whole_values = list(record.region_whole_spearman.values())
        inter_median = float(np.median(inter_values)) if inter_values else float("-inf")
        whole_median = float(np.median(whole_values)) if whole_values else float("-inf")
        collapsed_this_epoch = (
            inter_median >= self.threshold and whole_median >= self.threshold
        )
        self.consecutive_collapsed_epochs = (
            self.consecutive_collapsed_epochs + 1 if collapsed_this_epoch else 0
        )
        return self.consecutive_collapsed_epochs >= self.required_consecutive
