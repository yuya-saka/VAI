"""補助region batch用のhuman / whole-negative / pseudoへの3分割。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SourcePools:
    """1 outer foldのtrain manifestを3つの重複なしプールへ分けた結果。"""

    human: pd.DataFrame
    negative: pd.DataFrame
    pseudo: pd.DataFrame


def split_source_pools(train_manifest: pd.DataFrame) -> SourcePools:
    """train manifestをhuman-annotated / whole-negative / pseudo-positiveへ分ける。

    3集団は次の条件で重複なく完全分割される:
    - human: has_region_target
    - negative: not has_region_target and vertebra_target == 0
    - pseudo: not has_region_target and vertebra_target == 1
    """
    required = {"has_region_target", "vertebra_target"}
    missing = required - set(train_manifest.columns)
    if missing:
        raise ValueError(f"train manifestに必要な列がありません: {sorted(missing)}")

    annotated = train_manifest["has_region_target"].astype(bool)
    positive = train_manifest["vertebra_target"].eq(1)

    human = train_manifest[annotated].reset_index(drop=True)
    negative = train_manifest[~annotated & ~positive].reset_index(drop=True)
    pseudo = train_manifest[~annotated & positive].reset_index(drop=True)

    total = len(human) + len(negative) + len(pseudo)
    if total != len(train_manifest):
        raise ValueError(
            f"3プールの合計がtrain manifestと一致しません: {total} != {len(train_manifest)}"
        )
    if human.empty or negative.empty or pseudo.empty:
        raise ValueError("human / negative / pseudoのいずれかのプールが空です")
    return SourcePools(human=human, negative=negative, pseudo=pseudo)
