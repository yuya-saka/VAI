"""region branchが学習できているかを見るための診断専用指標。

RSNA修正方針§10「評価も修正する」に対応する。train/val両方のregion APを
毎epoch記録し、trainも低ければ未学習、trainだけ高ければ過学習、
wholeだけ改善していればnegative transfer、と切り分けられるようにする。
`common/`のregion floor・Holm補正等の正式endpoint計算とは独立した、
epochごとの軽量診断であり、macro平均もここでは診断目的にのみ使う
（正式endpointの母集団・複雑度は`common/metrics.py::region_metrics`が担う）。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fracture_detection.common.constants import REGION_COLUMNS
from fracture_detection.common.metrics import safe_average_precision
from fracture_detection.core.steps import ArmAdapter, prepare_batch


@torch.no_grad()
def region_predictions(
    model: nn.Module,
    adapter: ArmAdapter,
    loader: DataLoader[Any],
    device: torch.device,
) -> pd.DataFrame:
    """region scoreだけを集めた予測表を返す（vertebra側の集計はしない）。

    `core.trainer.evaluate`はvertebra targetに両クラスが存在する前提で
    pos/neg score平均を計算する。region-annotated bagだけを渡す
    train診断passにそのまま使うと、268 bag全例がvertebra陽性
    （`vertebra_target==1`）なので`scores[targets==0]`が空配列になり、
    `RuntimeWarning: Mean of empty slice`が毎epoch出る。ここではregion
    診断に不要なvertebra側の統計を最初から計算しない。
    """
    model.eval()
    records: list[dict[str, float | str | bool]] = []
    for batch in loader:
        prepared = prepare_batch(batch, device, adapter.input_channels)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = adapter.forward(model, prepared.inputs)
        if output.region_logits is None:
            raise ValueError("modelがregion logitsを返しません")
        if prepared.region_targets is None or prepared.region_target_valid is None:
            raise ValueError("batchにregion target/validがありません")
        region_scores = output.region_logits.sigmoid().mean(dim=1).float().cpu()
        region_targets = prepared.region_targets.float().cpu()
        region_valid = prepared.region_target_valid.bool().cpu()
        study_ids = _strings(batch, "study_id")
        levels = _strings(batch, "level")
        for index, (study_id, level) in enumerate(zip(study_ids, levels, strict=True)):
            record: dict[str, float | str | bool] = {
                "study_id": study_id,
                "level": level,
                "has_region_target": bool(region_valid[index].all()),
            }
            for region_index, column in enumerate(REGION_COLUMNS):
                record[f"{column}_score"] = float(region_scores[index, region_index])
                record[f"{column}_target"] = float(region_targets[index, region_index])
            records.append(record)
    if not records:
        raise ValueError("region predictions loaderが空です")
    return pd.DataFrame(records)


def _strings(batch: dict[str, object], key: str) -> list[str]:
    values = batch.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"batchの{key}は文字列listである必要があります")
    return values


def region_average_precision(predictions: pd.DataFrame) -> dict[str, float]:
    """`has_region_target`行だけを使い領域別・macro APを返す（bootstrapなし）。"""
    missing = {"has_region_target"} | {
        f"{column}_{suffix}"
        for column in REGION_COLUMNS
        for suffix in ("target", "score")
    }
    missing -= set(predictions.columns)
    if missing:
        raise ValueError(f"予測表に必要な列がありません: {sorted(missing)}")
    annotated = predictions[predictions["has_region_target"].astype(bool)]
    result: dict[str, float] = {}
    for column in REGION_COLUMNS:
        if annotated.empty:
            result[column] = math.nan
            continue
        result[column] = safe_average_precision(
            annotated[f"{column}_target"].to_numpy(),
            annotated[f"{column}_score"].to_numpy(),
        )
    finite = [value for value in result.values() if math.isfinite(value)]
    result["macro"] = float(sum(finite) / len(finite)) if finite else math.nan
    return result
