"""評価サンプルのヒートマップ画像保存。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.visualization import save_heatmap_grid, save_heatmap_overlay
from .model import VERTEBRA_TO_IDX

Batch = dict[str, Any]


def _vertebra_indices(batch: Batch, device: torch.device) -> torch.Tensor:
    """バッチの椎体名をモデル入力用インデックスへ変換する。"""
    return torch.as_tensor(
        [VERTEBRA_TO_IDX.get(vertebra, 0) for vertebra in batch["vertebra"]],
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def save_examples(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    out_dir: Path,
    n_save: int = 12,
    tag: str = "VAL",
) -> None:
    """GTと予測ヒートマップのグリッド・オーバーレイを保存する。"""
    model.eval()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for batch in loader:
        images = batch["image"].to(device).float()
        vertebra_indices = _vertebra_indices(batch, device)
        gt_heatmaps = batch["heatmaps"].to(device).float()
        pred_heatmaps = torch.sigmoid(model(images, vertebra_indices))

        images_numpy = images.cpu().numpy()
        gt_numpy = gt_heatmaps.cpu().numpy()
        pred_numpy = pred_heatmaps.cpu().numpy()

        for batch_index in range(images_numpy.shape[0]):
            ct_image = images_numpy[batch_index, 0]
            sample = batch["sample"][batch_index]
            vertebra = batch["vertebra"][batch_index]
            slice_index = int(batch["slice_idx"][batch_index])
            name = f"{sample}_{vertebra}_slice{slice_index:03d}"

            save_heatmap_grid(
                ct_image,
                gt_numpy[batch_index],
                output_dir / f"{tag}_{name}_GT_grid.png",
            )
            save_heatmap_grid(
                ct_image,
                pred_numpy[batch_index],
                output_dir / f"{tag}_{name}_PRED_grid.png",
            )
            save_heatmap_overlay(
                ct_image,
                gt_numpy[batch_index],
                output_dir / f"{tag}_{name}_GT_merged.png",
            )
            save_heatmap_overlay(
                ct_image,
                pred_numpy[batch_index],
                output_dir / f"{tag}_{name}_PRED_merged.png",
            )

            saved_count += 1
            if saved_count >= n_save:
                return
