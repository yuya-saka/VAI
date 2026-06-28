"""モデル読み込みと学習データ統計の計算。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from line_only.src.model import TinyUNet
from line_only.utils.detection import line_extent

from .constants import FALLBACK_LINE_LENGTH_PX, LINE_KEYS


def load_models(
    ckpt_dir: Path,
    n_folds: int,
    device: torch.device,
) -> list[TinyUNet]:
    """全 fold の TinyUNet をロードする。"""
    models: list[TinyUNet] = []
    for fold in range(n_folds):
        p = ckpt_dir / f"best_fold{fold}.pt"
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location=device, weights_only=True)
        cfg = ckpt.get("cfg", {})
        mc = cfg.get("model", {})
        m = TinyUNet(
            in_ch=int(mc.get("in_channels", 2)),
            out_ch=int(mc.get("out_channels", 4)),
            feats=tuple(mc.get("features", [16, 32, 64, 128])),
            dropout=0.0,
            num_vertebra=int(mc.get("num_vertebra", 7))
            if mc.get("use_vertebra_conditioning", False)
            else 0,
        ).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def compute_avg_line_lengths(dataset_dir: Path) -> dict[str, float]:
    """学習データから各線の平均長さ (px) を計算する。"""
    sums: dict[str, float] = {k: 0.0 for k in LINE_KEYS}
    counts: dict[str, int] = {k: 0 for k in LINE_KEYS}
    for lines_path in sorted(dataset_dir.glob("sample*/C*/lines.json")):
        try:
            data = json.loads(lines_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for slice_data in data.values():
            if not isinstance(slice_data, dict):
                continue
            for key in LINE_KEYS:
                length = line_extent(slice_data.get(key))
                if length > 1e-6:
                    sums[key] += length
                    counts[key] += 1
    return {
        k: sums[k] / counts[k] if counts[k] > 0 else FALLBACK_LINE_LENGTH_PX
        for k in LINE_KEYS
    }
