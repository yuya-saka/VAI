"""
設定読み込み・シード設定・実効 config 保存・CV 分割ユーティリティ。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from learning.src.dataset import collect_bags


def set_seed(seed: int = 42) -> None:
    """再現性のためにグローバル乱数シードを設定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str | Path) -> dict:
    """YAML 設定ファイルを dict として読み込む。"""
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_effective_config(cfg: dict, output_dir: Path) -> Path:
    """
    CLI 上書きを反映した実効 config を output_dir/config.yaml に保存する。

    学習のたびに呼び出すことで、各 run の設定を永続化・再現可能にする。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "config.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] 実効 config を保存しました: {output_path}")
    return output_path


def split_bags_cv(
    bags: list[dict],
    n_splits: int = 5,
    val_fold: int = 0,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    全bagを患者単位でtrain/valに分割する。

    Returns:
        (train_bags, val_bags)
    """
    if not 0 <= val_fold < n_splits:
        raise ValueError(f"val_foldは0〜{n_splits - 1}で指定してください")

    labels = np.array([b["label"] for b in bags])
    groups = np.array([b["patient_id"] for b in bags])
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_idx, val_idx = list(splitter.split(bags, labels, groups))[val_fold]
    return [bags[i] for i in train_idx], [bags[i] for i in val_idx]


def collect_bags_from_cfg(dataset_dir: Path) -> list[dict]:
    """collect_bags のラッパー（train.py からの呼び出し用）。"""
    return collect_bags(dataset_dir)
