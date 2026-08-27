"""Baseline 0 checkpointからregion branch modelを初期化する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fracture_detection.region_branch.modeling.model import (
    RegionBranchModel,
    build_model,
)

BASELINE_PREFIX_MAP = {
    "encoder.": "encoder.",
    "lstm.": "whole_lstm.",
    "head.": "whole_head.",
}
PRETRAINED_PREFIXES = ("encoder.", "whole_lstm.", "whole_head.")
REGION_PREFIXES = ("fpn.", "region_lstm.", "region_heads.")


@dataclass(frozen=True)
class InitializationReport:
    """fold-matched初期化の監査情報。"""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_role: str
    outer_fold: int
    loaded_key_count: int
    random_key_count: int
    loaded_prefixes: tuple[str, ...]
    random_prefixes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON保存可能な辞書へ変換する。"""
        return asdict(self)


def build_initialized_model(
    config: dict[str, Any],
) -> tuple[RegionBranchModel, InitializationReport]:
    """modelを構築し、対応outer foldのBaseline 0重みを読み込む。"""
    model = build_model(config)
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("fold-matched初期化にはconfig.runtimeが必要です")
    outer_fold = runtime.get("outer_fold")
    if not isinstance(outer_fold, int) or isinstance(outer_fold, bool):
        raise ValueError("runtime.outer_foldは整数である必要があります")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("config.modelはmappingである必要があります")
    if model_config.get("initialization") != "baseline0_fold_matched":
        raise ValueError("model.initializationはbaseline0_fold_matchedが必要です")
    checkpoint_root = model_config.get("baseline0_checkpoint_root")
    if not isinstance(checkpoint_root, str) or not checkpoint_root:
        raise ValueError("model.baseline0_checkpoint_rootは非空文字列が必要です")
    checkpoint_path = Path(checkpoint_root) / f"outer{outer_fold}" / "best_model.pt"
    report = load_baseline0_weights(model, checkpoint_path, runtime)
    return model, report


def load_baseline0_weights(
    model: RegionBranchModel,
    checkpoint_path: Path,
    expected_runtime: dict[str, Any],
) -> InitializationReport:
    """Baseline 0のwhole model重みを対応するregion model moduleへ移す。"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline 0 checkpointがありません: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpointの形式が不正です: {checkpoint_path}")
    if checkpoint.get("checkpoint_role") != "best_val_auroc":
        raise ValueError("Baseline 0 checkpoint roleはbest_val_aurocが必要です")
    checkpoint_runtime = checkpoint.get("config", {}).get("runtime")
    if checkpoint_runtime != expected_runtime:
        raise ValueError(
            "Baseline 0 checkpointのnested fold設定がstudentと一致しません: "
            f"checkpoint={checkpoint_runtime}, student={expected_runtime}"
        )
    baseline_state = checkpoint.get("model")
    if not isinstance(baseline_state, dict):
        raise ValueError("Baseline 0 checkpointにmodel stateがありません")

    mapped_state = {
        _map_baseline_key(key): value for key, value in baseline_state.items()
    }
    model_state = model.state_dict()
    expected_loaded = {
        key for key in model_state if key.startswith(PRETRAINED_PREFIXES)
    }
    expected_random = {key for key in model_state if key.startswith(REGION_PREFIXES)}
    unknown_model_keys = set(model_state) - expected_loaded - expected_random
    if unknown_model_keys:
        raise ValueError(
            f"初期化分類できないregion model keyがあります: {sorted(unknown_model_keys)}"
        )
    if set(mapped_state) != expected_loaded:
        missing = sorted(expected_loaded - set(mapped_state))
        unexpected = sorted(set(mapped_state) - expected_loaded)
        raise ValueError(
            "Baseline 0とregion whole pathのkeyが一致しません: "
            f"missing={missing}, unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(mapped_state, strict=False)
    expected_missing = {
        key for key in expected_random if not key.endswith(".num_batches_tracked")
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "Baseline 0初期化後のstate契約が不正です: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    outer_fold = expected_runtime.get("outer_fold")
    if not isinstance(outer_fold, int):
        raise ValueError("expected_runtime.outer_foldは整数が必要です")
    return InitializationReport(
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_role="best_val_auroc",
        outer_fold=outer_fold,
        loaded_key_count=len(expected_loaded),
        random_key_count=len(expected_random),
        loaded_prefixes=PRETRAINED_PREFIXES,
        random_prefixes=REGION_PREFIXES,
    )


def save_initialization_report(report: InitializationReport, output_path: Path) -> None:
    """初期化監査情報を既存内容との一致を確認して保存する。"""
    serialized = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なる初期化reportがすでに存在します: {output_path}")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(output_path)


def _map_baseline_key(key: str) -> str:
    for baseline_prefix, region_prefix in BASELINE_PREFIX_MAP.items():
        if key.startswith(baseline_prefix):
            return region_prefix + key.removeprefix(baseline_prefix)
    raise ValueError(f"Baseline 0 checkpointに未知のmodel keyがあります: {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
