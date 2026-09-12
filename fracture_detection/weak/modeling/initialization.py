"""Transfer only the CNN trunk from a fold-matched Baseline 0 checkpoint.

Reference design: ``fracture_detection/REGION_MIL_DESIGN.md`` section 7. Only
``encoder.*`` is transferred; Baseline 0's ``lstm.*``/``head.*`` (its whole
BiLSTM and whole head) have no counterpart in ``WeakRegionMilModel`` and are
discarded -- FPN/region BiLSTM/region head are always randomly initialized.

Mirrors the checkpoint-validation and key-accounting pattern used by
``region_branch/modeling/initialization.py`` (fold-match / checkpoint-role
checks, atomic report save, hard-fail on any key mismatch), adapted to this
package's single-mode ("baseline0_encoder_transfer") initialization -- there
is no ``joint_from_start`` alternative here, since the design requires
CNN transfer from a fold-matched Baseline 0 checkpoint always.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fracture_detection.weak.modeling.model import WeakRegionMilModel

TRANSFERRED_PREFIX = "encoder."
NEW_PREFIXES = ("fpn.", "region_lstm.", "region_head.")
EXPECTED_CHECKPOINT_ROLE = "best_val_auroc"


@dataclass(frozen=True)
class InitializationReport:
    """Audit record of one model's checkpoint transfer."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_role: str
    outer_fold: int
    loaded_key_count: int
    random_key_count: int
    loaded_prefixes: tuple[str, ...]
    random_prefixes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


def load_baseline0_encoder(
    model: WeakRegionMilModel,
    checkpoint_path: Path,
    expected_runtime: dict[str, Any],
) -> InitializationReport:
    """Transfer ``encoder.*`` from a Baseline 0 checkpoint; randomize the rest."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline 0 checkpointがありません: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpointの形式が不正です: {checkpoint_path}")
    if checkpoint.get("checkpoint_role") != EXPECTED_CHECKPOINT_ROLE:
        raise ValueError(
            f"Baseline 0 checkpoint roleは{EXPECTED_CHECKPOINT_ROLE!r}が必要です"
        )
    checkpoint_runtime = checkpoint.get("config", {}).get("runtime")
    if checkpoint_runtime != expected_runtime:
        raise ValueError(
            "Baseline 0 checkpointのnested fold設定がstudentと一致しません: "
            f"checkpoint={checkpoint_runtime}, student={expected_runtime}"
        )
    baseline_state = checkpoint.get("model")
    if not isinstance(baseline_state, dict):
        raise ValueError("Baseline 0 checkpointにmodel stateがありません")

    encoder_state = {
        key: value
        for key, value in baseline_state.items()
        if key.startswith(TRANSFERRED_PREFIX)
    }
    if not encoder_state:
        raise ValueError("Baseline 0 checkpointにencoder.*のkeyがありません")

    model_state = model.state_dict()
    expected_loaded = {key for key in model_state if key.startswith(TRANSFERRED_PREFIX)}
    expected_random = {key for key in model_state if key.startswith(NEW_PREFIXES)}
    unknown_model_keys = set(model_state) - expected_loaded - expected_random
    if unknown_model_keys:
        raise ValueError(
            f"初期化分類できないmodel keyがあります: {sorted(unknown_model_keys)}"
        )
    if set(encoder_state) != expected_loaded:
        missing = sorted(expected_loaded - set(encoder_state))
        unexpected = sorted(set(encoder_state) - expected_loaded)
        raise ValueError(
            "Baseline 0とweak modelのencoder keyが一致しません: "
            f"missing={missing}, unexpected={unexpected}"
        )

    incompatible = model.load_state_dict(encoder_state, strict=False)
    expected_missing = {
        key for key in expected_random if not key.endswith(".num_batches_tracked")
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "初期化後のstate契約が不正です: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    outer_fold = expected_runtime.get("outer_fold")
    if not isinstance(outer_fold, int) or isinstance(outer_fold, bool):
        raise ValueError("expected_runtime.outer_foldは整数が必要です")
    return InitializationReport(
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_role=EXPECTED_CHECKPOINT_ROLE,
        outer_fold=outer_fold,
        loaded_key_count=len(expected_loaded),
        random_key_count=len(expected_random),
        loaded_prefixes=(TRANSFERRED_PREFIX,),
        random_prefixes=NEW_PREFIXES,
    )


def save_initialization_report(report: InitializationReport, output_path: Path) -> None:
    """Save the report atomically, refusing to silently overwrite a different one."""
    serialized = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"異なる初期化reportがすでに存在します: {output_path}")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
