"""校正artifactと正式実験manifestのimmutable管理。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fracture_detection.common.constants import FOLDS_CSV, INPUT_MANIFEST_CSV, REPO_ROOT

CALIBRATION_PROTOCOL = "fracture-gradient-calibration-v1"
FROZEN_MANIFEST_PROTOCOL = "fracture-frozen-experiment-v1"


def sha256_file(path: Path) -> str:
    """file内容のSHA256を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    """JSON正規化したobjectのSHA256を返す。"""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


OPERATIONAL_SECTIONS = ("runtime", "experiment", "parallel", "wandb")
OPERATIONAL_DATA_KEYS = ("start_outer_fold", "end_outer_fold")
OPERATIONAL_TRAINING_KEYS = ("gpu_id",)


def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """実験結果に影響しない運用設定を除いた凍結比較用configを返す。

    出力先(experiment)、実行するfold範囲、GPU割当(parallel/training.gpu_id)、
    W&B追跡は運用上の指定であり、モデル・損失・データ経路・乱数系列を変えないため
    凍結対象から除外する。
    """
    normalized = copy.deepcopy(dict(config))
    for section in OPERATIONAL_SECTIONS:
        normalized.pop(section, None)
    data = normalized.get("data")
    if isinstance(data, dict):
        for key in OPERATIONAL_DATA_KEYS:
            data.pop(key, None)
    training = normalized.get("training")
    if isinstance(training, dict):
        for key in OPERATIONAL_TRAINING_KEYS:
            training.pop(key, None)
    return normalized


def write_calibration_artifact(
    path: Path,
    *,
    kind: str,
    reference_arm: str,
    outer_folds: Mapping[int, Mapping[str, object]],
    reference_config_hashes: Mapping[int, str],
) -> Path:
    """5 foldのraw校正結果を新規fileへ保存する。"""
    if kind not in {"lambda", "beta"}:
        raise ValueError("校正kindはlambdaまたはbetaが必要です")
    if set(outer_folds) != set(range(5)) or set(reference_config_hashes) != set(
        range(5)
    ):
        raise ValueError("校正artifactにはouter 0〜4が必要です")
    payload = {
        "protocol_version": CALIBRATION_PROTOCOL,
        "kind": kind,
        "reference_arm": reference_arm,
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
        "outer_folds": {str(key): dict(value) for key, value in outer_folds.items()},
        "reference_config_hashes": {
            str(key): value for key, value in reference_config_hashes.items()
        },
    }
    return write_new_json(path, payload)


def combine_loss_weights(lambda_path: Path, beta_path: Path, output_path: Path) -> Path:
    """raw λ/βが両方揃った時だけimmutable loss weightsを生成する。"""
    lambda_payload = _read_calibration(lambda_path, "lambda", "baseline1_b")
    beta_payload = _read_calibration(beta_path, "beta", "proposed_b")
    folds: dict[str, dict[str, float]] = {}
    for outer_fold in range(5):
        key = str(outer_fold)
        lambda_value = float(lambda_payload["outer_folds"][key]["coefficient"])
        beta_value = float(beta_payload["outer_folds"][key]["coefficient"])
        if not all(
            math.isfinite(value) and value > 0 for value in (lambda_value, beta_value)
        ):
            raise ValueError(f"outer {outer_fold}の校正係数が不正です")
        folds[key] = {"lambda": lambda_value, "beta": beta_value}
    payload = {
        "protocol_version": "fracture-loss-weights-v1",
        "outer_folds": folds,
        "lambda_artifact": {
            "path": str(lambda_path),
            "sha256": sha256_file(lambda_path),
        },
        "beta_artifact": {
            "path": str(beta_path),
            "sha256": sha256_file(beta_path),
        },
    }
    return write_new_json(output_path, payload)


def source_tree_sha256(root: Path = REPO_ROOT) -> str:
    """正式実装に関与するPython source treeのhashを返す。

    arm configの内容は`config_sha256`（normalized_config経由）が個別に凍結するため、
    ここでは対象外にする。含めると実験名やGPU割当などの運用設定を変えるだけで
    このhashも変わってしまい、config_sha256側の正規化が意味を失う。
    """
    fracture_root = root / "fracture_detection"
    files = sorted(
        path
        for path in fracture_root.rglob("*.py")
        if "outputs" not in path.parts
        and "__pycache__" not in path.parts
        and not path.name.startswith(".")
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def dependency_sha256(root: Path = REPO_ROOT) -> str:
    """pyproject.tomlとuv.lockをまとめたdependency hashを返す。"""
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def create_frozen_manifest(
    path: Path,
    *,
    configs: Mapping[str, Mapping[str, Any]],
    loss_weights_path: Path,
    resource_profiles: Sequence[Path],
    fold_execution_plans: Sequence[Path],
) -> Path:
    """full training前のsource/config/artifact/resource契約を一度だけ凍結する。"""
    required_arms = {
        "baseline0",
        "control_b",
        "baseline1_b",
        "proposed_b",
        "proposed_max",
        "proposed_max_beta0",
    }
    if set(configs) != required_arms:
        raise ValueError(
            f"6構成configが必要です: {sorted(required_arms - set(configs))}"
        )
    if not resource_profiles:
        raise ValueError("resource profileが必要です")
    profile_payloads = [_read_json(profile) for profile in resource_profiles]
    validate_resource_profiles(profile_payloads)
    calibration_artifacts = _validate_loss_weights_for_configs(
        loss_weights_path, configs
    )
    provenance = {
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
    }
    for profile in profile_payloads:
        arm = str(profile["arm"])
        expected = {
            **provenance,
            "config_sha256": canonical_sha256(normalized_config(configs[arm])),
        }
        mismatches = {
            key: (profile.get(key), value)
            for key, value in expected.items()
            if profile.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"resource profileのprovenanceが不一致です: arm={arm}, {mismatches}"
            )
    payload = {
        "protocol_version": FROZEN_MANIFEST_PROTOCOL,
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
        "config_sha256": {
            arm: canonical_sha256(normalized_config(config))
            for arm, config in sorted(configs.items())
        },
        "loss_weights": {
            "path": str(loss_weights_path),
            "sha256": sha256_file(loss_weights_path),
        },
        "calibration_artifacts": calibration_artifacts,
        "resource_profiles": [
            {"path": str(profile), "sha256": sha256_file(profile)}
            for profile in resource_profiles
        ],
        "fold_execution_plans": [
            {"path": str(plan), "sha256": sha256_file(plan)}
            for plan in fold_execution_plans
        ],
        "hypothesis_order": [
            "H1:baseline1_b>control_b",
            "H2:proposed_max>proposed_max_beta0",
        ],
    }
    return write_new_json(path, payload)


def validate_resource_profiles(profiles: Sequence[Mapping[str, Any]]) -> None:
    """必須5構成・同一GPU・β経路parity・49GB gateを検証する。"""
    by_arm = {str(profile.get("arm")): profile for profile in profiles}
    required = {
        "baseline0",
        "baseline1_b",
        "proposed_b",
        "proposed_max",
        "proposed_max_beta0",
    }
    if len(by_arm) != len(profiles):
        raise ValueError("resource profileのarmが重複しています")
    if set(by_arm) != required:
        raise ValueError(f"resource profile構成が不正です: {sorted(set(by_arm))}")
    if any(not profile.get("resource_gate_passed", False) for profile in profiles):
        raise RuntimeError("resource gate未通過の構成があります")
    gpu_signatures = {
        (
            profile.get("gpu", {}).get("name"),
            profile.get("gpu", {}).get("compute_capability"),
            profile.get("gpu", {}).get("total_memory_bytes"),
        )
        for profile in profiles
    }
    if len(gpu_signatures) != 1:
        raise RuntimeError("resource profileは同一GPU modelで実行する必要があります")
    beta = by_arm["proposed_max"]
    beta_zero = by_arm["proposed_max_beta0"]
    if beta.get("parameters") != beta_zero.get("parameters"):
        raise RuntimeError("β>0とβ=0のparameter経路が一致しません")
    if not beta.get("attention_computed") or not beta_zero.get("attention_computed"):
        raise RuntimeError("β=0でもattentionを計算する必要があります")
    for key in ("peak_memory_reserved_bytes", "median_step_seconds"):
        first = float(beta[key])
        second = float(beta_zero[key])
        relative = abs(first - second) / max(first, second, 1e-12)
        if relative > 0.10:
            raise RuntimeError(f"β>0/β=0 profileが10%超乖離しました: {key}")


def verify_frozen_manifest(
    config: Mapping[str, Any], manifest_path: Path, loss_weights_path: Path
) -> None:
    """run/resume時に全hash guardを検証する。"""
    payload = _read_json(manifest_path)
    if payload.get("protocol_version") != FROZEN_MANIFEST_PROTOCOL:
        raise ValueError("frozen manifest protocolが不正です")
    checks = {
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
    }
    mismatches: dict[str, tuple[object, object]] = {
        key: (payload.get(key), actual)
        for key, actual in checks.items()
        if payload.get(key) != actual
    }
    arm_name = str(config["arm"]["name"])
    expected_config = payload.get("config_sha256", {}).get(arm_name)
    actual_config = canonical_sha256(normalized_config(config))
    if expected_config != actual_config:
        mismatches["config_sha256"] = (expected_config, actual_config)
    expected_weights = payload.get("loss_weights", {}).get("sha256")
    actual_weights = sha256_file(loss_weights_path)
    if expected_weights != actual_weights:
        mismatches["loss_weights"] = (expected_weights, actual_weights)
    for item in payload.get("calibration_artifacts", []):
        item_path = Path(item["path"])
        actual = sha256_file(item_path)
        if actual != item["sha256"]:
            mismatches[f"calibration:{item_path}"] = (item["sha256"], actual)
    for item in payload.get("resource_profiles", []):
        item_path = Path(item["path"])
        if sha256_file(item_path) != item["sha256"]:
            mismatches[f"resource:{item_path}"] = (
                item["sha256"],
                sha256_file(item_path),
            )
    for item in payload.get("fold_execution_plans", []):
        item_path = Path(item["path"])
        if sha256_file(item_path) != item["sha256"]:
            mismatches[f"fold_plan:{item_path}"] = (
                item["sha256"],
                sha256_file(item_path),
            )
    if mismatches:
        raise RuntimeError(f"frozen manifest hash不一致です: {mismatches}")


def write_new_json(path: Path, payload: Mapping[str, object]) -> Path:
    """既存fileを必ず拒否してJSONをatomic作成する。"""
    if path.exists():
        raise FileExistsError(f"immutable artifactは上書きできません: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _read_calibration(path: Path, kind: str, reference: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("protocol_version") != CALIBRATION_PROTOCOL:
        raise ValueError(f"校正artifact protocolが不正です: {path}")
    if payload.get("kind") != kind or payload.get("reference_arm") != reference:
        raise ValueError(f"校正artifactのkind/referenceが不正です: {path}")
    if set(payload.get("outer_folds", {})) != {str(value) for value in range(5)}:
        raise ValueError(f"校正artifactに5 foldがありません: {path}")
    expected_provenance = {
        "source_tree_sha256": source_tree_sha256(),
        "dependency_sha256": dependency_sha256(),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_CSV),
        "folds_sha256": sha256_file(FOLDS_CSV),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected_provenance.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"校正artifactのprovenanceが不一致です: {mismatches}")
    return payload


def _validate_loss_weights_for_configs(
    path: Path, configs: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    """loss weightsからraw校正artifactまでconfig/hashを辿って検証する。"""
    payload = _read_json(path)
    if payload.get("protocol_version") != "fracture-loss-weights-v1":
        raise ValueError("loss weights protocolが不正です")
    if set(payload.get("outer_folds", {})) != {str(value) for value in range(5)}:
        raise ValueError("loss weightsに5 foldがありません")
    artifacts: list[dict[str, str]] = []
    for kind, reference in (("lambda", "baseline1_b"), ("beta", "proposed_b")):
        item = payload.get(f"{kind}_artifact")
        if not isinstance(item, Mapping):
            raise ValueError(f"loss weightsに{kind} artifactがありません")
        artifact_path = Path(str(item.get("path")))
        expected_sha = str(item.get("sha256"))
        actual_sha = sha256_file(artifact_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{kind} calibration artifactのhashが不一致です")
        calibration = _read_calibration(artifact_path, kind, reference)
        expected_config_hash = canonical_sha256(normalized_config(configs[reference]))
        reference_hashes = calibration.get("reference_config_hashes", {})
        if set(reference_hashes) != {str(value) for value in range(5)} or set(
            reference_hashes.values()
        ) != {expected_config_hash}:
            raise RuntimeError(f"{kind} calibrationのreference configが不一致です")
        artifacts.append({"path": str(artifact_path), "sha256": actual_sha})
    return artifacts


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifactはobjectが必要です: {path}")
    return payload
