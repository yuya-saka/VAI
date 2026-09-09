"""lambda_k の勾配ノルム校正（学習前・optimizer更新なし・一度だけ）。

`.claude/docs/REGION_MODEL_DESIGN_JA.md` §5.2/§7.3の手続きに従う。cross-case
pairwise rankingの退役に伴いalpha校正も廃止し、shared trunk上の`L_whole`と
条件付き統一region BCEの勾配ノルム比だけからlambdaを一度に測る。統合4領域モデル
（`cam_soft` objective）をreferenceとして測定し、同じouter foldの
`no_pseudo`/`cam_soft_shuffled`アームはこの結果を読むだけで再利用する
（再校正しない）。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from fracture_detection.baseline0.modeling.losses import broadcast_bce_loss
from fracture_detection.region_branch.data_pipeline.batching import batch_tensors
from fracture_detection.region_branch.data_pipeline.loaders import OuterFoldLoaders
from fracture_detection.region_branch.modeling.losses import (
    compute_conditional_region_losses,
    region_bag_logits,
)
from fracture_detection.region_branch.modeling.model import RegionBranchModel

N_CALIBRATION_BATCHES = 64
GRAD_NORM_EPSILON = 1e-8
LAMBDA_TARGET = 0.25
LAMBDA_MIN, LAMBDA_MAX = 0.01, 10.0


@dataclass(frozen=True)
class CalibrationResult:
    """1 outer foldのlambda_k校正結果。"""

    lambda_: float
    lambda_raw_ratio_median: float
    lambda_clipped: bool
    whole_trunk_norms: list[float]
    region_trunk_norms: list[float]
    n_batches: int
    seed: int
    outer_fold: int
    calibration_version: str = ""
    config_fingerprint: str = ""


def calibrate(
    model: RegionBranchModel,
    loaders: OuterFoldLoaders,
    outer_fold: int,
    pos_weight: float,
    active_regions: tuple[int, ...],
    seed: int,
    device: torch.device,
    n_batches: int = N_CALIBRATION_BATCHES,
) -> CalibrationResult:
    """optimizer更新前の決定的なn_batchesからlambda_kを一度だけ測る。

    natural batch一本から、shared trunk `blocks[4]`上のL_whole勾配ノルムと
    条件付き統一region BCE勾配ノルムを同じbatchについて測定する（region経路もnatural
    streamを共有するため、補助batchは不要）。
    """
    if n_batches < 1:
        raise ValueError("n_batchesは1以上である必要があります")
    model.to(device)
    # train()が必須: cuDNNのLSTM backwardはeval()では実行できない
    # （"cudnn RNN backward can only be called in training mode"）。
    # 実際の学習もtrain()で行うため、この方がgradientの実測としても忠実になる。
    model.train()

    torch.manual_seed(seed)
    whole_norms, region_norms = _measure_lambda_norms(
        model, loaders, outer_fold, pos_weight, active_regions, n_batches, device
    )
    if not region_norms:
        raise ValueError("lambda校正batchにwhole陽性region targetがありません")
    lambda_value, lambda_ratio_median, lambda_clipped = _calibrate_coefficient(
        whole_norms, region_norms, LAMBDA_TARGET, LAMBDA_MIN, LAMBDA_MAX
    )

    return CalibrationResult(
        lambda_=lambda_value,
        lambda_raw_ratio_median=lambda_ratio_median,
        lambda_clipped=lambda_clipped,
        whole_trunk_norms=whole_norms,
        region_trunk_norms=region_norms,
        n_batches=len(region_norms),
        seed=seed,
        outer_fold=outer_fold,
    )


def _measure_lambda_norms(
    model: RegionBranchModel,
    loaders: OuterFoldLoaders,
    outer_fold: int,
    pos_weight: float,
    active_regions: tuple[int, ...],
    n_batches: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """shared trunk blocks[4]上のwhole・条件付きregion勾配ノルムを測る。"""
    natural_iter = iter(loaders.natural)
    parameters = model.shared_parameters()

    whole_norms: list[float] = []
    region_norms: list[float] = []
    for _ in tqdm(
        range(n_batches),
        desc=f"outer{outer_fold} lambda校正",
        leave=False,
        dynamic_ncols=True,
    ):
        natural_batch = next(natural_iter)
        norms = _joint_norms_for_batch(
            model,
            natural_batch,
            pos_weight,
            active_regions,
            parameters,
            device,
        )
        if norms is None:
            continue
        whole_norm, region_norm = norms
        whole_norms.append(whole_norm)
        region_norms.append(region_norm)
    return whole_norms, region_norms


def _joint_norms_for_batch(
    model: RegionBranchModel,
    natural_batch: dict[str, Any],
    pos_weight: float,
    active_regions: tuple[int, ...],
    parameters: list[nn.Parameter],
    device: torch.device,
) -> tuple[float, float] | None:
    """1 encoder forwardから同じbatchのwhole・region勾配ノルムを測る。"""
    bt = batch_tensors(natural_batch, device)
    positive = bt.vertebra_target.eq(1.0)
    if not positive.any():
        return None
    positive_indices = positive.nonzero(as_tuple=False).flatten()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(
            bt.inputs,
            bt.region_mask,
            need_whole=True,
            need_region=True,
            region_sample_indices=positive_indices,
        )
    if (
        output.whole_plane_logits is None
        or output.region_plane_logits is None
        or output.region_plane_valid is None
    ):
        raise RuntimeError("lambda校正に必要なmodel出力が計算されませんでした")
    l_whole = broadcast_bce_loss(
        output.whole_plane_logits, bt.vertebra_target, pos_weight
    )
    bag_logits, cell_valid = region_bag_logits(
        output.region_plane_logits, output.region_plane_valid
    )
    active = list(active_regions)
    effective_target_valid = cell_valid & bt.region_target_valid[positive][:, active]
    region_losses = compute_conditional_region_losses(
        bag_logits,
        bt.region_target[positive][:, active],
        effective_target_valid,
        bt.vertebra_target[positive],
    )
    if region_losses.valid_cells == 0:
        return None
    whole_norm = _grad_norm(l_whole, parameters, retain_graph=True)
    region_norm = _grad_norm(region_losses.bce, parameters, retain_graph=False)
    return whole_norm, region_norm


def _grad_norm(
    loss: Tensor, parameters: list[nn.Parameter], retain_graph: bool = True
) -> float:
    """optimizer stepを起こさずにparameters上の勾配L2ノルムを測る。

    `retain_graph=False`は、そのlossがこの1回しかgradを取らない（同じforwardの
    graphを他のlossと共有しない）場合にだけ渡し、活性化のpeak memoryを早く解放する。
    """
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    total = torch.zeros((), device=loss.device)
    for grad in grads:
        if grad is None:
            continue
        total = total + grad.detach().float().pow(2).sum()
    value = float(torch.sqrt(total).item())
    if not math.isfinite(value):
        raise FloatingPointError("勾配ノルムが非有限値です")
    return value


def _calibrate_coefficient(
    numerator_norms: list[float],
    denominator_norms: list[float],
    target: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float, bool]:
    """`target*exp(median_b log((num+eps)/(den+eps)))`をclipして返す。"""
    if len(numerator_norms) != len(denominator_norms) or not numerator_norms:
        raise ValueError("numerator/denominator長が一致しないか空です")
    log_ratios = [
        math.log((num + GRAD_NORM_EPSILON) / (den + GRAD_NORM_EPSILON))
        for num, den in zip(numerator_norms, denominator_norms, strict=True)
    ]
    median_log_ratio = statistics.median(log_ratios)
    raw_value = target * math.exp(median_log_ratio)
    if not math.isfinite(raw_value):
        raise FloatingPointError("校正係数が非有限値です")
    clipped_value = min(max(raw_value, minimum), maximum)
    return clipped_value, median_log_ratio, clipped_value != raw_value


def save_calibration(result: CalibrationResult, path: Path) -> None:
    """校正結果をJSONへアトミック保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(
            f"同じcalibration versionに異なる結果があります: {path}。"
            "config.calibration.versionを更新してください"
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)


def load_calibration(path: Path) -> CalibrationResult:
    """校正結果JSONを読み込む。"""
    if not path.is_file():
        raise FileNotFoundError(f"校正結果がありません: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationResult(**payload)


def attach_calibration_metadata(
    result: CalibrationResult, config: dict[str, Any]
) -> CalibrationResult:
    """versionと校正関連config fingerprintを結果へ付加する。"""
    return replace(
        result,
        calibration_version=_calibration_version(config),
        config_fingerprint=calibration_config_fingerprint(config),
    )


def validate_calibration_compatibility(
    result: CalibrationResult, config: dict[str, Any]
) -> None:
    """校正artifactが現在のversion・設定・outer foldと一致することを検証する。"""
    expected_version = _calibration_version(config)
    if result.calibration_version != expected_version:
        raise ValueError(
            "校正artifactのversionがconfigと一致しません: "
            f"artifact={result.calibration_version!r}, config={expected_version!r}"
        )
    expected_fingerprint = calibration_config_fingerprint(config)
    if result.config_fingerprint != expected_fingerprint:
        raise ValueError(
            "校正artifactと現在のconfigが一致しません。"
            "config.calibration.versionを更新して再校正してください"
        )
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or result.outer_fold != runtime.get("outer_fold"):
        raise ValueError("校正artifactのouter foldがconfig.runtimeと一致しません")


def calibration_config_fingerprint(config: dict[str, Any]) -> str:
    """出力先・pseudo_arm等を除いた校正・学習条件のSHA-256を返す。

    `pseudo_arm`/`pseudo_label_dir`は除外する: 校正はouter foldごとに`cam_soft`
    objectiveで一度だけ測り、同じfoldの`no_pseudo`/`cam_soft_shuffled`アームへ
    同じlambdaを配る（`.claude/docs/REGION_MODEL_DESIGN_JA.md` §7.3）。これらの
    キーを指紋へ含めると、アームが違うだけでconfigが「別物」と判定され、共有
    すべき同一artifactを誤って拒否してしまう。
    """
    data = _config_section(config, "data")
    region = _config_section(config, "region")
    training = _config_section(config, "training")
    relevant = {
        "protocol_version": config.get("protocol_version"),
        "data": {
            "random_seed": data.get("random_seed"),
            "n_folds": data.get("n_folds"),
        },
        "model": _config_section(config, "model"),
        "region": {
            key: value
            for key, value in region.items()
            if key not in {"active_regions", "pseudo_arm", "pseudo_label_dir"}
        },
        "training": {key: value for key, value in training.items() if key != "gpu_id"},
        "augmentation": _config_section(config, "augmentation"),
        "runtime": _config_section(config, "runtime"),
    }
    serialized = json.dumps(
        relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _calibration_version(config: dict[str, Any]) -> str:
    calibration = _config_section(config, "calibration")
    version = calibration.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("calibration.versionは非空文字列が必要です")
    return version


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config.{name}はmappingである必要があります")
    return section
