"""alpha_k / lambda_k の勾配ノルム校正（学習前・optimizer更新なし・一度だけ）。

`.claude/docs/research/20260825-region-loss-balancing.md`の手続きに従う。
統合4領域モデルをreferenceとして測定し、同じouter foldの単一領域4モデルは
この結果を読むだけで再利用する（再校正しない）。
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
from fracture_detection.region_branch.data_pipeline.loaders import OuterFoldLoaders
from fracture_detection.region_branch.data_pipeline.sampling import (
    batch_tensors,
    concatenate_batches,
)
from fracture_detection.region_branch.modeling.losses import (
    combine_exact_terms,
    compute_exact_loss_terms,
    region_bag_logits,
    region_rank_loss,
)
from fracture_detection.region_branch.modeling.model import RegionBranchModel

N_CALIBRATION_BATCHES = 64
GRAD_NORM_EPSILON = 1e-8
ALPHA_TARGET = 0.25
ALPHA_MIN, ALPHA_MAX = 0.01, 1.0
LAMBDA_TARGET = 0.25
LAMBDA_MIN, LAMBDA_MAX = 0.01, 10.0


@dataclass(frozen=True)
class CalibrationResult:
    """1 outer foldのalpha_k / lambda_k校正結果。"""

    alpha: float
    lambda_: float
    alpha_raw_ratio_median: float
    lambda_raw_ratio_median: float
    alpha_clipped: bool
    lambda_clipped: bool
    human_region_norms: list[float]
    rank_region_norms: list[float]
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
    seed: int,
    device: torch.device,
    n_batches: int = N_CALIBRATION_BATCHES,
) -> CalibrationResult:
    """optimizer更新前の決定的なn_batchesからalpha_k・lambda_kを一度だけ測る。"""
    if n_batches < 1:
        raise ValueError("n_batchesは1以上である必要があります")
    model.to(device)
    # train()が必須: cuDNNのLSTM backwardはeval()では実行できない
    # （"cudnn RNN backward can only be called in training mode"）。
    # 実際の学習もtrain()で行うため、この方がgradientの実測としても忠実になる。
    # Dropoutのstochasticityは、両passの直前で同じseedへ固定して再現性を保つ。
    model.train()

    torch.manual_seed(seed)
    human_norms, rank_norms = _measure_alpha_norms(
        model, loaders, outer_fold, n_batches, device
    )
    alpha, alpha_ratio_median, alpha_clipped = _calibrate_coefficient(
        human_norms, rank_norms, ALPHA_TARGET, ALPHA_MIN, ALPHA_MAX
    )

    torch.manual_seed(seed)
    whole_norms, region_norms = _measure_lambda_norms(
        model, loaders, outer_fold, alpha, pos_weight, n_batches, device
    )
    lambda_value, lambda_ratio_median, lambda_clipped = _calibrate_coefficient(
        whole_norms, region_norms, LAMBDA_TARGET, LAMBDA_MIN, LAMBDA_MAX
    )

    return CalibrationResult(
        alpha=alpha,
        lambda_=lambda_value,
        alpha_raw_ratio_median=alpha_ratio_median,
        lambda_raw_ratio_median=lambda_ratio_median,
        alpha_clipped=alpha_clipped,
        lambda_clipped=lambda_clipped,
        human_region_norms=human_norms,
        rank_region_norms=rank_norms,
        whole_trunk_norms=whole_norms,
        region_trunk_norms=region_norms,
        n_batches=n_batches,
        seed=seed,
        outer_fold=outer_fold,
    )


def _measure_alpha_norms(
    model: RegionBranchModel,
    loaders: OuterFoldLoaders,
    outer_fold: int,
    n_batches: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """region BiLSTM上のL_exact・L_rank勾配ノルムをn_batches回測る。"""
    human_iter = iter(loaders.human)
    negative_iter = iter(loaders.negative)
    pseudo_iter = iter(loaders.pseudo)
    parameters = model.region_lstm_parameters()

    human_norms: list[float] = []
    rank_norms: list[float] = []
    for batch_index in tqdm(
        range(n_batches),
        desc=f"outer{outer_fold} alpha校正",
        leave=False,
        dynamic_ncols=True,
    ):
        aux = concatenate_batches(
            [next(human_iter), next(negative_iter), next(pseudo_iter)]
        )
        g_h, g_p = _alpha_norms_for_batch(
            model,
            aux,
            outer_fold,
            batch_index,
            loaders.temperatures,
            parameters,
            device,
        )
        human_norms.append(g_h)
        rank_norms.append(g_p)
    return human_norms, rank_norms


def _alpha_norms_for_batch(
    model: RegionBranchModel,
    aux: dict[str, Any],
    outer_fold: int,
    batch_index: int,
    temperatures: Tensor,
    parameters: list[nn.Parameter],
    device: torch.device,
) -> tuple[float, float]:
    """1 aux batch分のL_exact・L_rank勾配ノルムを測る。

    関数呼び出しにすることで、forward活性化を保持するローカル変数が戻り値だけ
    残してこのスコープごと解放され、次batchのforwardと同時に生きたままにならない。
    """
    bt = batch_tensors(aux, device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(bt.inputs, bt.region_mask, need_whole=False, need_region=True)
    bag_logits, cell_valid = region_bag_logits(
        output.region_plane_logits, output.region_plane_valid
    )
    terms = compute_exact_loss_terms(
        bag_logits,
        bt.region_targets,
        bt.region_target_valid,
        cell_valid,
        bt.vertebra_target,
    )
    l_exact, _, _ = combine_exact_terms(terms, bag_logits)
    g_h = _grad_norm(l_exact, parameters)

    generator = torch.Generator().manual_seed(_pair_seed(outer_fold, batch_index))
    teacher_outer_fold = torch.full(
        (bag_logits.shape[0],), outer_fold, dtype=torch.int64, device=device
    )
    l_rank, _ = region_rank_loss(
        bag_logits,
        bt.region_scores,
        bt.vertebra_target,
        teacher_outer_fold,
        temperatures,
        generator,
    )
    g_p = _grad_norm(l_rank, parameters, retain_graph=False)
    return g_h, g_p


def _measure_lambda_norms(
    model: RegionBranchModel,
    loaders: OuterFoldLoaders,
    outer_fold: int,
    alpha: float,
    pos_weight: float,
    n_batches: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """shared trunk blocks[4]上のL_whole・weighted region loss勾配ノルムを測る。"""
    natural_iter = iter(loaders.natural)
    human_iter = iter(loaders.human)
    negative_iter = iter(loaders.negative)
    pseudo_iter = iter(loaders.pseudo)
    parameters = model.shared_parameters()

    whole_norms: list[float] = []
    region_norms: list[float] = []
    for batch_index in tqdm(
        range(n_batches),
        desc=f"outer{outer_fold} lambda校正",
        leave=False,
        dynamic_ncols=True,
    ):
        g_w = _whole_norm_for_batch(
            model, next(natural_iter), pos_weight, parameters, device
        )
        whole_norms.append(g_w)

        aux = concatenate_batches(
            [next(human_iter), next(negative_iter), next(pseudo_iter)]
        )
        g_r = _region_norm_for_batch(
            model,
            aux,
            outer_fold,
            batch_index,
            alpha,
            loaders.temperatures,
            parameters,
            device,
        )
        region_norms.append(g_r)
    return whole_norms, region_norms


def _whole_norm_for_batch(
    model: RegionBranchModel,
    natural_batch: dict[str, Any],
    pos_weight: float,
    parameters: list[nn.Parameter],
    device: torch.device,
) -> float:
    """1 natural batch分のL_whole勾配ノルムを測る（関数scopeで活性化を解放する）。"""
    nbt = batch_tensors(natural_batch, device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        whole_output = model(
            nbt.inputs, nbt.region_mask, need_whole=True, need_region=False
        )
    l_whole = broadcast_bce_loss(
        whole_output.whole_plane_logits, nbt.vertebra_target, pos_weight
    )
    return _grad_norm(l_whole, parameters, retain_graph=False)


def _region_norm_for_batch(
    model: RegionBranchModel,
    aux: dict[str, Any],
    outer_fold: int,
    batch_index: int,
    alpha: float,
    temperatures: Tensor,
    parameters: list[nn.Parameter],
    device: torch.device,
) -> float:
    """1 aux batch分のweighted region loss勾配ノルムを測る（関数scopeで活性化を解放する）。"""
    bt = batch_tensors(aux, device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        region_output = model(
            bt.inputs, bt.region_mask, need_whole=False, need_region=True
        )
    bag_logits, cell_valid = region_bag_logits(
        region_output.region_plane_logits, region_output.region_plane_valid
    )
    terms = compute_exact_loss_terms(
        bag_logits,
        bt.region_targets,
        bt.region_target_valid,
        cell_valid,
        bt.vertebra_target,
    )
    l_exact, _, _ = combine_exact_terms(terms, bag_logits)
    generator = torch.Generator().manual_seed(_pair_seed(outer_fold, batch_index))
    teacher_outer_fold = torch.full(
        (bag_logits.shape[0],), outer_fold, dtype=torch.int64, device=device
    )
    l_rank, _ = region_rank_loss(
        bag_logits,
        bt.region_scores,
        bt.vertebra_target,
        teacher_outer_fold,
        temperatures,
        generator,
    )
    l_region = l_exact + alpha * l_rank
    return _grad_norm(l_region, parameters, retain_graph=False)


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


def _pair_seed(outer_fold: int, batch_index: int) -> int:
    """校正pass間で同一batchに同一ranking pairを再現するseed。"""
    return (outer_fold * 1_000_003 + batch_index) % (2**31 - 1)


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
    """出力先等を除いた校正・学習条件のSHA-256を返す。"""
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
            key: value for key, value in region.items() if key != "active_regions"
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
