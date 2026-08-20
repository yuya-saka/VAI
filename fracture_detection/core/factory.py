"""統一configからmodel・adapterを構築する。"""

from __future__ import annotations

from typing import Any

from torch import nn

from fracture_detection.baseline0.modeling.model import Baseline0Model
from fracture_detection.core.steps import ArmAdapter
from fracture_detection.mtl.model import EarlyFusionMtlModel
from fracture_detection.proposed.model import ProposedModel


def build_model(config: dict[str, Any]) -> nn.Module:
    """登録済みarmだけを構築する。"""
    arm = config["arm"]
    model = config["model"]
    backbone_name = str(model["backbone"])
    pretrained = bool(model["pretrained"])
    drop_rate = float(model["drop_rate"])
    drop_path_rate = float(model["drop_path_rate"])
    head_dropout = float(model["head_dropout"])
    lstm_hidden = int(model["lstm_hidden"])
    lstm_layers = int(model["lstm_layers"])
    n_planes = int(model["n_planes"])
    if arm["kind"] == "baseline0":
        return Baseline0Model(
            backbone_name=backbone_name,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_planes=n_planes,
        )
    if arm["kind"] == "mtl":
        return EarlyFusionMtlModel(
            backbone_name=backbone_name,
            in_chans=int(arm["input_channels"]),
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_planes=n_planes,
        )
    if arm["kind"] == "proposed":
        return ProposedModel(
            backbone_name=backbone_name,
            pretrained=pretrained,
            whole_method=str(arm["whole_method"]),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_planes=n_planes,
        )
    raise ValueError(f"未対応arm kindです: {arm['kind']}")


def build_adapter(config: dict[str, Any]) -> ArmAdapter:
    """configのarm契約をtrainer adapterへ変換する。"""
    arm = config["arm"]
    return ArmAdapter(
        input_channels=int(arm["input_channels"]),
        region_enabled=bool(arm["region_enabled"]),
        attention_enabled=bool(arm["attention_enabled"]),
        legacy_tensor_output=arm["kind"] == "baseline0",
    )
