from __future__ import annotations

import pytest
import torch

from fracture_detection.baseline0.modeling.model import Baseline0Model


def test_model_returns_one_logit_per_plane_and_freezes_batch_norm() -> None:
    model = Baseline0Model(
        backbone_name="tf_efficientnetv2_b0",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        head_dropout=0.0,
    )
    model.set_backbone_trainable(False)
    model.train()

    logits = model(torch.zeros(1, 15, 6, 64, 64))

    assert logits.shape == (1, 15)
    assert not any(parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(
        not module.training
        for module in model.encoder.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )


def test_model_rejects_invalid_plane_count() -> None:
    model = Baseline0Model(
        backbone_name="tf_efficientnetv2_b0",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        head_dropout=0.0,
    )

    with pytest.raises(ValueError, match="入力shape"):
        model(torch.zeros(1, 14, 6, 64, 64))
