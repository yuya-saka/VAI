"""
model.py のユニットテスト。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from learning.src.model import NUM_VERTEBRAE, FractureResNet18


def test_model_uses_resnet18_block_layout() -> None:
    """各stageが2 blockのResNet18構成である。"""
    model = FractureResNet18(pretrained=False)

    assert len(model.backbone.layer1) == 2
    assert len(model.backbone.layer2) == 2
    assert len(model.backbone.layer3) == 2
    assert len(model.backbone.layer4) == 2


def test_classifier_accepts_vertebra_one_hot() -> None:
    """分類headがResNet特徴量とC1-C7 one-hotを受け取る。"""
    model = FractureResNet18(pretrained=False)

    linear = model.classifier[1]

    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 512 + NUM_VERTEBRAE


def test_forward_uses_single_vertebra_index_for_all_slices() -> None:
    """bag内全sliceへ同じ椎体部位条件を展開できる。"""
    model = FractureResNet18(pretrained=False)
    model.eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        logits = model(images, vertebra_index=2)

    assert logits.shape == (2,)


def test_forward_rejects_invalid_vertebra_index() -> None:
    """C1-C7外の部位indexを拒否する。"""
    model = FractureResNet18(pretrained=False)
    model.eval()
    images = torch.randn(1, 3, 64, 64)

    with pytest.raises(ValueError, match="0〜6"):
        model(images, vertebra_index=7)


def test_batch_norm_stays_frozen_in_train_mode() -> None:
    """bag 単位学習中も BatchNorm が eval 固定される。"""
    model = FractureResNet18(pretrained=False, freeze_batch_norm=True)

    model.train()

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norm_layers
    assert all(not module.training for module in batch_norm_layers)
    assert all(
        not parameter.requires_grad
        for module in batch_norm_layers
        for parameter in module.parameters()
    )


def test_batch_norm_can_remain_trainable() -> None:
    """設定を無効化した場合は通常どおり BatchNorm を学習する。"""
    model = FractureResNet18(pretrained=False, freeze_batch_norm=False)

    model.train()

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norm_layers
    assert all(module.training for module in batch_norm_layers)
