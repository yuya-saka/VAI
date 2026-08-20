"""共有trainerと各アームを接続する型契約。"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ArmOutput:
    """1回のforwardで得られる共通出力。"""

    whole_logits: Tensor
    region_logits: Tensor | None = None
    spatial_attention: Tensor | None = None


@dataclass(frozen=True)
class LossWeights:
    """outer foldごとに凍結する補助損失係数。"""

    region: float = 0.0
    attention: float = 0.0
