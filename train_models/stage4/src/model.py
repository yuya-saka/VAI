"""Stage4 model reuses the frozen Stage3 architecture."""

from __future__ import annotations

from train_models.stage3.src.model import Stage3Model, Stage3Output

STAGE4_ARCHITECTURE_VERSION = 1


class Stage4Model(Stage3Model):
    """Stage3 hierarchy trained with Stage4 mixed region supervision."""


Stage4Output = Stage3Output
