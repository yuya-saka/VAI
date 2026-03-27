"""Line-only implementation: 直線検出パイプライン

モジュール構成:
    src/        - 訓練パイプライン (model, dataset, data_utils, trainer)
    utils/      - 汎用ロジック (losses, metrics, detection, visualization)
    shim/       - 旧コード保管（参照用、import不可）
"""

from .utils.losses import (
    compute_line_loss,
    extract_gt_line_params,
    extract_pred_line_params_batch,
    get_warmup_weight,
)
from .utils.metrics import (
    compute_angle_error,
    compute_perpendicular_distance,
    compute_rho_error,
)

__all__ = [
    "extract_gt_line_params",
    "extract_pred_line_params_batch",
    "compute_line_loss",
    "get_warmup_weight",
    "compute_angle_error",
    "compute_rho_error",
    "compute_perpendicular_distance",
]
