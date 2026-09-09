"""region_branchの固定データ契約。"""

from pathlib import Path

from fracture_detection.baseline0.data.constants import (
    DATASET_DIR as DATASET_DIR,
)
from fracture_detection.baseline0.data.constants import (
    EXPECTED_MASK_SHAPE,
)
from fracture_detection.baseline0.data.constants import (
    INPUT_MANIFEST_CSV as INPUT_MANIFEST_CSV,
)
from fracture_detection.baseline0.data.constants import (
    N_PLANES as N_PLANES,
)
from fracture_detection.baseline0.data.constants import (
    N_REGIONS as N_REGIONS,
)
from fracture_detection.baseline0.data.constants import (
    REGION_COLUMNS as REGION_COLUMNS,
)
from fracture_detection.baseline0.data.constants import (
    REGION_NAMES as REGION_NAMES,
)
from fracture_detection.baseline0.data.constants import (
    REGION_TARGET_VALID_COLUMNS as REGION_TARGET_VALID_COLUMNS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REGION_BRANCH_DIR = Path(__file__).resolve().parents[1]

# alpha_k/lambda_kは統合4領域modelがouter foldごとに一度だけ校正し、単一領域4modelは
# 再校正せずここを読む（.claude/docs/research/20260825-region-loss-balancing.md）。
# experiment.phase/nameは試行のたびに変えることが多いため、校正結果はどのconfigの
# 出力先が変わっても場所がぶれないよう、experimentのoutputs treeとは独立させる。
CALIBRATION_DIR = REGION_BRANCH_DIR / "outputs" / "calibration"

DEFAULT_PSEUDO_LABEL_DIR = (
    REPO_ROOT / "fracture_detection/baseline0/outputs/09_04/pseudo_labels"
)
# CAM-soft pseudo-target artifacts (baseline0/cli/generate_pseudo_labels.py).
PSEUDO_REGION_TARGETS_CSV = "pseudo_region_targets.csv"
PSEUDO_CALIBRATION_CSV = "pseudo_target_calibration.csv"
PSEUDO_METADATA_JSON = "pseudo_target_generation_metadata.json"

REGION_MASK_FILENAME = "region_4class.npy"
EXPECTED_REGION_MASK_SHAPE = EXPECTED_MASK_SHAPE  # (15, 224, 224)
REGION_MASK_VALUES = (0, 1, 2, 3, 4)  # 0=背景, 1..4=REGION_COLUMNS

REGION_SHARE_COLUMNS = tuple(f"{column}_cam_share" for column in REGION_COLUMNS)
REGION_PSEUDO_TARGET_COLUMNS = tuple(
    f"{column}_pseudo_target" for column in REGION_COLUMNS
)
