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
    REPO_ROOT / "fracture_detection/baseline0/outputs/08_19/pseudo_labels"
)
PSEUDO_SCORES_CSV = "pseudo_label_scores.csv"
PSEUDO_TEMPERATURES_CSV = "pseudo_label_temperatures.csv"
PSEUDO_METADATA_JSON = "pseudo_label_generation_metadata.json"

REGION_MASK_FILENAME = "region_4class.npy"
EXPECTED_REGION_MASK_SHAPE = EXPECTED_MASK_SHAPE  # (15, 224, 224)
REGION_MASK_VALUES = (0, 1, 2, 3, 4)  # 0=背景, 1..4=REGION_COLUMNS

REGION_SCORE_COLUMNS = tuple(f"{column}_score" for column in REGION_COLUMNS)

# 補助region batchの3ソース。値はTensor化するときのsource id。
SOURCE_HUMAN = 0
SOURCE_NEGATIVE = 1
SOURCE_PSEUDO = 2
SOURCE_NAMES = ("human", "negative", "pseudo")

EXPECTED_HUMAN_BAGS = 268
EXPECTED_NEGATIVE_BAGS = 12_100
EXPECTED_PSEUDO_BAGS = 1_064
