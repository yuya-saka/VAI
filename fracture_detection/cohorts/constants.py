"""matchedコホートの固定契約。"""

from pathlib import Path

from fracture_detection.common.constants import REPO_ROOT, TRAIN_CSV

COHORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = COHORT_DIR / "outputs"
MATCHED_COHORT_CSV = OUTPUT_DIR / "matched_cohort.csv"
MATCHED_COHORT_META_JSON = OUTPUT_DIR / "matched_cohort_meta.json"
INPUT_MANIFEST_CSV = REPO_ROOT / "fracture_detection/common/outputs/input_manifest.csv"

SEED = 20260807
COHORT_ROLE_COLUMN = "cohort_role"
ANNOTATED_ROLE = "annotated"
NEGATIVE_ROLE = "negative"
EXPECTED_ANNOTATED_ROWS = 268

__all__ = [
    "ANNOTATED_ROLE",
    "COHORT_ROLE_COLUMN",
    "EXPECTED_ANNOTATED_ROWS",
    "INPUT_MANIFEST_CSV",
    "MATCHED_COHORT_CSV",
    "MATCHED_COHORT_META_JSON",
    "NEGATIVE_ROLE",
    "OUTPUT_DIR",
    "SEED",
    "TRAIN_CSV",
]
