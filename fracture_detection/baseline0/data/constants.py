"""Baseline 0の固定データ契約。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data/rsna_data/fracture_dataset"
TRAIN_CSV = REPO_ROOT / "data/rsna_data/train.csv"
REGION_CSV = REPO_ROOT / "data/rsna_data/fracture_region_labels_dicom.csv"
EXCLUDED_STUDIES_CSV = REPO_ROOT / "data/rsna_data/excluded_studies.csv"
EXCLUDED_LEVELS_CSV = REPO_ROOT / "data/rsna_data/excluded_levels.csv"
INPUT_MANIFEST_CSV = (
    REPO_ROOT / "fracture_detection/baseline0/resources/input_manifest.csv"
)
INPUT_MANIFEST_META_JSON = (
    REPO_ROOT / "fracture_detection/baseline0/resources/input_manifest_meta.json"
)

LEVELS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
REGION_COLUMNS = ("region_1", "region_2", "region_3", "region_4")
ANNOTATION_COMPLETE_COLUMN = "annotation_complete"
REGION_TARGET_VALID_COLUMNS = tuple(
    f"{region}_target_valid" for region in REGION_COLUMNS
)
REGION_NAMES = (
    "vertebral_body",
    "right_transverse_foramen",
    "left_transverse_foramen",
    "posterior_elements",
)

EXPECTED_CT_SHAPE = (15, 5, 224, 224)
EXPECTED_MASK_SHAPE = (15, 224, 224)
EXPECTED_CT_DTYPE = "uint8"
MASK_CHANNELS = 5
CT_CHANNELS = 5
N_PLANES = 15
N_REGIONS = 4
EXPECTED_MANIFEST_ROWS = 13_432
EXPECTED_MANIFEST_STUDIES = 2_009

BAG_FILE_COLUMNS = ("ct_bytes", "vertebra_mask_bytes", "region_4class_bytes")
MANIFEST_COLUMNS = (
    "study_id",
    "level",
    "fold",
    "vertebra_target",
    "has_region_target",
    *REGION_COLUMNS,
)
SUPERVISED_MANIFEST_COLUMNS = (
    *MANIFEST_COLUMNS,
    ANNOTATION_COMPLETE_COLUMN,
    *REGION_TARGET_VALID_COLUMNS,
)
