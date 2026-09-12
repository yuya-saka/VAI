"""Fixed data contract for the weak-label region-MIL model.

Re-exports the shared bag-file contract from baseline0 (dataset location,
CT/mask shapes, region column names) and adds the constants specific to this
package's region-mask input.
"""

from __future__ import annotations

from pathlib import Path

from fracture_detection.baseline0.data.constants import (
    DATASET_DIR as DATASET_DIR,
)
from fracture_detection.baseline0.data.constants import (
    EXPECTED_CT_DTYPE as EXPECTED_CT_DTYPE,
)
from fracture_detection.baseline0.data.constants import (
    EXPECTED_CT_SHAPE as EXPECTED_CT_SHAPE,
)
from fracture_detection.baseline0.data.constants import (
    EXPECTED_MASK_SHAPE as EXPECTED_MASK_SHAPE,
)
from fracture_detection.baseline0.data.constants import (
    INPUT_MANIFEST_CSV as INPUT_MANIFEST_CSV,
)
from fracture_detection.baseline0.data.constants import (
    LEVELS as LEVELS,
)
from fracture_detection.baseline0.data.constants import (
    MANIFEST_COLUMNS as MANIFEST_COLUMNS,
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
    REPO_ROOT as REPO_ROOT,
)

WEAK_DIR = Path(__file__).resolve().parents[1]

# region_4class.npy holds the anatomical region label map: 0 = background,
# 1..N_REGIONS = REGION_COLUMNS. It is never treated as a segmentation GT for
# fracture presence -- only as a readout mask for mask_normalized_pool.
REGION_MASK_FILENAME = "region_4class.npy"
EXPECTED_REGION_MASK_SHAPE = EXPECTED_MASK_SHAPE  # (15, 224, 224)
REGION_MASK_VALUES = tuple(range(N_REGIONS + 1))  # (0, 1, 2, 3, 4)

DEFAULT_BASELINE0_CHECKPOINT_ROOT = (
    REPO_ROOT / "fracture_detection/baseline0/outputs/09_04/baseline0_aug追加"
)
