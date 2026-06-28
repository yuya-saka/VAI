"""RSNA 4 領域パイプライン共通定数。"""

from __future__ import annotations

from pathlib import Path

# ─── パス ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
FRACTURE_DATASET_DIR = RSNA_DATA_DIR / "fracture_dataset"
PROCESSING_METADATA_DIR = RSNA_DATA_DIR / "processing_metadata"
TRAINING_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

DEFAULT_CKPT_DIR = (
    PROJECT_ROOT
    / "Unet"
    / "outputs"
    / "line_20260616"
    / "sig4.0_ALL(CC適用)"
    / "checkpoints"
)

# ─── モデル・推論パラメータ ───────────────────────────────────────────────
IMAGE_SIZE = 224
N_SEG_PLANES = 5
CENTER_CHANNEL = 2
SEG_INDEX_OFFSETS = (-2, -1, 0, 1, 2)
FALLBACK_LINE_LENGTH_PX = 80.0

DEFAULT_HEATMAP_THRESHOLD: dict[str, object] = {
    "mode": "adaptive",
    "min": 0.10,
    "peak_ratio": 0.4,
}

# ─── ラベル ──────────────────────────────────────────────────────────────
VERTEBRA_LEVELS = [f"C{i}" for i in range(1, 8)]
LINE_KEYS = tuple(f"line_{i}" for i in range(1, 5))
N_CLASSIFIER_PLANES = 15

# ─── 色定義 (BGR) ────────────────────────────────────────────────────────
LINE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "line_1": (0, 220, 0),
    "line_2": (0, 60, 255),
    "line_3": (255, 100, 0),
    "line_4": (0, 220, 220),
}

LINE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    k: (v[2], v[1], v[0]) for k, v in LINE_COLORS_BGR.items()
}

REGION_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 200, 0),
    (0, 0, 200),
    (200, 0, 0),
    (0, 200, 200),
)

REGION_NAMES = ("background", "body", "right_foramen", "left_foramen", "posterior")
