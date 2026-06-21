"""Export classifier-plane arrays as inspection PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image

VERTEBRA_LEVELS: Final = tuple(f"C{level}" for level in range(1, 8))
PLANE_COUNT: Final = 15
CHANNEL_COUNT: Final = 5
CENTER_CHANNEL_INDEX: Final = 2
MONTAGE_COLUMNS: Final = 5
OVERLAY_COLOR: Final = np.asarray((255, 48, 48), dtype=np.float32)
OVERLAY_ALPHA: Final = 0.45


def export_study_pngs(study_directory: Path) -> tuple[Path, ...]:
    """Export CT, mask-overlay, and montage PNGs for one study."""
    if not study_directory.is_dir():
        raise FileNotFoundError(f"Study output directory not found: {study_directory}")

    generated_paths: list[Path] = []
    for level in VERTEBRA_LEVELS:
        level_directory = study_directory / level
        if not level_directory.is_dir():
            continue
        generated_paths.extend(export_vertebra_pngs(level_directory))
    if not generated_paths:
        raise ValueError("No vertebra classifier-plane arrays were found")
    return tuple(generated_paths)


def export_vertebra_pngs(level_directory: Path) -> tuple[Path, ...]:
    """Export one vertebra's classifier planes to PNG files."""
    ct = np.load(level_directory / "ct.npy")
    mask = np.load(level_directory / "vertebra_mask.npy")
    _validate_arrays(ct, mask)

    preview_directory = level_directory / "preview"
    ct_directory = preview_directory / "ct"
    overlay_directory = preview_directory / "overlay"
    ct_directory.mkdir(parents=True, exist_ok=True)
    overlay_directory.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    montage_tiles: list[Image.Image] = []
    center_ct = ct[:, CENTER_CHANNEL_INDEX]
    for index, (ct_plane, mask_plane) in enumerate(zip(center_ct, mask, strict=True)):
        ct_image = Image.fromarray(ct_plane, mode="L")
        ct_path = ct_directory / f"plane_{index:02d}.png"
        ct_image.save(ct_path)

        overlay_image = Image.fromarray(_mask_overlay(ct_plane, mask_plane), mode="RGB")
        overlay_path = overlay_directory / f"plane_{index:02d}.png"
        overlay_image.save(overlay_path)

        generated_paths.extend((ct_path, overlay_path))
        montage_tiles.append(overlay_image)

    montage_path = preview_directory / "montage.png"
    _montage(montage_tiles).save(montage_path)
    generated_paths.append(montage_path)
    return tuple(generated_paths)


def _mask_overlay(
    ct_plane: npt.NDArray[np.uint8],
    mask_plane: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    grayscale = np.repeat(ct_plane[..., None], 3, axis=2).astype(np.float32)
    foreground = mask_plane > 0
    grayscale[foreground] = (1.0 - OVERLAY_ALPHA) * grayscale[
        foreground
    ] + OVERLAY_ALPHA * OVERLAY_COLOR
    return np.asarray(np.clip(grayscale, 0.0, 255.0), dtype=np.uint8)


def _montage(tiles: list[Image.Image]) -> Image.Image:
    width, height = tiles[0].size
    row_count = (len(tiles) + MONTAGE_COLUMNS - 1) // MONTAGE_COLUMNS
    montage = Image.new(
        "RGB",
        (MONTAGE_COLUMNS * width, row_count * height),
        color="black",
    )
    for index, tile in enumerate(tiles):
        montage.paste(
            tile,
            ((index % MONTAGE_COLUMNS) * width, (index // MONTAGE_COLUMNS) * height),
        )
    return montage


def _validate_arrays(
    ct: npt.NDArray[np.generic],
    mask: npt.NDArray[np.generic],
) -> None:
    if ct.shape != (PLANE_COUNT, CHANNEL_COUNT, 224, 224):
        raise ValueError("Expected CT with shape (15, 5, 224, 224)")
    if mask.shape != (PLANE_COUNT, 224, 224):
        raise ValueError("Expected mask with shape (15, 224, 224)")
    if ct.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("CT and mask arrays must use uint8")
    if not np.all(np.isin(mask, (0, 1))):
        raise ValueError("Vertebra mask values must be 0 or 1")


def main() -> None:
    """Export PNG previews from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study_directory",
        type=Path,
        help="Study directory under data/rsna_data/fracture_dataset",
    )
    arguments = parser.parse_args()
    generated_paths = export_study_pngs(arguments.study_directory)
    print(f"Generated {len(generated_paths)} PNG files")


if __name__ == "__main__":
    main()
