"""中心画像を前後2スライスの文脈から予測するDataset。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import polyline_to_line_params, preprocess_polyline

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
VERTEBRAE = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


@dataclass(frozen=True)
class CenteredSliceRecord:
    """1枚の教師画像と、その前後スライス入力を表す。"""

    sample: str
    vertebra: str
    center_slice_index: int
    context_slice_indices: tuple[int, ...]
    context_valid: tuple[bool, ...]
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    lines: dict[str, list[list[float]]]


def validate_context_offsets(offsets: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """中心対称な文脈スライスoffsetを検証する。"""
    values = tuple(int(offset) for offset in offsets)
    if not values:
        raise ValueError("context_offsetsは1要素以上が必要です")
    if values != tuple(sorted(set(values))):
        raise ValueError("context_offsetsは重複なしの昇順が必要です")
    if values != tuple(-offset for offset in reversed(values)):
        raise ValueError("context_offsetsは0を中心に対称である必要があります")
    if 0 not in values:
        raise ValueError("context_offsetsには0が必要です")
    return values


def vertebra_names_from_group(group: str) -> tuple[str, ...]:
    """設定上の椎体groupを展開する。"""
    if group == "ALL":
        return VERTEBRAE
    if group == "C3_C7":
        return VERTEBRAE[2:]
    if group in VERTEBRAE:
        return (group,)
    raise ValueError(f"未知の椎体groupです: {group}")


def get_transforms(
    phase: str,
    augmentation_config: dict[str, Any] | None = None,
) -> A.ReplayCompose | None:
    """全5スライスへ同じ変換を適用するReplayComposeを返す。"""
    if phase != "train":
        return None
    config = augmentation_config or {}
    transforms: list[A.BasicTransform] = []
    if config.get("rotation", False):
        transforms.append(
            A.Rotate(
                limit=float(config.get("rotation_limit", 20)),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                fill_mask=0.0,
                p=0.5,
            )
        )
    if config.get("scale", False):
        scale_limit = float(config.get("scale_limit", 0.1))
        transforms.append(
            A.Affine(
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                translate_percent=0.0,
                rotate=0.0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                fill_mask=0.0,
                p=0.5,
            )
        )
    if config.get("horizontal_flip", False):
        transforms.append(
            A.HorizontalFlip(p=float(config.get("horizontal_flip_prob", 0.1)))
        )
    if config.get("brightness_contrast", False):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(config.get("brightness_limit", 0.1)),
                contrast_limit=float(config.get("contrast_limit", 0.1)),
                p=0.5,
            )
        )
    if config.get("gaussian_noise", False):
        transforms.append(A.GaussNoise(p=0.3))
    return A.ReplayCompose(
        transforms,
        additional_targets={
            "line_2": "keypoints",
            "line_3": "keypoints",
            "line_4": "keypoints",
        },
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _read_json(path: Path) -> Any:
    """JSONを読み込み、失敗時は空dictを返す。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _valid_lines(lines: Any) -> bool:
    """4本すべてが2点以上の手動線か判定する。"""
    return isinstance(lines, dict) and all(
        isinstance(lines.get(key), list) and len(lines[key]) >= 2 for key in LINE_KEYS
    )


def _load_bad_slices(annotation_root: Path) -> set[tuple[str, str, int]]:
    """全体除外JSONを検索用集合へ変換する。"""
    data = _read_json(annotation_root / "bad_slices_all.json")
    entries = data if isinstance(data, list) else data.get("bad_slices", [])
    excluded: set[tuple[str, str, int]] = set()
    for entry in entries:
        slice_value = entry.get("slice_idx", entry.get("slice"))
        if slice_value is not None:
            excluded.add(
                (str(entry["sample"]), str(entry["vertebra"]), int(slice_value))
            )
    return excluded


def _load_qc_excluded(path: Path) -> set[int]:
    """qc_scores.jsonのdict/list両形式から除外indexを返す。"""
    entries = _read_json(path)
    if isinstance(entries, dict):
        return {
            int(slice_key)
            for slice_key, value in entries.items()
            if isinstance(value, dict) and value.get("label") == "exclude"
        }
    if isinstance(entries, list):
        return {
            int(entry["slice_idx"])
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("label") == "exclude"
            and "slice_idx" in entry
        }
    return set()


def _load_manual_labels(
    annotation_root: Path,
    sample: str,
    vertebra: str,
    bad_slices: set[tuple[str, str, int]],
) -> dict[int, dict[str, list[list[float]]]]:
    """有効な手動線だけを読み込む。"""
    vertebra_dir = annotation_root / sample / vertebra
    raw_labels = _read_json(vertebra_dir / "lines.json")
    qc_excluded = _load_qc_excluded(vertebra_dir / "qc_scores.json")
    labels: dict[int, dict[str, list[list[float]]]] = {}
    if not isinstance(raw_labels, dict):
        return labels
    for slice_key, lines in raw_labels.items():
        slice_index = int(slice_key)
        if not _valid_lines(lines):
            continue
        if (sample, vertebra, slice_index) in bad_slices or slice_index in qc_excluded:
            continue
        processed = {key: preprocess_polyline(lines[key]) or [] for key in LINE_KEYS}
        if _valid_lines(processed):
            labels[slice_index] = processed
    return labels


def _dense_slice_paths(
    dense_root: Path,
    sample: str,
    vertebra: str,
) -> dict[int, tuple[Path, Path]]:
    """画像とmaskが揃う密スライスを列挙する。"""
    vertebra_dir = dense_root / sample / vertebra
    images = {
        int(path.stem.split("_")[-1]): path
        for path in (vertebra_dir / "images").glob("slice_*.png")
    }
    masks = {
        int(path.stem.split("_")[-1]): path
        for path in (vertebra_dir / "masks").glob("slice_*.png")
    }
    return {
        index: (images[index], masks[index])
        for index in sorted(images.keys() & masks.keys())
    }


def _contiguous_bounds(indices: set[int], center: int) -> tuple[int, int]:
    """中心を含む連続スライスrunの両端を返す。"""
    low = center
    high = center
    while low - 1 in indices:
        low -= 1
    while high + 1 in indices:
        high += 1
    return low, high


def build_centered_records(
    dense_root: Path,
    annotation_root: Path,
    sample_names: list[str],
    group: str,
    context_offsets: tuple[int, ...],
) -> list[CenteredSliceRecord]:
    """手動線1画像につき1つの中心付き5スライスrecordを作る。"""
    offsets = validate_context_offsets(context_offsets)
    bad_slices = _load_bad_slices(annotation_root)
    records: list[CenteredSliceRecord] = []
    for sample in sorted(sample_names):
        for vertebra in vertebra_names_from_group(group):
            dense_paths = _dense_slice_paths(dense_root, sample, vertebra)
            available = set(dense_paths)
            if not available:
                continue
            labels = _load_manual_labels(
                annotation_root,
                sample,
                vertebra,
                bad_slices,
            )
            for center, lines in sorted(labels.items()):
                if center not in available:
                    continue
                low, high = _contiguous_bounds(available, center)
                desired = tuple(center + offset for offset in offsets)
                context_indices = tuple(min(high, max(low, index)) for index in desired)
                records.append(
                    CenteredSliceRecord(
                        sample=sample,
                        vertebra=vertebra,
                        center_slice_index=center,
                        context_slice_indices=context_indices,
                        context_valid=tuple(
                            low <= index <= high and index in available
                            for index in desired
                        ),
                        image_paths=tuple(
                            dense_paths[index][0] for index in context_indices
                        ),
                        mask_paths=tuple(
                            dense_paths[index][1] for index in context_indices
                        ),
                        lines=lines,
                    )
                )
    return records


class CenteredLineDataset(Dataset[dict[str, Any]]):
    """前後2スライスを入力し、中心画像の手動線を教師にするDataset。"""

    def __init__(
        self,
        records: list[CenteredSliceRecord],
        image_size: int,
        sigma: float,
        transform: A.ReplayCompose | None = None,
        augmentation_config: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.sigma = sigma
        self.transform = transform
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        """教師画像数を返す。"""
        return len(self.records)

    def _load_image(self, path: Path, is_mask: bool) -> np.ndarray:
        """PNGを0-1のfloat画像として読み込む。"""
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if image.shape != (self.image_size, self.image_size):
            interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=interpolation,
            )
        return image

    def _heatmap_from_polyline(self, points_xy: list[list[float]]) -> np.ndarray:
        """ポリラインからGaussian heatmapを作る。"""
        points = np.asarray(points_xy, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, self.image_size - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.image_size - 1)
        raster = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        cv2.polylines(
            raster,
            [points.astype(np.int32).reshape(-1, 1, 2)],
            isClosed=False,
            color=1,
            thickness=1,
        )
        distance = cv2.distanceTransform(1 - raster, cv2.DIST_L2, 5)
        sigma_squared = max(1e-6, self.sigma**2)
        return np.exp(-(distance**2) / (2.0 * sigma_squared)).astype(np.float32)

    def _did_flip(self, replay: dict[str, Any]) -> bool:
        """replayから水平反転の有無を返す。"""
        return any(
            transform.get("applied", False)
            and str(transform.get("__class_fullname__", "")).endswith("HorizontalFlip")
            for transform in replay.get("transforms", [])
        )

    def _swap_lines(
        self,
        lines: dict[str, list[list[float]]],
    ) -> dict[str, list[list[float]]]:
        """水平反転時に左右の線channelを交換する。"""
        swap = self.augmentation_config.get("hflip_channel_swap")
        if swap is None:
            return lines
        reordered = [lines[LINE_KEYS[int(value) - 1]] for value in swap]
        return dict(zip(LINE_KEYS, reordered, strict=True))

    def _apply_transform(
        self,
        ct_images: list[np.ndarray],
        mask_images: list[np.ndarray],
        lines: dict[str, list[list[float]]],
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, list[list[float]]]]:
        """中心で生成した変換を全スライスへ共有する。"""
        if self.transform is None:
            return ct_images, mask_images, lines
        center_position = len(ct_images) // 2
        center_output = self.transform(
            image=ct_images[center_position],
            mask=mask_images[center_position],
            keypoints=lines["line_1"],
            line_2=lines["line_2"],
            line_3=lines["line_3"],
            line_4=lines["line_4"],
        )
        replay = center_output["replay"]
        transformed_ct = list(ct_images)
        transformed_masks = list(mask_images)
        transformed_ct[center_position] = np.asarray(
            center_output["image"], dtype=np.float32
        )
        transformed_masks[center_position] = np.asarray(
            center_output["mask"], dtype=np.float32
        )
        for index in range(len(ct_images)):
            if index == center_position:
                continue
            output = A.ReplayCompose.replay(
                replay,
                image=ct_images[index],
                mask=mask_images[index],
                keypoints=[],
                line_2=[],
                line_3=[],
                line_4=[],
            )
            transformed_ct[index] = np.asarray(output["image"], dtype=np.float32)
            transformed_masks[index] = np.asarray(output["mask"], dtype=np.float32)
        transformed_lines = {
            "line_1": [list(point[:2]) for point in center_output["keypoints"]],
            "line_2": [list(point[:2]) for point in center_output["line_2"]],
            "line_3": [list(point[:2]) for point in center_output["line_3"]],
            "line_4": [list(point[:2]) for point in center_output["line_4"]],
        }
        if self._did_flip(replay):
            transformed_lines = self._swap_lines(transformed_lines)
        return transformed_ct, transformed_masks, transformed_lines

    def __getitem__(self, index: int) -> dict[str, Any]:
        """5スライス入力と中心画像教師を返す。"""
        record = self.records[index]
        ct_images = [self._load_image(path, False) for path in record.image_paths]
        mask_images = [self._load_image(path, True) for path in record.mask_paths]
        lines = record.lines
        ct_images, mask_images, lines = self._apply_transform(
            ct_images,
            mask_images,
            lines,
        )
        inputs = np.stack(
            [
                np.stack(
                    [
                        np.clip(ct_image, 0.0, 1.0),
                        np.clip(mask_image, 0.0, 1.0),
                    ],
                    axis=0,
                )
                for ct_image, mask_image in zip(ct_images, mask_images, strict=True)
            ],
            axis=0,
        ).astype(np.float32)
        heatmaps = np.stack(
            [self._heatmap_from_polyline(lines[key]) for key in LINE_KEYS],
            axis=0,
        )
        line_params = np.asarray(
            [polyline_to_line_params(lines[key], self.image_size) for key in LINE_KEYS],
            dtype=np.float32,
        )
        return {
            "image": torch.from_numpy(inputs),
            "heatmaps": torch.from_numpy(heatmaps),
            "line_params_gt": torch.from_numpy(line_params),
            "lines_json": json.dumps(lines, ensure_ascii=True),
            "context_valid": torch.as_tensor(record.context_valid, dtype=torch.bool),
            "context_slice_indices": torch.as_tensor(
                record.context_slice_indices,
                dtype=torch.long,
            ),
            "sample": record.sample,
            "vertebra": record.vertebra,
            "slice_idx": record.center_slice_index,
        }
