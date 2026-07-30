"""密なz画像と疎な手動線教師からスラブを構築する。"""

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

from ..utils.detection import extract_gt_line_params

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
VERTEBRAE = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


@dataclass(frozen=True)
class SlabRecord:
    """1つの連続スラブを表すindex情報。"""

    sample: str
    vertebra: str
    slice_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    labels: dict[int, dict[str, list[list[float]]]]


def preprocess_polyline(
    points_xy: list[list[float]] | None,
    duplicate_threshold: float = 1.0,
) -> list[list[float]] | None:
    """近接した中間点だけを除去し、線形状を保持する。"""
    if points_xy is None or len(points_xy) < 2:
        return points_xy
    points = np.asarray(points_xy, dtype=np.float64)
    keep_indices = [0]
    for index in range(1, len(points) - 1):
        if (
            np.linalg.norm(points[index] - points[keep_indices[-1]])
            >= duplicate_threshold
        ):
            keep_indices.append(index)
    keep_indices.append(len(points) - 1)
    filtered = points[keep_indices]
    return filtered.tolist() if len(filtered) >= 2 else points_xy


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
    """スラブ全体へreplay可能な変換を構築する。"""
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
                brightness_limit=float(config.get("brightness_limit", 0.2)),
                contrast_limit=float(config.get("contrast_limit", 0.2)),
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


def _load_bad_slices(annotation_root: Path) -> set[tuple[str, str, int]]:
    """全体除外JSONを `(sample, vertebra, z)` 集合へ変換する。"""
    data = _read_json(annotation_root / "bad_slices_all.json")
    entries = data if isinstance(data, list) else data.get("bad_slices", [])
    excluded: set[tuple[str, str, int]] = set()
    for entry in entries:
        slice_value = entry.get("slice_idx", entry.get("slice"))
        if slice_value is None:
            continue
        excluded.add((str(entry["sample"]), str(entry["vertebra"]), int(slice_value)))
    return excluded


def _valid_lines(lines: Any) -> bool:
    """4本すべてが2点以上の手動線か判定する。"""
    if not isinstance(lines, dict):
        return False
    return all(
        isinstance(lines.get(key), list) and len(lines[key]) >= 2 for key in LINE_KEYS
    )


def load_manual_labels(
    annotation_root: Path,
    sample: str,
    vertebra: str,
    bad_slices: set[tuple[str, str, int]],
) -> dict[int, dict[str, list[list[float]]]]:
    """手動lines.jsonだけを読み、有効な教師を返す。"""
    vertebra_dir = annotation_root / sample / vertebra
    raw_labels = _read_json(vertebra_dir / "lines.json")
    qc_entries = _read_json(vertebra_dir / "qc_scores.json")
    qc_excluded = {
        int(entry["slice_idx"])
        for entry in qc_entries
        if isinstance(entry, dict) and entry.get("label") == "exclude"
    }
    labels: dict[int, dict[str, list[list[float]]]] = {}
    if not isinstance(raw_labels, dict):
        return labels
    for slice_key, lines in raw_labels.items():
        if not _valid_lines(lines):
            continue
        slice_index = int(slice_key)
        if (sample, vertebra, slice_index) in bad_slices:
            continue
        if slice_index in qc_excluded:
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
    """画像とmaskが揃う密スライスのpathを列挙する。"""
    vertebra_dir = dense_root / sample / vertebra
    image_paths = {
        int(path.stem.split("_")[-1]): path
        for path in (vertebra_dir / "images").glob("slice_*.png")
    }
    mask_paths = {
        int(path.stem.split("_")[-1]): path
        for path in (vertebra_dir / "masks").glob("slice_*.png")
    }
    return {
        index: (image_paths[index], mask_paths[index])
        for index in sorted(image_paths.keys() & mask_paths.keys())
    }


def _window_starts(
    slice_indices: list[int],
    slab_size: int,
    stride: int,
    ensure_edge_coverage: bool,
) -> list[int]:
    """連続窓の開始位置を返し、推論時は末尾窓も保証する。"""
    valid_starts: list[int] = []
    for offset in range(0, max(0, len(slice_indices) - slab_size + 1), stride):
        window = slice_indices[offset : offset + slab_size]
        if len(window) == slab_size and all(
            second - first == 1
            for first, second in zip(window, window[1:], strict=False)
        ):
            valid_starts.append(offset)
    last_offset = len(slice_indices) - slab_size
    if ensure_edge_coverage and last_offset >= 0 and last_offset not in valid_starts:
        window = slice_indices[last_offset : last_offset + slab_size]
        if all(
            second - first == 1
            for first, second in zip(window, window[1:], strict=False)
        ):
            valid_starts.append(last_offset)
    return sorted(set(valid_starts))


def build_slab_records(
    dense_root: Path,
    annotation_root: Path,
    sample_names: list[str],
    group: str,
    slab_size: int,
    stride: int,
    min_labeled_slices: int,
    require_labels: bool,
) -> list[SlabRecord]:
    """指定sample群から学習または推論スラブを構築する。"""
    bad_slices = _load_bad_slices(annotation_root)
    records: list[SlabRecord] = []
    for sample in sorted(sample_names):
        for vertebra in vertebra_names_from_group(group):
            dense_paths = _dense_slice_paths(dense_root, sample, vertebra)
            if len(dense_paths) < slab_size:
                continue
            labels = load_manual_labels(
                annotation_root,
                sample,
                vertebra,
                bad_slices,
            )
            indices = sorted(dense_paths)
            starts = _window_starts(
                indices,
                slab_size,
                stride,
                ensure_edge_coverage=not require_labels,
            )
            for start in starts:
                window = tuple(indices[start : start + slab_size])
                window_labels = {
                    index: labels[index] for index in window if index in labels
                }
                if require_labels and len(window_labels) < min_labeled_slices:
                    continue
                records.append(
                    SlabRecord(
                        sample=sample,
                        vertebra=vertebra,
                        slice_indices=window,
                        image_paths=tuple(dense_paths[index][0] for index in window),
                        mask_paths=tuple(dense_paths[index][1] for index in window),
                        labels=window_labels,
                    )
                )
    return records


class SlabLineDataset(Dataset[dict[str, Any]]):
    """連続スライスをchannel方向へ積んだ線学習Dataset。"""

    def __init__(
        self,
        records: list[SlabRecord],
        image_size: int,
        sigma: float,
        transform: A.ReplayCompose | None = None,
        augmentation_config: dict[str, Any] | None = None,
    ):
        self.records = records
        self.image_size = image_size
        self.sigma = sigma
        self.transform = transform
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        """スラブ数を返す。"""
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

    def _heatmap_from_polyline(
        self,
        points_xy: list[list[float]] | None,
    ) -> np.ndarray:
        """ポリラインから距離変換ベースのGaussian heatmapを作る。"""
        heatmap = np.zeros(
            (self.image_size, self.image_size),
            dtype=np.float32,
        )
        if points_xy is None or len(points_xy) < 2:
            return heatmap
        points = np.asarray(points_xy, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, self.image_size - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.image_size - 1)
        raster = np.zeros_like(heatmap, dtype=np.uint8)
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
        """replayに水平反転が含まれるか判定する。"""
        return any(
            transform.get("applied", False)
            and str(transform.get("__class_fullname__", "")).endswith("HorizontalFlip")
            for transform in replay.get("transforms", [])
        )

    def _swap_lines(
        self,
        polylines: dict[str, list[list[float]]],
    ) -> dict[str, list[list[float]]]:
        """水平反転後に左右の線channelを入れ替える。"""
        swap = self.augmentation_config.get("hflip_channel_swap")
        if swap is None:
            return polylines
        reordered = [polylines[LINE_KEYS[int(value) - 1]] for value in swap]
        return dict(zip(LINE_KEYS, reordered, strict=True))

    def _apply_transform(
        self,
        ct_images: list[np.ndarray],
        mask_images: list[np.ndarray],
        polylines_by_slice: list[dict[str, list[list[float]]]],
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[dict[str, list[list[float]]]],
    ]:
        """1枚目のreplayをスラブ全体へ共有する。"""
        if self.transform is None:
            return ct_images, mask_images, polylines_by_slice
        transformed_ct: list[np.ndarray] = []
        transformed_masks: list[np.ndarray] = []
        transformed_lines: list[dict[str, list[list[float]]]] = []
        replay: dict[str, Any] | None = None
        for index, (ct_image, mask_image, polylines) in enumerate(
            zip(ct_images, mask_images, polylines_by_slice, strict=True)
        ):
            arguments = {
                "image": ct_image,
                "mask": mask_image,
                "keypoints": polylines["line_1"],
                "line_2": polylines["line_2"],
                "line_3": polylines["line_3"],
                "line_4": polylines["line_4"],
            }
            if index == 0:
                output = self.transform(**arguments)
                replay = output["replay"]
            else:
                if replay is None:
                    raise RuntimeError("augmentation replayが生成されていません")
                output = A.ReplayCompose.replay(replay, **arguments)
            output_lines = {
                "line_1": [list(point[:2]) for point in output["keypoints"]],
                "line_2": [list(point[:2]) for point in output["line_2"]],
                "line_3": [list(point[:2]) for point in output["line_3"]],
                "line_4": [list(point[:2]) for point in output["line_4"]],
            }
            if replay is not None and self._did_flip(replay):
                output_lines = self._swap_lines(output_lines)
            transformed_ct.append(np.asarray(output["image"], dtype=np.float32))
            transformed_masks.append(np.asarray(output["mask"], dtype=np.float32))
            transformed_lines.append(output_lines)
        return transformed_ct, transformed_masks, transformed_lines

    def __getitem__(self, index: int) -> dict[str, Any]:
        """1スラブをtensor辞書として返す。"""
        record = self.records[index]
        ct_images = [self._load_image(path, False) for path in record.image_paths]
        mask_images = [self._load_image(path, True) for path in record.mask_paths]
        polylines = [
            record.labels.get(
                slice_index,
                {key: [] for key in LINE_KEYS},
            )
            for slice_index in record.slice_indices
        ]
        ct_images, mask_images, polylines = self._apply_transform(
            ct_images,
            mask_images,
            polylines,
        )
        input_channels: list[np.ndarray] = []
        heatmaps: list[np.ndarray] = []
        label_mask = np.zeros((len(record.slice_indices), 4), dtype=np.bool_)
        line_params_gt = np.full(
            (len(record.slice_indices), 4, 2),
            np.nan,
            dtype=np.float32,
        )
        for slab_index, (ct_image, mask_image, lines) in enumerate(
            zip(ct_images, mask_images, polylines, strict=True)
        ):
            input_channels.extend(
                [
                    np.clip(ct_image, 0.0, 1.0).astype(np.float32),
                    np.clip(mask_image, 0.0, 1.0).astype(np.float32),
                ]
            )
            line_heatmaps = []
            for line_index, key in enumerate(LINE_KEYS):
                line_heatmaps.append(self._heatmap_from_polyline(lines[key]))
                line_params_gt[slab_index, line_index] = extract_gt_line_params(
                    lines[key],
                    self.image_size,
                )
                label_mask[slab_index, line_index] = not np.isnan(
                    line_params_gt[slab_index, line_index]
                ).any()
            heatmaps.append(np.stack(line_heatmaps, axis=0))
        return {
            "image": torch.from_numpy(np.stack(input_channels, axis=0)),
            "heatmaps": torch.from_numpy(np.stack(heatmaps, axis=0)),
            "label_mask": torch.from_numpy(label_mask),
            "line_params_gt": torch.from_numpy(line_params_gt),
            "slice_indices": torch.as_tensor(record.slice_indices, dtype=torch.long),
            "sample": record.sample,
            "vertebra": record.vertebra,
        }
