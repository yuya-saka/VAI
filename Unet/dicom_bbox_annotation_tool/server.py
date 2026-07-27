"""元DICOMスライスへRSNA骨折bboxを直接描画するアノテーションサーバー。

``train_bounding_boxes.csv`` の座標をリサイズ・再投影せず、対応する元DICOMの
ピクセル座標へそのまま描画する。CSVには椎体レベルが含まれないため、前処理時に
``processing_metadata/*.json`` へ保存された確定済み割当だけを使用する。

既存の ``Unet/fracture_annotation_tool`` とは独立しており、判定結果も既定では
``fracture_region_labels_dicom.csv`` へ保存する。

使い方:
    uv run python Unet/dicom_bbox_annotation_tool/server.py
    uv run python Unet/dicom_bbox_annotation_tool/server.py --port 8767
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import mimetypes
import threading
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qs, urlparse

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset

if TYPE_CHECKING:
    from .mask_overlay import (
        MaskOverlayError,
        VertebraMaskCache,
        VertebraMaskVolume,
        load_mask_volume,
        sample_mask_on_dicom,
    )
elif __package__:
    from .mask_overlay import (
        MaskOverlayError,
        VertebraMaskCache,
        VertebraMaskVolume,
        load_mask_volume,
        sample_mask_on_dicom,
    )
else:
    from mask_overlay import (
        MaskOverlayError,
        VertebraMaskCache,
        VertebraMaskVolume,
        load_mask_volume,
        sample_mask_on_dicom,
    )

ROOT_DIR: Final = Path(__file__).resolve().parents[2]
STATIC_DIR: Final = Path(__file__).resolve().parent
DATA_DIR: Final = ROOT_DIR / "data" / "rsna_data"
DEFAULT_BBOX_CSV: Final = DATA_DIR / "train_bounding_boxes.csv"
DEFAULT_METADATA_DIR: Final = DATA_DIR / "processing_metadata"
DEFAULT_TRAIN_IMAGES_DIR: Final = DATA_DIR / "train_images"
DEFAULT_SEGMENTATION_DIR: Final = DATA_DIR / "segmentations"
DEFAULT_LABEL_CSV: Final = DATA_DIR / "fracture_region_labels_dicom.csv"
DEFAULT_TRAIN_CSV: Final = DATA_DIR / "train.csv"
DEFAULT_AUGMENT_CACHE: Final = STATIC_DIR / ".cache" / "missing_level_recovery.json"

LEVELS: Final = tuple(f"C{index}" for index in range(1, 8))
REQUIRED_BBOX_COLUMNS: Final = {
    "StudyInstanceUID",
    "x",
    "y",
    "width",
    "height",
    "slice_number",
}
REGION_KEYS: Final = tuple(f"region_{index}" for index in range(1, 5))
LABEL_COLUMNS: Final = ("study_id", "level", "run_id", *REGION_KEYS)
WINDOW_LEVEL_HU: Final = 400.0
WINDOW_WIDTH_HU: Final = 2000.0
MAX_REQUEST_BODY_BYTES: Final = 16_384
MAX_IMAGE_CACHE_ITEMS: Final = 256
MASK_TINT_RGBA: Final = (35, 190, 255, 92)


class DicomBboxToolError(ValueError):
    """入力データが安全に表示できない場合に送出する。"""


@dataclass(frozen=True)
class RawBbox:
    """CSVから読み取った変換前の1 bbox行。"""

    csv_line: int
    study_id: str
    slice_number: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DicomBbox:
    """元DICOMファイルと確定椎体レベルを付与した1 bbox行。"""

    csv_line: int
    study_id: str
    level: str
    slice_number: int
    series_index: int
    source_file: str
    x: float
    y: float
    width: float
    height: float

    def as_json(self, row_index: int) -> dict[str, object]:
        """ブラウザ表示用の安全なJSON表現を返す。"""
        return {
            "row_index": row_index,
            "csv_line": self.csv_line,
            "slice_number": self.slice_number,
            "series_index": self.series_index,
            "source_file": self.source_file,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class AnnotationTarget:
    """同一study・椎体・連続bbox区間からなるアノテーション単位。

    ``note`` は前処理metadataの確定割当だけでは骨折椎体にrunが作れなかった
    場合の補足情報: ``coverage_recovered`` はbbox被覆率から椎体を再推定して
    復元したrun、``bbox_missing`` は骨折ラベルがあるのに対応するCSV bboxが
    1行も無いことを示す（画像を持たない警告専用run）。
    """

    target_id: int
    study_id: str
    level: str
    run_id: str
    rows: tuple[DicomBbox, ...]
    note: str | None = None

    @property
    def label_key(self) -> tuple[str, str, str]:
        """ラベルCSVで用いる一意キーを返す。"""
        return self.study_id, self.level, self.run_id

    def as_json(self, annotation: dict[str, int] | None) -> dict[str, object]:
        """ブラウザ表示用の安全なJSON表現を返す。"""
        return {
            "target_id": self.target_id,
            "study_id": self.study_id,
            "short_id": self.study_id.rsplit(".", maxsplit=1)[-1],
            "level": self.level,
            "run_id": self.run_id,
            "note": self.note,
            "annotated": annotation is not None,
            "regions": annotation,
            "rows": [row.as_json(index) for index, row in enumerate(self.rows)],
        }


@dataclass(frozen=True)
class ToolState:
    """HTTPハンドラが参照する読み取り専用データと保存先。"""

    targets: tuple[AnnotationTarget, ...]
    train_images_dir: Path
    segmentation_dir: Path
    label_store: LabelStore
    image_cache: ImageCache
    mask_cache: VertebraMaskCache


class LabelStore:
    """run単位の4領域ラベルをCSVへ原子的に保存する。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> dict[tuple[str, str, str], dict[str, int]]:
        """現在の全ラベルをキー付き辞書として返す。"""
        if not self.path.is_file():
            return {}
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            missing = set(LABEL_COLUMNS) - fieldnames
            if missing:
                raise DicomBboxToolError(
                    f"ラベルCSVの列が不足しています: {sorted(missing)}"
                )
            labels: dict[tuple[str, str, str], dict[str, int]] = {}
            for line_number, row in enumerate(reader, start=2):
                key = (
                    str(row["study_id"]),
                    str(row["level"]),
                    str(row["run_id"]),
                )
                try:
                    regions = {
                        region_key: int(row[region_key]) for region_key in REGION_KEYS
                    }
                except (TypeError, ValueError) as error:
                    raise DicomBboxToolError(
                        f"ラベルCSV {line_number}行目の値が不正です"
                    ) from error
                labels[key] = normalize_regions(regions)
        return labels

    def write(
        self,
        key: tuple[str, str, str],
        regions: dict[str, int],
    ) -> None:
        """1 targetのラベルを既存データへ安全に反映する。"""
        normalized = normalize_regions(regions)
        with self._lock:
            labels = self.read()
            updated = {**labels, key: normalized}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
                writer.writeheader()
                for study_id, level, run_id in sorted(updated):
                    writer.writerow(
                        {
                            "study_id": study_id,
                            "level": level,
                            "run_id": run_id,
                            **updated[(study_id, level, run_id)],
                        }
                    )
            temporary_path.replace(self.path)


class ImageCache:
    """表示済みDICOM PNGを小さなFIFOキャッシュへ保持する。"""

    def __init__(self, max_items: int = MAX_IMAGE_CACHE_ITEMS) -> None:
        if max_items <= 0:
            raise ValueError("max_itemsは正数である必要があります")
        self._max_items = max_items
        self._items: dict[tuple[int, int], bytes] = {}
        self._lock = threading.Lock()

    def get(
        self,
        target: AnnotationTarget,
        row_index: int,
        train_images_dir: Path,
        segmentation_dir: Path,
        mask_cache: VertebraMaskCache,
    ) -> bytes:
        """キャッシュ済みPNGを返し、未生成なら元DICOMから生成する。"""
        key = (target.target_id, row_index)
        with self._lock:
            cached = self._items.get(key)
        if cached is not None:
            return cached
        mask_volume = mask_cache.get(
            segmentation_dir,
            target.study_id,
            target.level,
        )
        rendered = render_dicom_bbox(
            train_images_dir / target.study_id / target.rows[row_index].source_file,
            target.rows[row_index],
            mask_volume=mask_volume,
        )
        with self._lock:
            if len(self._items) >= self._max_items:
                self._items.pop(next(iter(self._items)))
            self._items[key] = rendered
        return rendered


_state: ToolState | None = None


def normalize_regions(data: object) -> dict[str, int]:
    """外部入力を4個の0/1領域フラグへ正規化する。"""
    if not isinstance(data, dict):
        raise DicomBboxToolError("領域ラベルはJSONオブジェクトで指定してください")
    unexpected = set(data) - set(REGION_KEYS)
    if unexpected:
        raise DicomBboxToolError(f"未知の領域キーです: {sorted(unexpected)}")
    result: dict[str, int] = {}
    for region_key in REGION_KEYS:
        value = data.get(region_key, 0)
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int) or value not in (0, 1):
            raise DicomBboxToolError(f"{region_key} は0または1で指定してください")
        result[region_key] = value
    return result


def load_raw_bboxes(path: Path) -> tuple[RawBbox, ...]:
    """RSNA bbox CSVを検証し、座標値を変換せず読み込む。"""
    if not path.is_file():
        raise FileNotFoundError(f"bbox CSVが見つかりません: {path}")
    raw_bboxes: list[RawBbox] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_BBOX_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise DicomBboxToolError(f"bbox CSVの列が不足しています: {sorted(missing)}")
        for csv_line, row in enumerate(reader, start=2):
            try:
                bbox = RawBbox(
                    csv_line=csv_line,
                    study_id=str(row["StudyInstanceUID"]),
                    slice_number=int(row["slice_number"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                )
            except (TypeError, ValueError) as error:
                raise DicomBboxToolError(
                    f"bbox CSV {csv_line}行目の値が不正です"
                ) from error
            values = (bbox.x, bbox.y, bbox.width, bbox.height)
            if not all(math.isfinite(value) for value in values):
                raise DicomBboxToolError(f"bbox CSV {csv_line}行目に非有限値があります")
            if bbox.width <= 0.0 or bbox.height <= 0.0:
                raise DicomBboxToolError(
                    f"bbox CSV {csv_line}行目の幅・高さは正数である必要があります"
                )
            raw_bboxes.append(bbox)
    return tuple(raw_bboxes)


def canonical_slice_to_level(metadata: Mapping[str, object]) -> dict[int, str]:
    """前処理メタデータから確定済みbbox slice→椎体割当を読む。"""
    vertebrae = metadata.get("vertebrae")
    if not isinstance(vertebrae, dict):
        raise DicomBboxToolError("メタデータにvertebraeがありません")
    mapping: dict[int, str] = {}
    for level in LEVELS:
        vertebra = vertebrae.get(level)
        if not isinstance(vertebra, dict):
            continue
        classifier = vertebra.get("classifier_planes")
        if not isinstance(classifier, dict):
            continue
        assigned = classifier.get("assigned_bbox_slice_numbers")
        if assigned is None:
            raise DicomBboxToolError(
                f"メタデータに{level}のassigned_bbox_slice_numbersがありません"
            )
        if not isinstance(assigned, list):
            raise DicomBboxToolError(f"{level}のbbox割当形式が不正です")
        for raw_slice_number in assigned:
            slice_number = int(raw_slice_number)
            previous = mapping.get(slice_number)
            if previous is not None:
                raise DicomBboxToolError(
                    f"slice {slice_number}が{previous}と{level}へ重複割当されています"
                )
            mapping[slice_number] = level
    return mapping


def build_targets(
    bbox_csv: Path,
    metadata_dir: Path,
) -> tuple[AnnotationTarget, ...]:
    """CSV全行を確定椎体別・DICOM系列上の連続run別にまとめる。"""
    raw_bboxes = load_raw_bboxes(bbox_csv)
    bboxes_by_study: dict[str, list[RawBbox]] = {}
    for bbox in raw_bboxes:
        bboxes_by_study.setdefault(bbox.study_id, []).append(bbox)

    grouped_rows: list[tuple[str, str, tuple[DicomBbox, ...]]] = []
    for study_id in sorted(bboxes_by_study):
        metadata_path = metadata_dir / f"{study_id}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"処理メタデータが見つかりません: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise DicomBboxToolError(f"処理メタデータの形式が不正です: {metadata_path}")
        slice_lookup = _build_slice_lookup(metadata)
        slice_to_level = canonical_slice_to_level(metadata)
        rows_by_level: dict[str, list[DicomBbox]] = {}
        for bbox in bboxes_by_study[study_id]:
            slice_info = slice_lookup.get(bbox.slice_number)
            if slice_info is None:
                raise DicomBboxToolError(
                    f"{study_id}: slice {bbox.slice_number}のDICOMが見つかりません"
                )
            level = slice_to_level.get(bbox.slice_number)
            if level is None:
                raise DicomBboxToolError(
                    f"{study_id}: slice {bbox.slice_number}に椎体割当がありません"
                )
            series_index, source_file = slice_info
            rows_by_level.setdefault(level, []).append(
                DicomBbox(
                    csv_line=bbox.csv_line,
                    study_id=study_id,
                    level=level,
                    slice_number=bbox.slice_number,
                    series_index=series_index,
                    source_file=source_file,
                    x=bbox.x,
                    y=bbox.y,
                    width=bbox.width,
                    height=bbox.height,
                )
            )
        for level in LEVELS:
            level_rows = rows_by_level.get(level, [])
            for run in _split_contiguous_rows(level_rows):
                grouped_rows.append((study_id, level, run))

    targets: list[AnnotationTarget] = []
    run_counts: dict[tuple[str, str], int] = {}
    for study_id, level, rows in grouped_rows:
        key = study_id, level
        run_index = run_counts.get(key, 0)
        run_counts[key] = run_index + 1
        targets.append(
            AnnotationTarget(
                target_id=len(targets),
                study_id=study_id,
                level=level,
                run_id=f"run_{run_index:02d}",
                rows=rows,
            )
        )
    assigned_row_count = sum(len(target.rows) for target in targets)
    if assigned_row_count != len(raw_bboxes):
        raise DicomBboxToolError(
            f"bbox割当行数が一致しません: {assigned_row_count} != {len(raw_bboxes)}"
        )
    return tuple(targets)


def _build_slice_lookup(metadata: dict[str, object]) -> dict[int, tuple[int, str]]:
    """slice_numberからDICOM系列位置と元ファイル名を引く辞書を作る。"""
    dicom_geometry = metadata.get("dicom_geometry")
    if not isinstance(dicom_geometry, dict):
        raise DicomBboxToolError("メタデータにdicom_geometryがありません")
    slices = dicom_geometry.get("slices")
    if not isinstance(slices, list):
        raise DicomBboxToolError("メタデータにDICOM slice一覧がありません")
    stem_lookup: dict[int, tuple[int, str]] = {}
    instance_lookup: dict[int, tuple[int, str]] = {}
    for series_index, slice_metadata in enumerate(slices):
        if not isinstance(slice_metadata, dict):
            raise DicomBboxToolError("DICOM sliceメタデータの形式が不正です")
        source_file = str(slice_metadata.get("source_file", ""))
        if not source_file or Path(source_file).name != source_file:
            raise DicomBboxToolError(f"不正なDICOMファイル名です: {source_file}")
        try:
            stem = int(Path(source_file).stem)
        except ValueError:
            stem = None
        if stem is not None:
            if stem in stem_lookup:
                raise DicomBboxToolError(f"DICOMファイル番号が重複しています: {stem}")
            stem_lookup[stem] = series_index, source_file
        instance_number = slice_metadata.get("instance_number")
        if instance_number is not None:
            instance = int(instance_number)
            instance_lookup.setdefault(instance, (series_index, source_file))
    return {**instance_lookup, **stem_lookup}


def _split_contiguous_rows(rows: list[DicomBbox]) -> tuple[tuple[DicomBbox, ...], ...]:
    """数値ファイル名ではなくDICOM系列順の隣接性でrunへ分割する。"""
    ordered = sorted(rows, key=lambda row: row.series_index)
    if not ordered:
        return ()
    indices = [row.series_index for row in ordered]
    if len(indices) != len(set(indices)):
        raise DicomBboxToolError("同じDICOM sliceに複数bbox行があります")
    grouped: list[list[DicomBbox]] = [[ordered[0]]]
    for row in ordered[1:]:
        if row.series_index == grouped[-1][-1].series_index + 1:
            grouped[-1].append(row)
        else:
            grouped.append([row])
    return tuple(tuple(group) for group in grouped)


def load_fracture_labels(path: Path) -> dict[str, dict[str, int]]:
    """train.csvからstudy×椎体の骨折ラベルを読み込む。"""
    if not path.is_file():
        return {}
    labels: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels[str(row["StudyInstanceUID"])] = {
                level: int(row[level]) for level in LEVELS if level in row
            }
    return labels


def _missing_fractured_levels(
    targets: tuple[AnnotationTarget, ...],
    fracture_labels: dict[str, dict[str, int]],
) -> dict[str, list[str]]:
    """bboxを持つstudyのうち、骨折ラベルはあるがrunが無い椎体を列挙する。"""
    levels_with_run: dict[str, set[str]] = {}
    for target in targets:
        levels_with_run.setdefault(target.study_id, set()).add(target.level)
    missing: dict[str, list[str]] = {}
    for study_id, run_levels in levels_with_run.items():
        labels = fracture_labels.get(study_id, {})
        fractured = {level for level in LEVELS if labels.get(level) == 1}
        absent = sorted(fractured - run_levels)
        if absent:
            missing[study_id] = absent
    return missing


def _bbox_coverage(
    dataset: Dataset, bbox: RawBbox, volume: VertebraMaskVolume
) -> float:
    """bbox矩形内における椎体maskの被覆率を返す。"""
    sampled = sample_mask_on_dicom(dataset, volume) > 0
    row_start = max(0, math.floor(bbox.y))
    column_start = max(0, math.floor(bbox.x))
    row_stop = min(
        sampled.shape[0], max(row_start + 1, math.ceil(bbox.y + bbox.height))
    )
    column_stop = min(
        sampled.shape[1], max(column_start + 1, math.ceil(bbox.x + bbox.width))
    )
    patch = sampled[row_start:row_stop, column_start:column_stop]
    return float(patch.mean()) if patch.size else 0.0


def _augmentation_fingerprint(
    bbox_csv: Path, metadata_dir: Path, train_csv: Path, segmentation_dir: Path
) -> dict[str, float]:
    """キャッシュの有効性判定に使う入力ファイルの更新時刻を返す。"""
    return {
        "bbox_csv": bbox_csv.stat().st_mtime,
        "metadata_dir": metadata_dir.stat().st_mtime,
        "train_csv": train_csv.stat().st_mtime,
        "segmentation_dir": segmentation_dir.stat().st_mtime,
    }


def _load_augmentation_cache(
    cache_path: Path, fingerprint: dict[str, float]
) -> list[dict[str, object]] | None:
    """入力ファイルに変更が無ければキャッシュ済みの復元結果を返す。"""
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    added = payload.get("added")
    return added if isinstance(added, list) else None


def _save_augmentation_cache(
    cache_path: Path,
    fingerprint: dict[str, float],
    added: list[dict[str, object]],
) -> None:
    """次回起動を高速化するため、復元結果をJSONへ原子的に保存する。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps({"fingerprint": fingerprint, "added": added}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)


def _serialize_added_target(target: AnnotationTarget) -> dict[str, object]:
    """復元/欠損マークrunをキャッシュ保存用のJSON互換dictへ変換する。"""
    return {
        "study_id": target.study_id,
        "level": target.level,
        "run_id": target.run_id,
        "note": target.note,
        "rows": [
            {
                "csv_line": row.csv_line,
                "slice_number": row.slice_number,
                "series_index": row.series_index,
                "source_file": row.source_file,
                "x": row.x,
                "y": row.y,
                "width": row.width,
                "height": row.height,
            }
            for row in target.rows
        ],
    }


def _deserialize_added_target(
    data: dict[str, object], target_id: int
) -> AnnotationTarget:
    """キャッシュ済みJSON dictから復元/欠損マークrunを復元する。"""
    study_id = str(data["study_id"])
    level = str(data["level"])
    rows = tuple(
        DicomBbox(
            csv_line=int(row["csv_line"]),
            study_id=study_id,
            level=level,
            slice_number=int(row["slice_number"]),
            series_index=int(row["series_index"]),
            source_file=str(row["source_file"]),
            x=float(row["x"]),
            y=float(row["y"]),
            width=float(row["width"]),
            height=float(row["height"]),
        )
        for row in data["rows"]
    )
    return AnnotationTarget(
        target_id=target_id,
        study_id=study_id,
        level=level,
        run_id=str(data["run_id"]),
        rows=rows,
        note=data.get("note"),
    )


def augment_with_missing_fractured_levels(
    targets: tuple[AnnotationTarget, ...],
    bbox_csv: Path,
    metadata_dir: Path,
    train_csv: Path,
    train_images_dir: Path,
    segmentation_dir: Path,
    cache_path: Path = DEFAULT_AUGMENT_CACHE,
) -> tuple[AnnotationTarget, ...]:
    """骨折ラベルはあるがrunが無い椎体を、bbox被覆率で復元 or 欠損マークする。

    前処理metadataの ``assigned_bbox_slice_numbers`` はz範囲だけで椎体を
    割り当てるため、隣接椎体のz範囲が重なる箇所でbbox行が別椎体へ誤って
    割り当てられることがある。ここではtrain.csvで骨折=1なのに対応するrunが
    1つも無い椎体だけを対象に、bbox矩形内の椎体mask被覆率を全レベルで比較し、
    最も被覆するレベルへrunを復元する。該当bboxが1行も見つからない場合は、
    画像を持たない警告専用runとして残し、ツール上で見落としが起きないようにする。
    対象は「runが無い骨折椎体」に限定するため、全235 studyの全レベルを
    再計算するコストは発生しない。

    mask読み込みとDICOMヘッダ読み込みが支配的なコストのため、結果は
    ``cache_path`` へ保存し、入力ファイル(bbox CSV・metadata・train.csv・
    segmentation)の更新時刻が変わらない限り2回目以降はキャッシュを使う。
    """
    fracture_labels = load_fracture_labels(train_csv)
    missing = _missing_fractured_levels(targets, fracture_labels)
    if not missing:
        return targets

    fingerprint = _augmentation_fingerprint(
        bbox_csv, metadata_dir, train_csv, segmentation_dir
    )
    cached = _load_augmentation_cache(cache_path, fingerprint)
    if cached is not None:
        added_from_cache = [
            _deserialize_added_target(data, len(targets) + index)
            for index, data in enumerate(cached)
        ]
        return targets + tuple(added_from_cache)

    raw_bboxes = load_raw_bboxes(bbox_csv)
    bboxes_by_study: dict[str, list[RawBbox]] = {}
    for bbox in raw_bboxes:
        bboxes_by_study.setdefault(bbox.study_id, []).append(bbox)

    added: list[AnnotationTarget] = []
    next_target_id = len(targets)
    for study_id, missing_levels in sorted(missing.items()):
        metadata = json.loads(
            (metadata_dir / f"{study_id}.json").read_text(encoding="utf-8")
        )
        slice_lookup = _build_slice_lookup(metadata)
        volumes: dict[str, VertebraMaskVolume] = {}
        for level in LEVELS:
            try:
                volumes[level] = load_mask_volume(
                    segmentation_dir / study_id / f"vertebrae_{level}.nii.gz"
                )
            except MaskOverlayError:
                continue

        recovered_rows: dict[str, list[DicomBbox]] = {
            level: [] for level in missing_levels
        }
        for bbox in bboxes_by_study.get(study_id, []):
            slice_info = slice_lookup.get(bbox.slice_number)
            if slice_info is None or not volumes:
                continue
            series_index, source_file = slice_info
            dataset = pydicom.dcmread(
                train_images_dir / study_id / source_file, stop_before_pixels=True
            )
            missing_coverage = {
                level: _bbox_coverage(dataset, bbox, volumes[level])
                for level in missing_levels
                if level in volumes
            }
            if not missing_coverage or max(missing_coverage.values()) <= 0.0:
                continue
            all_coverage = {
                level: _bbox_coverage(dataset, bbox, volume)
                for level, volume in volumes.items()
            }
            best_level = max(all_coverage, key=lambda level: all_coverage[level])
            if best_level not in recovered_rows or all_coverage[best_level] <= 0.0:
                continue
            recovered_rows[best_level].append(
                DicomBbox(
                    csv_line=bbox.csv_line,
                    study_id=study_id,
                    level=best_level,
                    slice_number=bbox.slice_number,
                    series_index=series_index,
                    source_file=source_file,
                    x=bbox.x,
                    y=bbox.y,
                    width=bbox.width,
                    height=bbox.height,
                )
            )

        for level in missing_levels:
            rows = tuple(
                sorted(recovered_rows.get(level, []), key=lambda row: row.series_index)
            )
            added.append(
                AnnotationTarget(
                    target_id=next_target_id,
                    study_id=study_id,
                    level=level,
                    run_id="supplement" if rows else "missing",
                    rows=rows,
                    note="coverage_recovered" if rows else "bbox_missing",
                )
            )
            next_target_id += 1

    _save_augmentation_cache(
        cache_path,
        fingerprint,
        [_serialize_added_target(target) for target in added],
    )
    return targets + tuple(added)


def apply_bone_window(
    hu_values: np.ndarray,
    *,
    level_hu: float = WINDOW_LEVEL_HU,
    width_hu: float = WINDOW_WIDTH_HU,
) -> np.ndarray:
    """元解像度を維持したまま固定bone windowをuint8へ変換する。"""
    if width_hu <= 0.0:
        raise ValueError("window幅は正数である必要があります")
    lower = level_hu - width_hu / 2.0
    upper = level_hu + width_hu / 2.0
    normalized = (np.clip(hu_values, lower, upper) - lower) / (upper - lower)
    return np.asarray(normalized * 255.0, dtype=np.uint8)


def render_dicom_bbox(
    dicom_path: Path,
    bbox: DicomBbox,
    *,
    mask_volume: VertebraMaskVolume | None = None,
) -> bytes:
    """元サイズDICOMへ割当椎体maskとCSV矩形を重ねる。"""
    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOMが見つかりません: {dicom_path}")
    dataset = pydicom.dcmread(dicom_path)
    try:
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
        slope = float(dataset.get("RescaleSlope", 1.0))
        intercept = float(dataset.get("RescaleIntercept", 0.0))
    except (AttributeError, TypeError, ValueError) as error:
        raise DicomBboxToolError(f"DICOM画素を読み込めません: {dicom_path}") from error
    if pixels.ndim != 2:
        raise DicomBboxToolError(f"2次元でないDICOMです: {dicom_path}")
    hu_values = pixels * slope + intercept
    if not np.all(np.isfinite(hu_values)):
        raise DicomBboxToolError(f"DICOMに非有限画素があります: {dicom_path}")
    windowed = apply_bone_window(hu_values)
    if str(dataset.get("PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        windowed = 255 - windowed

    image = Image.fromarray(windowed, mode="L").convert("RGBA")
    if mask_volume is not None:
        sampled_mask = sample_mask_on_dicom(dataset, mask_volume) > 0
        mask_rgba = np.zeros((*sampled_mask.shape, 4), dtype=np.uint8)
        mask_rgba[sampled_mask] = MASK_TINT_RGBA
        image = Image.alpha_composite(image, Image.fromarray(mask_rgba, mode="RGBA"))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(min(image.size) * 0.006))
    rectangle = (
        bbox.x,
        bbox.y,
        bbox.x + bbox.width,
        bbox.y + bbox.height,
    )
    draw.rectangle(rectangle, fill=(255, 30, 30, 42))
    draw.rectangle(rectangle, outline=(255, 48, 48, 255), width=line_width)
    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue()


class DicomBboxAnnotationHandler(BaseHTTPRequestHandler):
    """DICOM bbox画像、対象一覧、独立ラベル保存APIを提供する。"""

    timeout = 30

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
        elif parsed.path == "/api/samples":
            self._api_samples()
        elif parsed.path == "/api/image":
            self._api_image(params)
        else:
            self._send_bytes(404, b"Not Found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotation":
            self._send_bytes(404, b"Not Found", "text/plain; charset=utf-8")
            return
        self._api_annotation(parse_qs(parsed.query))

    def _api_samples(self) -> None:
        state = _require_state()
        try:
            labels = state.label_store.read()
        except (OSError, DicomBboxToolError) as error:
            self._send_json({"error": str(error)}, status=500)
            return
        self._send_json(
            [target.as_json(labels.get(target.label_key)) for target in state.targets]
        )

    def _api_image(self, params: dict[str, list[str]]) -> None:
        state = _require_state()
        target_id = self._integer_param(params, "target")
        row_index = self._integer_param(params, "row")
        if target_id is None or row_index is None:
            self._send_json({"error": "targetとrowが必要です"}, status=400)
            return
        if not 0 <= target_id < len(state.targets):
            self._send_json({"error": "targetが範囲外です"}, status=400)
            return
        target = state.targets[target_id]
        if not 0 <= row_index < len(target.rows):
            self._send_json({"error": "rowが範囲外です"}, status=400)
            return
        try:
            png_bytes = state.image_cache.get(
                target,
                row_index,
                state.train_images_dir,
                state.segmentation_dir,
                state.mask_cache,
            )
        except (OSError, DicomBboxToolError, MaskOverlayError) as error:
            self._send_json({"error": str(error)}, status=500)
            return
        self._send_bytes(
            200, png_bytes, "image/png", cache_control="private, max-age=3600"
        )

    def _api_annotation(self, params: dict[str, list[str]]) -> None:
        state = _require_state()
        target_id = self._integer_param(params, "target")
        if target_id is None or not 0 <= target_id < len(state.targets):
            self._send_json({"error": "targetが不正です"}, status=400)
            return
        if not state.targets[target_id].rows:
            self._send_json(
                {"error": "bboxが存在しないrunのため保存できません"}, status=400
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Content-Lengthが不正です"}, status=400)
            return
        if not 0 < content_length <= MAX_REQUEST_BODY_BYTES:
            self._send_json({"error": "リクエストサイズが不正です"}, status=400)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            regions = normalize_regions(payload)
            state.label_store.write(state.targets[target_id].label_key, regions)
        except (json.JSONDecodeError, OSError, DicomBboxToolError) as error:
            self._send_json({"error": str(error)}, status=400)
            return
        self._send_json({"ok": True})

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_bytes(404, b"Not Found", "text/plain; charset=utf-8")
            return
        mime_type, _ = mimetypes.guess_type(path.name)
        self._send_bytes(
            200,
            path.read_bytes(),
            mime_type or "application/octet-stream",
        )

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _integer_param(params: dict[str, list[str]], key: str) -> int | None:
        values = params.get(key)
        if not values:
            return None
        try:
            return int(values[0])
        except ValueError:
            return None

    def log_message(self, format: str, *args: object) -> None:
        """通常のアクセスログは抑制する。"""


def _require_state() -> ToolState:
    """初期化済みのアプリケーション状態を返す。"""
    if _state is None:
        raise RuntimeError("サーバー状態が初期化されていません")
    return _state


def main() -> None:
    """CLI引数を読み、localhost限定HTTPサーバーを起動する。"""
    parser = argparse.ArgumentParser(
        description="元DICOMへRSNA bboxを直接描画する独立アノテーションツール"
    )
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--bbox-csv", type=Path, default=DEFAULT_BBOX_CSV)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=DEFAULT_TRAIN_IMAGES_DIR,
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
    )
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args()

    if not arguments.train_images_dir.is_dir():
        raise FileNotFoundError(
            f"DICOMディレクトリが見つかりません: {arguments.train_images_dir}"
        )
    if not arguments.segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"椎体maskディレクトリが見つかりません: {arguments.segmentation_dir}"
        )
    targets = build_targets(arguments.bbox_csv, arguments.metadata_dir)
    targets = augment_with_missing_fractured_levels(
        targets,
        arguments.bbox_csv,
        arguments.metadata_dir,
        arguments.train_csv,
        arguments.train_images_dir,
        arguments.segmentation_dir,
    )
    global _state
    _state = ToolState(
        targets=targets,
        train_images_dir=arguments.train_images_dir,
        segmentation_dir=arguments.segmentation_dir,
        label_store=LabelStore(arguments.label_csv),
        image_cache=ImageCache(),
        mask_cache=VertebraMaskCache(),
    )
    bbox_row_count = sum(len(target.rows) for target in targets)
    study_count = len({target.study_id for target in targets})
    recovered_count = sum(
        1 for target in targets if target.note == "coverage_recovered"
    )
    missing_count = sum(1 for target in targets if target.note == "bbox_missing")
    print(
        f"対象: {study_count} studies / {len(targets)} runs / {bbox_row_count} bbox rows"
    )
    print("椎体割当: processing_metadataのassigned_bbox_slice_numbers")
    print(
        f"骨折ラベル補完: 被覆率から復元 {recovered_count} runs / "
        f"bbox欠損の警告 {missing_count} runs"
    )
    print(f"ラベル保存先: {arguments.label_csv}")
    url = f"http://localhost:{arguments.port}"
    print(f"起動: {url}")
    print("終了: Ctrl+C")

    server = ThreadingHTTPServer(
        ("localhost", arguments.port), DicomBboxAnnotationHandler
    )
    server.daemon_threads = True
    if not arguments.no_browser:
        Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバー停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
