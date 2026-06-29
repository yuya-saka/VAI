"""骨折領域アノテーションツール HTTPサーバー。

465件の (study, level) について、15プレーンを確認しながら
4領域のどれに骨折があるかをラベル付けする。

使い方:
    uv run python Unet/fracture_annotation_tool/server.py
    uv run python Unet/fracture_annotation_tool/server.py --port 8766
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data" / "rsna_data"
FRACTURE_DATASET_DIR = DATA_DIR / "fracture_dataset"
META_DIR = DATA_DIR / "processing_metadata"
BBOX_CSV = DATA_DIR / "train_bounding_boxes.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
LABEL_CSV = DATA_DIR / "fracture_region_labels.csv"

SAMPLING_PS = 0.4
OUTPUT_SIZE = 224
HALF_PX = (OUTPUT_SIZE - 1) / 2.0
LEVEL_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# 4領域の色 (RGB)
REGION_COLORS: dict[int, tuple[int, int, int]] = {
    1: (100, 149, 237),  # R1 前左: cornflower blue
    2: (50, 205, 50),    # R2 前右: lime green
    3: (220, 80, 80),    # R3 後左: tomato
    4: (255, 215, 0),    # R4 後右: gold
}

_targets: list[dict] | None = None
_bbox_df: pd.DataFrame | None = None
_data_lock = threading.Lock()

# (study_id, level) → 15枚のPNG bytesリスト
_plane_cache: dict[tuple[str, str], list[bytes]] = {}
_cache_lock = threading.Lock()
MAX_CACHE = 15


def load_data() -> None:
    """起動時に対象リストとbboxデータを読み込む。has_bboxを椎体レベルで判定する。"""
    global _targets, _bbox_df
    bbox_df = pd.read_csv(BBOX_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    _bbox_df = bbox_df

    studies_with_bbox = set(bbox_df["StudyInstanceUID"].unique())

    candidates: list[tuple[str, str]] = []
    for _, row in train_df.iterrows():
        sid = row["StudyInstanceUID"]
        if sid not in studies_with_bbox:
            continue
        for level in LEVEL_COLS:
            if row[level] != 1:
                continue
            if not (FRACTURE_DATASET_DIR / sid / level / "region_4class.npy").exists():
                continue
            candidates.append((sid, level))

    # メタデータをstudy単位で1回だけ読み、levelレベルでhas_bboxを判定
    study_to_levels: dict[str, list[str]] = {}
    for sid, level in candidates:
        study_to_levels.setdefault(sid, []).append(level)

    has_bbox_map: dict[tuple[str, str], bool] = {}
    for sid, levels in study_to_levels.items():
        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == sid]
        meta_path = META_DIR / f"{sid}.json"
        if study_bboxes.empty or not meta_path.exists():
            for lv in levels:
                has_bbox_map[(sid, lv)] = False
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            slices_meta = meta["dicom_geometry"]["slices"]
            sn_to_pos: dict[int, float] = {}
            for s in slices_meta:
                stem = Path(s["source_file"]).stem
                try:
                    sn_to_pos[int(stem)] = float(s["slice_position_mm"])
                except ValueError:
                    pass
                if s["instance_number"] is not None:
                    sn_to_pos.setdefault(int(s["instance_number"]), float(s["slice_position_mm"]))
            for lv in levels:
                found = False
                try:
                    full_range = meta["vertebrae"][lv]["classifier_planes"]["full_range_mm"]
                    z_min, z_max = min(full_range), max(full_range)
                    for sn in study_bboxes["slice_number"]:
                        sp = sn_to_pos.get(int(sn))
                        if sp is not None and z_min <= sp <= z_max:
                            found = True
                            break
                except (KeyError, ValueError):
                    pass
                has_bbox_map[(sid, lv)] = found
        except Exception:
            for lv in levels:
                has_bbox_map[(sid, lv)] = False

    targets: list[dict] = [
        {
            "study_id": sid,
            "level": level,
            "short_id": sid.split(".")[-1],
            "has_bbox": has_bbox_map.get((sid, level), False),
        }
        for sid, level in candidates
    ]
    no_bbox = sum(1 for t in targets if not t["has_bbox"])
    if no_bbox:
        print(f"  うちbboxなし: {no_bbox} 件")
    _targets = targets


def read_labels() -> dict[tuple[str, str], dict]:
    """CSVから {(study_id, level): {region_1:0,...}} を返す。"""
    if not LABEL_CSV.exists():
        return {}
    df = pd.read_csv(LABEL_CSV)
    result: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        key = (str(row["study_id"]), str(row["level"]))
        result[key] = {
            "region_1": int(row["region_1"]),
            "region_2": int(row["region_2"]),
            "region_3": int(row["region_3"]),
            "region_4": int(row["region_4"]),
        }
    return result


def write_label(study_id: str, level: str, regions: dict) -> None:
    """1件のラベルをCSVに書き込む（既存行は上書き）。"""
    with _data_lock:
        if LABEL_CSV.exists():
            df = pd.read_csv(LABEL_CSV)
        else:
            df = pd.DataFrame(
                columns=["study_id", "level", "region_1", "region_2", "region_3", "region_4"]
            )
        mask = ~((df["study_id"] == study_id) & (df["level"] == level))
        df = df[mask]
        new_row = pd.DataFrame([{
            "study_id": study_id,
            "level": level,
            "region_1": int(regions.get("region_1", 0)),
            "region_2": int(regions.get("region_2", 0)),
            "region_3": int(regions.get("region_3", 0)),
            "region_4": int(regions.get("region_4", 0)),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values(["study_id", "level"]).reset_index(drop=True)
        df.to_csv(LABEL_CSV, index=False)


def _build_slice_mapping(slices_meta: list[dict]) -> dict[int, int]:
    """slice_number → slices_metaインデックスのマッピングを作る。
    ファイルstem数値を優先し、instance_numberを補助にする（_slice_index_by_numberと同ロジック）。
    """
    result: dict[int, int] = {}
    for idx, s in enumerate(slices_meta):
        stem = Path(s["source_file"]).stem
        try:
            result[int(stem)] = idx
        except ValueError:
            pass
        if s["instance_number"] is not None:
            result.setdefault(s["instance_number"], idx)
    return result


def render_planes(study_id: str, level: str) -> list[bytes]:
    """15枚のプレーン画像をPNG bytesのリストで生成する。"""
    level_dir = FRACTURE_DATASET_DIR / study_id / level
    ct = np.load(level_dir / "ct.npy")               # (15, 5, 224, 224)
    vmask = np.load(level_dir / "vertebra_mask.npy")  # (15, 224, 224)
    region_mask = np.load(level_dir / "region_4class.npy")  # (15, 224, 224)

    with open(META_DIR / f"{study_id}.json") as f:
        meta = json.load(f)

    slices_meta = meta["dicom_geometry"]["slices"]
    row_dir = np.array(meta["dicom_geometry"]["row_direction_lps"])
    col_dir = np.array(meta["dicom_geometry"]["column_direction_lps"])
    ps_row, ps_col = meta["dicom_geometry"]["pixel_spacing_row_column_mm"]

    sn_to_idx = _build_slice_mapping(slices_meta)
    planes_meta = meta["vertebrae"][level]["classifier_planes"]["planes"]
    full_range = meta["vertebrae"][level]["classifier_planes"]["full_range_mm"]
    z_min, z_max = min(full_range), max(full_range)
    plane_positions = np.array([p["position_mm"] for p in planes_meta])

    # bboxを最近傍プレーンに割り当て
    plane_bbox_list: dict[int, list[dict]] = {i: [] for i in range(len(planes_meta))}
    for _, row in _bbox_df[_bbox_df["StudyInstanceUID"] == study_id].iterrows():
        sn = int(row["slice_number"])
        sidx = sn_to_idx.get(sn)
        if sidx is None:
            continue
        sp = slices_meta[sidx]["slice_position_mm"]
        if not (z_min <= sp <= z_max):
            continue
        nearest = int(np.argmin(np.abs(plane_positions - sp)))
        ip = slices_meta[sidx]["image_position_lps_mm"]
        plane_bbox_list[nearest].append({
            "x": float(row["x"]),
            "y": float(row["y"]),
            "w": float(row["width"]),
            "h": float(row["height"]),
            "img_pos": np.array(ip, dtype=np.float64),
        })

    def bbox_to_224(b: dict, plane: dict) -> tuple[float, float, float, float]:
        """DICOMのbbox → 224×224プレーン座標 (row_min, col_min, row_max, col_max)。"""
        center = np.array(plane["center_lps_mm"], dtype=np.float64)
        row_basis = np.array(plane["row_basis_lps"], dtype=np.float64)
        col_basis = np.array(plane["column_basis_lps"], dtype=np.float64)
        corners = [
            (b["x"], b["y"]),
            (b["x"] + b["w"], b["y"]),
            (b["x"] + b["w"], b["y"] + b["h"]),
            (b["x"], b["y"] + b["h"]),
        ]
        rs, cs = [], []
        for cx, cy in corners:
            lps = b["img_pos"] + cx * ps_col * row_dir + cy * ps_row * col_dir
            delta = lps - center
            # plane_sampling.py の _patient_coordinates の逆変換
            # row_offset ↔ col_basis, col_offset ↔ row_basis
            rs.append(float(np.dot(delta, col_basis)) / SAMPLING_PS + HALF_PX)
            cs.append(float(np.dot(delta, row_basis)) / SAMPLING_PS + HALF_PX)
        return min(rs), min(cs), max(rs), max(cs)

    result: list[bytes] = []
    for pi in range(15):
        ct_plane = ct[pi, 2].astype(np.uint8)  # 中央チャンネル (224, 224)
        rgb = np.stack([ct_plane] * 3, axis=-1).astype(np.float32)

        # region_4classカラーオーバーレイ (alpha=0.45)
        overlay = rgb.copy()
        for r, color in REGION_COLORS.items():
            mask = (region_mask[pi] == r) & (vmask[pi] > 0)
            overlay[mask] = color
        blended = (0.55 * rgb + 0.45 * overlay).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(blended, mode="RGB")
        draw = ImageDraw.Draw(img)

        # bboxを白矩形で描画
        for b in plane_bbox_list[pi]:
            try:
                r0, c0, r1, c1 = bbox_to_224(b, planes_meta[pi])
                x0 = max(0, int(c0))
                y0 = max(0, int(r0))
                x1 = min(OUTPUT_SIZE - 1, int(c1))
                y1 = min(OUTPUT_SIZE - 1, int(r1))
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 255, 255), width=2)
            except Exception:
                pass

        buf = io.BytesIO()
        img.save(buf, "PNG")
        result.append(buf.getvalue())

    return result


def get_cached_planes(study_id: str, level: str) -> list[bytes]:
    """キャッシュからプレーン画像を取得、なければ生成する。"""
    key = (study_id, level)
    with _cache_lock:
        if key not in _plane_cache:
            if len(_plane_cache) >= MAX_CACHE:
                _plane_cache.pop(next(iter(_plane_cache)))
            _plane_cache[key] = render_planes(study_id, level)
        return _plane_cache[key]


class FractureAnnotationHandler(BaseHTTPRequestHandler):
    """骨折領域アノテーションAPIとスタティックファイルを提供するハンドラ。"""

    timeout = 30

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
        elif path == "/api/samples":
            self._api_samples()
        elif path == "/api/image":
            self._api_image(params)
        elif path == "/api/annotation":
            self._api_get_annotation(params)
        else:
            self._send_bytes(404, b"Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotation":
            params = parse_qs(parsed.query)
            self._api_post_annotation(params)
        else:
            self._send_bytes(404, b"Not Found")

    def _api_samples(self) -> None:
        labels = read_labels()
        result = []
        for t in _targets:
            key = (t["study_id"], t["level"])
            ann = labels.get(key)
            result.append({
                "study_id": t["study_id"],
                "short_id": t["short_id"],
                "level": t["level"],
                "annotated": ann is not None,
                "has_bbox": t.get("has_bbox", True),
                "regions": ann,
            })
        self._send_json(result)

    def _api_image(self, params: dict) -> None:
        study_id = self._param(params, "study")
        level = self._param(params, "level")
        plane_str = self._param(params, "plane")

        if not all([study_id, level, plane_str]):
            self._send_bytes(400, b"Missing params")
            return

        try:
            plane_idx = int(plane_str)
        except ValueError:
            self._send_bytes(400, b"Invalid plane")
            return

        if not (0 <= plane_idx <= 14):
            self._send_bytes(400, b"Plane out of range")
            return

        try:
            planes = get_cached_planes(study_id, level)
        except Exception as e:
            self._send_bytes(500, f"Render error: {e}".encode())
            return

        png_bytes = planes[plane_idx]
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_bytes)))
        self.end_headers()
        self.wfile.write(png_bytes)

    def _api_get_annotation(self, params: dict) -> None:
        study_id = self._param(params, "study")
        level = self._param(params, "level")
        if not all([study_id, level]):
            self._send_json({})
            return
        labels = read_labels()
        ann = labels.get((study_id, level), {})
        self._send_json(ann)

    def _api_post_annotation(self, params: dict) -> None:
        study_id = self._param(params, "study")
        level = self._param(params, "level")
        if not all([study_id, level]):
            self._send_json({"error": "missing params"}, 400)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        write_label(study_id, level, data)
        self._send_json({"ok": True})

    def _serve_file(self, path: Path) -> None:
        if not path.exists():
            self._send_bytes(404, b"File not found")
            return
        content = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _param(params: dict, key: str) -> str | None:
        vals = params.get(key)
        return vals[0] if vals else None

    def log_message(self, format: str, *args: object) -> None:
        pass  # アクセスログを抑制


def main() -> None:
    parser = argparse.ArgumentParser(description="骨折領域アノテーションツールサーバー")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    print("データ読み込み中...")
    load_data()
    print(f"対象 {len(_targets)} 件")

    url = f"http://localhost:{args.port}"
    print(f"起動: {url}")
    print("終了: Ctrl+C")

    server = ThreadingHTTPServer(("localhost", args.port), FractureAnnotationHandler)
    server.daemon_threads = True
    Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバー停止。")


if __name__ == "__main__":
    main()
