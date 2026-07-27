"""骨折領域アノテーションツール HTTPサーバー。

bbox中心データセット (`data/rsna_data/bbox_centered_dataset/{study}/{level}/run_XX/`)
の各runについて、15プレーンを確認しながら4領域のどれに骨折があるかをラベル付けする。
1つの(study, level)が複数run（非連続なbbox区間）を持つ場合があるため、
run単位でアノテーション対象を扱う。

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

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data" / "rsna_data"
BBOX_CENTERED_DATASET_DIR = DATA_DIR / "bbox_centered_dataset"
LABEL_CSV = DATA_DIR / "fracture_region_labels.csv"

PLANE_COUNT = 15
CENTER_CHANNEL = 2  # ct.npy (15,5,224,224) の中央チャンネル
DEFAULT_RUN_ID = "run_00"  # run_id列がない旧CSV行の扱い

# 4領域の色 (RGB)。data_preprocessing/rsna_pipeline/visualize_bbox_centered.py と揃える。
REGION_COLORS: dict[int, tuple[int, int, int]] = {
    1: (100, 149, 237),  # R1 椎体: cornflower blue
    2: (50, 205, 50),  # R2 右椎間孔: lime green
    3: (220, 80, 80),  # R3 左椎間孔: tomato
    4: (255, 215, 0),  # R4 後方要素: gold
}
NO_REGION_TINT = (80, 180, 255)  # region_4class.npy が無い場合の椎体強調色

# bbox_corrected_occupancy.npy は4倍supersampling由来のuint8[0,255]。
# 255が完全占有、alphaは最大でも0.5に抑えて下地(領域色/CT)を見えるようにする。
OCCUPANCY_TINT = (255, 40, 40)
OCCUPANCY_ALPHA_MAX = 0.5
OCCUPANCY_ALPHA_SCALE = 510.0

CONTOUR_COLOR = (255, 255, 0)
CONTOUR_SIMPLIFY_TOLERANCE_PX = 0.75  # visualize_bbox_centered.py と同一の表示簡略化

_targets: list[dict] | None = None
_data_lock = threading.Lock()

# (study_id, level, run_id) → 15枚のPNG bytesリスト
_plane_cache: dict[tuple[str, str, str], list[bytes]] = {}
_cache_lock = threading.Lock()
MAX_CACHE = 15


def load_data() -> None:
    """起動時にbbox中心データセットの全runを対象リストとして読み込む。

    bbox_centered_dataset は元々bboxが割り当てられた陽性椎体からのみ生成されているため、
    train.csvとの突合は不要（旧ツールのfracture_dataset+fracture_bbox_planes.csv経路とは異なる）。
    """
    global _targets

    targets: list[dict] = []
    for run_dir in sorted(BBOX_CENTERED_DATASET_DIR.glob("*/*/run_*")):
        study_id = run_dir.parent.parent.name
        level = run_dir.parent.name
        run_id = run_dir.name
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        targets.append({
            "study_id": study_id,
            "level": level,
            "run_id": run_id,
            "short_id": study_id.split(".")[-1],
            "has_region": (run_dir / "region_4class.npy").exists(),
            "geometry_mode": metadata.get("geometry_mode", "unknown"),
        })
    _targets = targets


def read_labels() -> dict[tuple[str, str, str], dict]:
    """CSVから {(study_id, level, run_id): {region_1:0,...}} を返す。"""
    if not LABEL_CSV.exists():
        return {}
    df = pd.read_csv(LABEL_CSV)
    if "run_id" not in df.columns:
        # 旧スキーマ（run未対応時代）の行はrun_00とみなす
        df["run_id"] = DEFAULT_RUN_ID
    result: dict[tuple[str, str, str], dict] = {}
    for _, row in df.iterrows():
        key = (str(row["study_id"]), str(row["level"]), str(row["run_id"]))
        result[key] = {
            "region_1": int(row["region_1"]),
            "region_2": int(row["region_2"]),
            "region_3": int(row["region_3"]),
            "region_4": int(row["region_4"]),
        }
    return result


def write_label(study_id: str, level: str, run_id: str, regions: dict) -> None:
    """1件のラベルをCSVに書き込む（既存行は上書き）。"""
    with _data_lock:
        if LABEL_CSV.exists():
            df = pd.read_csv(LABEL_CSV)
            if "run_id" not in df.columns:
                df["run_id"] = DEFAULT_RUN_ID
        else:
            df = pd.DataFrame(
                columns=[
                    "study_id",
                    "level",
                    "run_id",
                    "region_1",
                    "region_2",
                    "region_3",
                    "region_4",
                ]
            )
        mask = ~(
            (df["study_id"] == study_id)
            & (df["level"] == level)
            & (df["run_id"] == run_id)
        )
        df = df[mask]
        new_row = pd.DataFrame([{
            "study_id": study_id,
            "level": level,
            "run_id": run_id,
            "region_1": int(regions.get("region_1", 0)),
            "region_2": int(regions.get("region_2", 0)),
            "region_3": int(regions.get("region_3", 0)),
            "region_4": int(regions.get("region_4", 0)),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values(["study_id", "level", "run_id"]).reset_index(drop=True)
        df.to_csv(LABEL_CSV, index=False)


def _simplify_contour(component: list[list[float]]) -> np.ndarray | None:
    """スーパーサンプリングの階段状ノイズを除去した表示用輪郭 (row, col) を返す。

    保存済みoccupancy自体は変更しない（表示のみの簡略化）。
    """
    polygon = np.asarray(component, dtype=np.float32)
    if len(polygon) < 3:
        return None
    simplified = cv2.approxPolyDP(
        polygon[:, ::-1].reshape(-1, 1, 2),  # (row,col) -> cv2の(x,y)=(col,row)
        epsilon=CONTOUR_SIMPLIFY_TOLERANCE_PX,
        closed=True,
    ).reshape(-1, 2)
    return simplified[:, ::-1]  # (col,row) -> (row,col)


def render_planes(study_id: str, level: str, run_id: str) -> list[bytes]:
    """15枚のプレーン画像をPNG bytesのリストで生成する。

    bboxは矩形ではなく、`bbox_corrected_occupancy.npy`（3D envelopeと補正断面の交差の
    partial occupancy）を半透明の赤で塗り、`bbox_corrected_contours.json` の輪郭を
    黄色線で重ねて表示する。形状は三角形・非凸・複数componentになり得る。
    """
    run_dir = BBOX_CENTERED_DATASET_DIR / study_id / level / run_id
    ct = np.load(run_dir / "ct.npy")  # (15, 5, 224, 224)
    vmask = np.load(run_dir / "vertebra_mask.npy")  # (15, 224, 224)
    occupancy = np.load(run_dir / "bbox_corrected_occupancy.npy")  # (15, 224, 224)
    contours = json.loads(
        (run_dir / "bbox_corrected_contours.json").read_text(encoding="utf-8")
    )
    contour_by_plane = {int(item["plane_index"]): item["components"] for item in contours}

    region_path = run_dir / "region_4class.npy"
    region_mask = np.load(region_path) if region_path.exists() else None

    result: list[bytes] = []
    for pi in range(PLANE_COUNT):
        ct_plane = ct[pi, CENTER_CHANNEL].astype(np.uint8)
        rgb = np.stack([ct_plane] * 3, axis=-1).astype(np.float32)
        vmask_bool = vmask[pi] > 0

        if region_mask is not None:
            overlay = rgb.copy()
            for r, color in REGION_COLORS.items():
                region_pixels = (region_mask[pi] == r) & vmask_bool
                overlay[region_pixels] = color
            blended = 0.55 * rgb + 0.45 * overlay
        else:
            # region_4class.npyが未生成のrun（QC除外分）はCT+椎体強調のみ表示
            blended = rgb.copy()
            tint = np.array(NO_REGION_TINT, dtype=np.float32)
            blended[vmask_bool] = 0.8 * blended[vmask_bool] + 0.2 * tint

        alpha = np.clip(
            occupancy[pi].astype(np.float32) / OCCUPANCY_ALPHA_SCALE,
            0.0,
            OCCUPANCY_ALPHA_MAX,
        )[..., None]
        tint = np.array(OCCUPANCY_TINT, dtype=np.float32)
        blended = blended * (1.0 - alpha) + tint * alpha
        blended = blended.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(blended, mode="RGB")
        draw = ImageDraw.Draw(img)
        for component in contour_by_plane.get(pi, []):
            polygon = _simplify_contour(component)
            if polygon is None:
                continue
            points = [(float(c), float(r)) for r, c in polygon]
            draw.line([*points, points[0]], fill=CONTOUR_COLOR, width=1)

        buf = io.BytesIO()
        img.save(buf, "PNG")
        result.append(buf.getvalue())

    return result


def get_cached_planes(study_id: str, level: str, run_id: str) -> list[bytes]:
    """キャッシュからプレーン画像を取得、なければ生成する。"""
    key = (study_id, level, run_id)
    with _cache_lock:
        if key not in _plane_cache:
            if len(_plane_cache) >= MAX_CACHE:
                _plane_cache.pop(next(iter(_plane_cache)))
            _plane_cache[key] = render_planes(study_id, level, run_id)
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
            key = (t["study_id"], t["level"], t["run_id"])
            ann = labels.get(key)
            result.append({
                "study_id": t["study_id"],
                "short_id": t["short_id"],
                "level": t["level"],
                "run_id": t["run_id"],
                "annotated": ann is not None,
                "has_region": t["has_region"],
                "geometry_mode": t["geometry_mode"],
                "regions": ann,
            })
        self._send_json(result)

    def _api_image(self, params: dict) -> None:
        study_id = self._param(params, "study")
        level = self._param(params, "level")
        run_id = self._param(params, "run")
        plane_str = self._param(params, "plane")

        if not all([study_id, level, run_id, plane_str]):
            self._send_bytes(400, b"Missing params")
            return

        try:
            plane_idx = int(plane_str)
        except ValueError:
            self._send_bytes(400, b"Invalid plane")
            return

        if not (0 <= plane_idx <= PLANE_COUNT - 1):
            self._send_bytes(400, b"Plane out of range")
            return

        try:
            planes = get_cached_planes(study_id, level, run_id)
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
        run_id = self._param(params, "run")
        if not all([study_id, level, run_id]):
            self._send_json({})
            return
        labels = read_labels()
        ann = labels.get((study_id, level, run_id), {})
        self._send_json(ann)

    def _api_post_annotation(self, params: dict) -> None:
        study_id = self._param(params, "study")
        level = self._param(params, "level")
        run_id = self._param(params, "run")
        if not all([study_id, level, run_id]):
            self._send_json({"error": "missing params"}, 400)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        write_label(study_id, level, run_id, data)
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
    print(f"対象 {len(_targets)} 件 (run単位)")

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
