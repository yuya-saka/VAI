"""
アノテーションツール用 HTTP サーバー。

追加パッケージ不要（Python 標準ライブラリのみ）。

使い方:
    uv run python Unet/annotation_tool/server.py
    uv run python Unet/annotation_tool/server.py --port 8765
    uv run python Unet/annotation_tool/server.py --dataset /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

# ライン番号 → (R, G, B) — convert_to_png.py と同色（BGRをRGBに変換）
_LINE_COLORS: dict[str, tuple[int, int, int]] = {
    "line_1": (  0, 255, 255),  # シアン
    "line_2": (255,   0, 255),  # マゼンタ
    "line_3": (  0, 255,   0),  # 緑
    "line_4": (255, 255,   0),  # 黄
}

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent

_dataset_dir: Path = ROOT_DIR / "dataset"


class AnnotationHandler(BaseHTTPRequestHandler):
    """アノテーション API とスタティックファイルを提供するリクエストハンドラ。"""
    timeout = 30  # 半端な接続で永久待機しないための読み取りタイムアウト（秒）

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
        elif path == "/api/samples":
            self._api_get_samples()
        elif path == "/api/image":
            self._api_get_image(params)
        elif path == "/api/annotation":
            self._api_get_annotation(params)
        elif path == "/api/qc":
            self._api_get_qc(params)
        else:
            self._send_bytes(404, b"Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/annotation":
            self._api_post_annotation(params)
        else:
            self._send_bytes(404, b"Not Found")

    # --- API handlers ---

    def _api_get_samples(self) -> None:
        result: list[dict] = []
        for sample_dir in sorted(_dataset_dir.iterdir()):
            if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
                continue
            for vertebra_dir in sorted(sample_dir.iterdir()):
                if not vertebra_dir.is_dir():
                    continue
                images_dir = vertebra_dir / "images"
                if not images_dir.exists():
                    continue
                slices = sorted(images_dir.glob("slice_*.png"))
                if not slices:
                    continue

                lines_path = vertebra_dir / "lines.json"
                n_annotated = 0
                if lines_path.exists():
                    with lines_path.open() as f:
                        data = json.load(f)
                    # 4本すべてに2点以上あるスライスのみカウント
                    for sd in data.values():
                        if all(len(sd.get(f"line_{i}", [])) >= 2 for i in range(1, 5)):
                            n_annotated += 1

                result.append({
                    "sample": sample_dir.name,
                    "vertebra": vertebra_dir.name,
                    "slices": [s.stem.replace("slice_", "") for s in slices],
                    "n_slices": len(slices),
                    "n_annotated": n_annotated,
                })
        self._send_json(result)

    def _api_get_image(self, params: dict) -> None:
        sample = self._param(params, "sample")
        vertebra = self._param(params, "vertebra")
        slice_id = self._param(params, "slice")
        img_type = self._param(params, "type", "ct")

        if not all([sample, vertebra, slice_id]):
            self._send_bytes(400, b"Missing params")
            return

        subdir = "masks" if img_type == "mask" else "images"
        img_path = _dataset_dir / sample / vertebra / subdir / f"slice_{slice_id}.png"

        if not img_path.exists():
            self._send_bytes(404, b"Image not found")
            return

        self._serve_file(img_path)

    def _api_get_annotation(self, params: dict) -> None:
        sample = self._param(params, "sample")
        vertebra = self._param(params, "vertebra")

        if not all([sample, vertebra]):
            self._send_json({})
            return

        lines_path = _dataset_dir / sample / vertebra / "lines.json"
        if not lines_path.exists():
            self._send_json({})
            return

        with lines_path.open() as f:
            self._send_json(json.load(f))

    def _api_get_qc(self, params: dict) -> None:
        # threshold: 端点間直線からの最大垂直距離（px, 224px空間）
        threshold = float((params.get("threshold") or ["10"])[0])
        result: list[dict] = []

        for sample_dir in sorted(_dataset_dir.iterdir()):
            if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
                continue
            for vertebra_dir in sorted(sample_dir.iterdir()):
                lines_path = vertebra_dir / "lines.json"
                if not lines_path.exists():
                    continue
                with lines_path.open() as f:
                    data = json.load(f)
                for slice_key, lines in data.items():
                    for line_key in ("line_1", "line_2", "line_3", "line_4"):
                        pts = lines.get(line_key, [])
                        dev = _max_perp_distance(pts)
                        if dev > threshold:
                            result.append({
                                "sample": sample_dir.name,
                                "vertebra": vertebra_dir.name,
                                "slice": slice_key,
                                "line": line_key,
                                "n_pts": len(pts),
                                "deviation": round(dev, 1),
                            })

        result.sort(key=lambda r: -r["deviation"])
        self._send_json(result)

    def _api_post_annotation(self, params: dict) -> None:
        sample = self._param(params, "sample")
        vertebra = self._param(params, "vertebra")

        if not all([sample, vertebra]):
            self._send_json({"error": "missing params"}, 400)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        vert_dir = _dataset_dir / sample / vertebra
        lines_path = vert_dir / "lines.json"
        with lines_path.open("w") as f:
            json.dump(data, f, indent=2)

        _generate_overlays(vert_dir, data)
        self._send_json({"ok": True})

    # --- Helpers ---

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
    def _param(params: dict, key: str, default: str | None = None) -> str | None:
        vals = params.get(key)
        return vals[0] if vals else default

    def log_message(self, format: str, *args: object) -> None:
        pass  # アクセスログを抑制


def _max_perp_distance(pts: list) -> float:
    """端点を結ぶ直線から中間点の最大垂直距離を返す（px, 224px空間）。"""
    if len(pts) < 3:
        return 0.0
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        # 端点が同一座標 → 全点の端点からの距離
        return max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in pts[1:])
    return max(abs(dx * (y0 - y) - dy * (x0 - x)) / length for x, y in pts[1:-1])


def _generate_overlays(vert_dir: Path, annotation: dict) -> None:
    """アノテーション済みスライスのオーバーレイ PNG を overlays/ に生成する。"""
    overlays_dir = vert_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    for slice_key, lines in annotation.items():
        # ゼロパディングしてファイル名を復元
        slice_id = int(slice_key)
        stem = f"slice_{slice_id:03d}"

        ct_path = vert_dir / "images" / f"{stem}.png"
        mask_path = vert_dir / "masks" / f"{stem}.png"
        if not ct_path.exists():
            continue

        # CT をベースにRGB化
        ct = Image.open(ct_path).convert("RGB")
        size = ct.size  # (224, 224)

        # マスクを青半透明でオーバーレイ（convert_to_png.py と同色）
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_rgba = Image.new("RGBA", size, (0, 0, 0, 0))
            mask_rgba.paste((0, 0, 255, 77), mask=mask)  # alpha=77 ≒ 0.3
            ct = ct.convert("RGBA")
            ct = Image.alpha_composite(ct, mask_rgba).convert("RGB")

        draw = ImageDraw.Draw(ct)

        # 各ラインを描画
        for line_key, color in _LINE_COLORS.items():
            pts = lines.get(line_key, [])
            if len(pts) < 2:
                continue
            coords = [(round(x), round(y)) for x, y in pts]
            draw.line(coords, fill=color, width=2)
            # 各点に小円
            for x, y in coords:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(255, 255, 255))

        ct.save(overlays_dir / f"{stem}.png")


def main() -> None:
    global _dataset_dir

    parser = argparse.ArgumentParser(description="ラインアノテーションツールサーバー")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset", default=None, help="データセットディレクトリのパス")
    args = parser.parse_args()

    if args.dataset:
        _dataset_dir = Path(args.dataset).resolve()

    url = f"http://localhost:{args.port}"
    print(f"起動: {url}")
    print(f"データセット: {_dataset_dir}")
    print("終了: Ctrl+C")

    server = ThreadingHTTPServer(("localhost", args.port), AnnotationHandler)
    server.daemon_threads = True  # 詰まった接続スレッドが終了処理を妨げないようにする
    Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバー停止。")


if __name__ == "__main__":
    main()
