"""境界面を z 軸を含む断面（冠状断・矢状断）にリフォーマットして可視化する。

面 x·cosφ(z) + y·sinφ(z) = ρ(z) を y=const（冠状断）で切ると z ごとに1点の曲線になる。
アンカー帯と外挿域が1枚の画像で比較できるため、GTなしの目視評価手段になる。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/mnt/nfs1/home/yamamoto-hiroto/research/VAI")
from Unet.multitask.utils.losses import extract_gt_line_params  # noqa: E402

LINE_KEYS = ("line_1", "line_2", "line_3", "line_4")
COLORS = {
    "line_1": (255, 60, 60),
    "line_2": (60, 170, 255),
    "line_3": (60, 255, 60),
    "line_4": (255, 255, 60),
}
S = 224
C = S / 2.0
D_NORM = np.sqrt(2) * S


def load_volume(vert_dir: Path) -> tuple[np.ndarray, list[int]]:
    """images/ を z 昇順に積んで (Z, H, W) のボリュームにする。"""
    paths = sorted((vert_dir / "images").glob("slice_*.png"))
    zs = [int(p.stem[6:]) for p in paths]
    vol = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths])
    return vol, zs


def line_params_per_z(vert_dir: Path, zs: list[int]) -> dict:
    """各 z の 4 本の (phi, rho_px) を返す。"""
    data = json.loads((vert_dir / "lines.json").read_text())
    out = {}
    for z in zs:
        e = data.get(str(z))
        if e is None or not all(e.get(k) and len(e[k]) >= 2 for k in LINE_KEYS):
            continue
        p = [extract_gt_line_params(e[k], S) for k in LINE_KEYS]
        if any(np.isnan(v[0]) for v in p):
            continue
        out[z] = [(v[0], v[1] * D_NORM) for v in p]
    return out


def coronal_curve(phi: float, rho: float, y_img: float) -> float | None:
    """冠状断（画像行 y_img 固定）と直線の交点の x（画像座標）。"""
    y_m = C - y_img
    if abs(np.cos(phi)) < 0.2:
        return None
    return (rho - y_m * np.sin(phi)) / np.cos(phi) + C


def sagittal_curve(phi: float, rho: float, x_img: float) -> float | None:
    """矢状断（画像列 x_img 固定）と直線の交点の y（画像座標）。"""
    x_m = x_img - C
    if abs(np.sin(phi)) < 0.2:
        return None
    return C - (rho - x_m * np.cos(phi)) / np.sin(phi)


def render(
    plane: np.ndarray,
    zs: list[int],
    curves: dict,
    anchor: tuple[int, int],
    title: str,
    scale: int = 3,
) -> Image.Image:
    """リフォーマット断面に曲線を重ねる。縦=z, 横=面内座標。"""
    h, w = plane.shape
    im = Image.fromarray(plane).convert("RGB")
    im = im.resize((w * scale, h * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(im)

    # アンカー帯を左端に帯で表示
    for i, z in enumerate(zs):
        if anchor[0] <= z <= anchor[1]:
            dr.rectangle([0, i * scale, 5, (i + 1) * scale], fill=(255, 255, 255))

    for key, color in COLORS.items():
        pts = curves.get(key, [])
        seg = []
        for i, val in pts:
            if val is None or not (-w * 0.5 < val < w * 1.5):
                if len(seg) > 1:
                    dr.line(seg, fill=color, width=2)
                seg = []
                continue
            seg.append((val * scale, (i + 0.5) * scale))
        if len(seg) > 1:
            dr.line(seg, fill=color, width=2)

    dr.text((8, 4), title, fill=(255, 255, 0))
    return im


def main() -> None:
    vert_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dataset_zprop/sample1/C4")
    report = json.loads(Path("data/dataset_zprop/batch_report.json").read_text())
    sample, vert = vert_dir.parts[-2], vert_dir.parts[-1]
    anchor = next(
        (r["anchor_range"] for r in report["reports"]
         if r["sample"] == sample and r["vertebra"] == vert),
        (0, 0),
    )

    vol, zs = load_volume(vert_dir)
    params = line_params_per_z(vert_dir, zs)

    # 帯内の junction 平均位置を基準面に使う
    band = [params[z] for z in zs if anchor[0] <= z <= anchor[1] and z in params]
    if not band:
        raise SystemExit("アンカー帯に有効な線がない")

    def intersect(p1, p2):
        A = np.array([[np.cos(p1[0]), np.sin(p1[0])], [np.cos(p2[0]), np.sin(p2[0])]])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        return np.linalg.solve(A, np.array([p1[1], p2[1]]))

    js = [intersect(p[0], p[1]) for p in band] + [intersect(p[2], p[3]) for p in band]
    js = np.array([j for j in js if j is not None])
    j_img = np.array([js[:, 0].mean() + C, C - js[:, 1].mean()])
    y0, x0 = float(np.clip(j_img[1], 20, S - 20)), float(np.clip(j_img[0], 20, S - 20))

    # 冠状断: 行 y0 を全 z で抜く → (Z, W)
    cor = vol[:, int(round(y0)), :]
    sag = vol[:, :, int(round(x0))]

    cor_curves = {k: [] for k in LINE_KEYS}
    sag_curves = {k: [] for k in LINE_KEYS}
    for i, z in enumerate(zs):
        p = params.get(z)
        if p is None:
            continue
        for li, key in enumerate(LINE_KEYS):
            cor_curves[key].append((i, coronal_curve(p[li][0], p[li][1], y0)))
            sag_curves[key].append((i, sagittal_curve(p[li][0], p[li][1], x0)))

    a = render(cor, zs, cor_curves, anchor, f"{sample}/{vert} 冠状断 y={y0:.0f}")
    b = render(sag, zs, sag_curves, anchor, f"{sample}/{vert} 矢状断 x={x0:.0f}")
    out = Image.new("RGB", (a.width + b.width + 12, max(a.height, b.height)), (0, 0, 0))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 12, 0))
    dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/reformat.png"
    out.save(dst)
    print(f"saved {dst}  ({out.size})  anchor z={anchor}, z範囲={zs[0]}-{zs[-1]}")


if __name__ == "__main__":
    main()
