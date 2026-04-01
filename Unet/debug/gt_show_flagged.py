"""
GT品質フラグ付きスライスの可視化スクリプト

連続性fail・geometry failのスライスをCT画像 + GTポリラインオーバーレイで表示する。

使い方:
    uv run python Unet/debug/gt_show_flagged.py
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
OUTPUT_DIR = Path("Unet/debug/gt_validation_results/flagged_vis")
IMAGE_SIZE = 224

# 可視化対象：連続性failの上位ケース
CONTINUITY_TARGETS = [
    ("sample27", "C1", [30, 31]),
    ("sample29", "C4", [39, 40, 42, 43]),
    ("sample15.2", "C2", [40, 41, 44, 45, 46, 47]),
    ("sample31", "C1", [30, 31]),
    ("sample3",  "C5", [35, 36, 37, 38]),
    ("sample3",  "C4", [58, 59]),
    ("sample29", "C5", [38, 39]),
    ("sample32", "C2", [54, 55, 56, 57]),
]

# geometry fail 上位ケース（annotations.csv の上位から）
GEOMETRY_TARGETS = [
    ("sample31", "C2", [36, 37]),
    ("sample32", "C7", [44, 45]),
    ("sample17", "C5", [32]),
]

# ラインごとの色 (BGR)
LINE_COLORS = {
    "line_1": (0, 255, 0),    # 緑
    "line_2": (0, 128, 255),  # オレンジ
    "line_3": (255, 0, 255),  # マゼンタ
    "line_4": (0, 255, 255),  # シアン
}


def load_slice(sample: str, vertebra: str, slice_idx: int):
    """CT画像・マスク・GTラインを読み込む"""
    v_dir = DATASET_ROOT / sample / vertebra
    img_path = v_dir / "images" / f"slice_{slice_idx:03d}.png"
    mask_path = v_dir / "masks" / f"slice_{slice_idx:03d}.png"
    lj = v_dir / "lines.json"

    ct = None
    if img_path.exists():
        ct = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    lines = None
    if lj.exists():
        data = json.loads(lj.read_text())
        lines = data.get(str(slice_idx))

    return ct, lines


def draw_polyline_on_image(bgr: np.ndarray, pts_xy: list, color: tuple, label: str = ""):
    """ポリラインと各点を描画する"""
    if not pts_xy or len(pts_xy) < 2:
        return
    pts = np.array(pts_xy, dtype=np.float32)
    pts_i = pts.astype(np.int32)

    # ライン描画
    for i in range(len(pts_i) - 1):
        p1 = tuple(pts_i[i])
        p2 = tuple(pts_i[i + 1])
        cv2.line(bgr, p1, p2, color, 2)

    # 各点を描画（近接重複点を確認しやすくする）
    for i, p in enumerate(pts_i):
        cv2.circle(bgr, tuple(p), 3, color, -1)
        cv2.circle(bgr, tuple(p), 3, (255, 255, 255), 1)  # 白枠

    # 最初の点にラベル
    if label and len(pts_i) > 0:
        cv2.putText(bgr, label, tuple(pts_i[0] + np.array([3, -3])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def make_slice_panel(sample: str, vertebra: str, slice_idx: int, title_extra: str = "") -> np.ndarray | None:
    """1スライスのオーバーレイパネルを生成する"""
    ct, lines = load_slice(sample, vertebra, slice_idx)

    if ct is None:
        # 画像なし → グレーパネル
        panel = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 80, dtype=np.uint8)
        cv2.putText(panel, "NO IMAGE", (40, 112), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
    else:
        ct_u8 = cv2.resize(ct, (IMAGE_SIZE, IMAGE_SIZE))
        panel = cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR)

    # GTライン描画
    if lines:
        for line_id, color in LINE_COLORS.items():
            pts = lines.get(line_id)
            if pts:
                # 近接重複点の検出
                pts_arr = np.array(pts)
                segs = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
                has_near_dup = len(segs) > 0 and segs.min() < 1.0
                # 近接重複点がある場合は色を明るくする
                draw_color = color if not has_near_dup else tuple(min(255, c + 80) for c in color)
                draw_polyline_on_image(panel, pts, draw_color, line_id[-1])

    # タイトル
    title = f"{sample}/{vertebra}/s{slice_idx:03d}"
    if title_extra:
        title += f" {title_extra}"
    cv2.putText(panel, title, (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (255, 255, 200), 1, cv2.LINE_AA)

    return panel


def compute_phi_rho_for_slice(lines: dict | None) -> dict:
    """スライスの全ラインのφ,ρを計算して返す"""
    if lines is None:
        return {}
    result = {}
    for line_id, pts in lines.items():
        if not pts or len(pts) < 2:
            result[line_id] = (float("nan"), float("nan"))
            continue
        center = IMAGE_SIZE / 2.0
        pts_arr = np.array(pts, dtype=np.float64)
        pm = np.column_stack([pts_arr[:, 0] - center, -(pts_arr[:, 1] - center)])
        cen = pm.mean(axis=0)
        xc = pm - cen
        cov = (xc.T @ xc) / max(1, len(pts_arr))
        if cov.max() < 1e-10:
            result[line_id] = (float("nan"), float("nan"))
            continue
        evals, evecs = np.linalg.eigh(cov)
        d = evecs[:, np.argmax(evals)]
        nx, ny = -d[1], d[0]
        if ny < 0 or (ny == 0 and nx < 0):
            nx, ny = -nx, -ny
        phi = math.atan2(ny, nx)
        rho = nx * cen[0] + ny * cen[1]
        D = math.sqrt(IMAGE_SIZE**2 + IMAGE_SIZE**2)
        result[line_id] = (math.degrees(phi), rho / D * IMAGE_SIZE)
    return result


def make_continuity_panel(sample: str, vertebra: str, slice_idxs: list) -> np.ndarray:
    """連続スライスを横に並べ、φ・ρ変化量を表示するパネルを生成する"""
    panels = []
    prev_params = None

    for idx in slice_idxs:
        _, lines = load_slice(sample, vertebra, idx)
        curr_params = compute_phi_rho_for_slice(lines)

        # φ変化量を計算してタイトルに付加
        extra = ""
        if prev_params is not None:
            deltas = []
            for lid in ["line_1", "line_2", "line_3", "line_4"]:
                pp = prev_params.get(lid, (float("nan"), float("nan")))
                cp = curr_params.get(lid, (float("nan"), float("nan")))
                if not any(math.isnan(v) for v in [pp[0], cp[0]]):
                    dphi = abs(cp[0] - pp[0])
                    dphi = min(dphi, 180 - dphi)
                    deltas.append(f"L{lid[-1]}:Δφ{dphi:.0f}")
            if deltas:
                extra = " ".join(deltas[:2])  # 2つまで表示

        panel = make_slice_panel(sample, vertebra, idx, extra)
        panels.append(panel)
        prev_params = curr_params

    if not panels:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # 凡例パネル
    legend = np.zeros((IMAGE_SIZE, 90, 3), dtype=np.uint8)
    cv2.putText(legend, "LEGEND", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for i, (lid, color) in enumerate(LINE_COLORS.items()):
        y = 35 + i * 18
        cv2.line(legend, (5, y), (30, y), color, 2)
        cv2.putText(legend, lid, (35, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    # 近接重複点の注釈
    cv2.putText(legend, "* bright =", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
    cv2.putText(legend, "near-dup", (5, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    panels.append(legend)
    return np.hstack(panels)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 連続性fail ケース
    print("[INFO] 連続性failケースを可視化中...")
    for sample, vertebra, slice_idxs in CONTINUITY_TARGETS:
        row = make_continuity_panel(sample, vertebra, slice_idxs)
        fname = f"continuity_{sample}_{vertebra}.png"
        cv2.imwrite(str(OUTPUT_DIR / fname), row)
        print(f"  → {fname}")

    # 2. geometry fail ケース
    print("[INFO] geometry failケースを可視化中...")
    for sample, vertebra, slice_idxs in GEOMETRY_TARGETS:
        row = make_continuity_panel(sample, vertebra, slice_idxs)
        fname = f"geometry_{sample}_{vertebra}.png"
        cv2.imwrite(str(OUTPUT_DIR / fname), row)
        print(f"  → {fname}")

    # 3. 全連続性failを1枚にまとめたサマリー
    print("[INFO] サマリー画像を生成中...")
    all_rows = []
    for sample, vertebra, slice_idxs in CONTINUITY_TARGETS:
        row = make_continuity_panel(sample, vertebra, slice_idxs)
        # ラベルバー追加
        label_bar = np.zeros((20, row.shape[1], 3), dtype=np.uint8)
        label = f"[CONTINUITY FAIL] {sample}/{vertebra}  slices: {slice_idxs}"
        cv2.putText(label_bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 220, 100), 1, cv2.LINE_AA)
        all_rows.append(np.vstack([label_bar, row]))

    for sample, vertebra, slice_idxs in GEOMETRY_TARGETS:
        row = make_continuity_panel(sample, vertebra, slice_idxs)
        label_bar = np.zeros((20, row.shape[1], 3), dtype=np.uint8)
        label = f"[GEOMETRY FAIL] {sample}/{vertebra}  slices: {slice_idxs}"
        cv2.putText(label_bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (100, 220, 255), 1, cv2.LINE_AA)
        all_rows.append(np.vstack([label_bar, row]))

    # 幅を揃えて縦に結合
    max_w = max(r.shape[1] for r in all_rows)
    padded = []
    for r in all_rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    summary = np.vstack(padded)
    cv2.imwrite(str(OUTPUT_DIR / "summary_flagged.png"), summary)
    print(f"  → summary_flagged.png  ({summary.shape[1]}x{summary.shape[0]}px)")
    print(f"\n[INFO] 出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
