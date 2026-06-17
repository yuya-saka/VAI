"""
予測線から生成した領域分割を GT マスクと並べて可視化する。

1椎骨について、anchor スライスと外挿スライス（距離別）を選び、
CT 画像に GT / 予測の領域オーバーレイを重ねて比較グリッドを出力する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib_fontja  # noqa: F401  日本語フォント自動設定
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing.learning_dataset.propagate_lines_z import (
    build_smooth_trajectory,
    compute_mask_geometry,
    evaluate_trajectory,
    extract_slice_state,
    reconstruct_lines_from_state,
)
from data_preprocessing.segmentation_dataset.generate_region_mask import (
    generate_region_mask,
)

# eval_region_seg と同じ補助関数を再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_region_seg import (  # noqa: E402
    LINE_KEYS,
    OUTLIER_THRESH_DEG,
    TARGET_SIZE,
    MIN_MASK_AREA,
    compute_dice_scores,
    get_max_angle_error,
    get_zprop_dir,
    is_valid_prediction,
    load_all_predictions,
    load_binary_mask,
    load_gt_mask,
    pred_lines_to_polylines,
    get_valid_z_range,
)

# RGB 色定義（preprocess_all.py の BGR を RGB に変換）
REGION_COLORS_RGB = np.array([
    [0, 0, 0],       # 0: 背景
    [0, 200, 0],     # 1: 椎体 body (緑)
    [200, 0, 0],     # 2: 右椎間孔 right foramen (赤)
    [0, 0, 200],     # 3: 左椎間孔 left foramen (青)
    [200, 200, 0],   # 4: 後方要素 posterior (黄)
], dtype=np.float32)
OVERLAY_ALPHA = 0.45


def load_ct_image(sample: str, vertebra: str, slice_idx: int) -> np.ndarray | None:
    """CT グレースケール画像を読み込む。"""
    path = get_zprop_dir(sample, vertebra) / "images" / f"slice_{slice_idx:03d}.png"
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("L"))


def make_overlay(ct_gray: np.ndarray, label_img: np.ndarray) -> np.ndarray:
    """CT 画像に領域ラベルのカラーオーバーレイを重ねた RGB 画像を返す。"""
    ct_rgb = np.stack([ct_gray] * 3, axis=-1).astype(np.float32)
    color_layer = REGION_COLORS_RGB[label_img]
    fg = (label_img > 0).astype(np.float32)[..., None]
    blended = ct_rgb * (1.0 - fg * OVERLAY_ALPHA) + color_layer * OVERLAY_ALPHA
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_propagation(sample: str, vertebra: str, slice_preds: dict,
                      outlier_thresh: float = OUTLIER_THRESH_DEG):
    """1椎骨の z 伝播を構築し、(spline, extrap_params, geometries, masks, anchor_z, areas) を返す。"""
    z_range = get_valid_z_range(sample, vertebra)
    if z_range is None:
        return None
    z_lo, z_hi = z_range
    all_z = list(range(z_lo, z_hi + 1))

    bin_masks, geometries, mask_areas = {}, {}, {}
    for z in all_z:
        bm = load_binary_mask(sample, vertebra, z)
        if bm is not None and np.sum(bm > 0) >= MIN_MASK_AREA:
            bin_masks[z] = bm
            geometries[z] = compute_mask_geometry(bm)
            mask_areas[z] = float(np.sum(bm > 0))

    # 非外れ値スライスをアンカーに（eval_region_seg と同ロジック）
    valid_preds = {}
    for z, pred in slice_preds.items():
        if not is_valid_prediction(pred):
            continue
        if get_max_angle_error(pred) < outlier_thresh:
            valid_preds[z] = pred

    anchor_slices = {z: p for z, p in valid_preds.items() if z in geometries}
    if len(anchor_slices) < 2:
        anchor_slices = {
            z: p for z, p in slice_preds.items()
            if is_valid_prediction(p) and z in geometries
        }
    if len(anchor_slices) < 2:
        return None

    anchor_states = {}
    for z, pred in anchor_slices.items():
        try:
            anchor_states[z] = extract_slice_state(pred_lines_to_polylines(pred), geometries[z])
        except Exception:
            pass
    if len(anchor_states) < 2:
        return None

    spline, extrap_params = build_smooth_trajectory(anchor_states)
    area_lo = mask_areas.get(int(extrap_params["z_lo"]), 1.0)
    area_hi = mask_areas.get(int(extrap_params["z_hi"]), 1.0)

    return {
        "spline": spline, "extrap_params": extrap_params,
        "geometries": geometries, "bin_masks": bin_masks, "mask_areas": mask_areas,
        "anchor_slices": anchor_slices, "area_lo": area_lo, "area_hi": area_hi,
        "all_z": all_z,
    }


def predict_label(prop: dict, sample: str, vertebra: str, z: int):
    """指定 z スライスの予測ラベル画像と provenance を返す。"""
    if z not in prop["bin_masks"]:
        return None, None
    if z in prop["anchor_slices"]:
        polylines = pred_lines_to_polylines(prop["anchor_slices"][z])
        prov = "anchor"
    else:
        state_vec, prov, _ = evaluate_trajectory(
            prop["spline"], prop["extrap_params"], z,
            prop["mask_areas"].get(z, 1.0), prop["area_lo"], prop["area_hi"],
        )
        polylines = reconstruct_lines_from_state(state_vec, prop["geometries"][z], TARGET_SIZE)
    try:
        seg, _ = generate_region_mask(
            line_1=polylines["line_1"], line_2=polylines["line_2"],
            line_3=polylines["line_3"], line_4=polylines["line_4"],
            vertebra_mask=prop["bin_masks"][z],
        )
        return np.argmax(seg, axis=0).astype(np.uint8), prov
    except Exception:
        return None, prov


def select_slices(prop: dict) -> list[tuple[int, str, int]]:
    """anchor + 外挿スライスを距離別に選ぶ。(z, label, dist) のリストを返す。"""
    anchor_z = sorted(prop["anchor_slices"])
    a_mid = anchor_z[len(anchor_z) // 2]
    selected = [(a_mid, "anchor", 0)]

    nearest = lambda z: min(abs(z - a) for a in anchor_z)
    candidates = sorted(z for z in prop["all_z"] if z in prop["bin_masks"] and z not in prop["anchor_slices"])
    # 距離 ~3, ~7, ~12, max を選ぶ
    targets = [3, 7, 12, 9999]
    used = set()
    for t in targets:
        best, best_diff = None, float("inf")
        for z in candidates:
            if z in used:
                continue
            dist = nearest(z)
            diff = abs(dist - t) if t != 9999 else -dist
            if diff < best_diff:
                best_diff, best = diff, z
        if best is not None:
            used.add(best)
            selected.append((best, "extrap", nearest(best)))
    return selected


def visualize_vertebra(sample: str, vertebra: str, slice_preds: dict, out_path: Path) -> bool:
    """1椎骨の比較グリッドを生成・保存する。"""
    prop = build_propagation(sample, vertebra, slice_preds)
    if prop is None:
        print(f"  {sample}/{vertebra}: 伝播構築失敗")
        return False

    slices = select_slices(prop)
    n = len(slices)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (z, label, dist) in enumerate(slices):
        ct = load_ct_image(sample, vertebra, z)
        gt = load_gt_mask(sample, vertebra, z)
        pred_lbl, prov = predict_label(prop, sample, vertebra, z)
        if ct is None or gt is None or pred_lbl is None:
            for row in range(2):
                axes[row, col].axis("off")
            continue

        dice = compute_dice_scores(pred_lbl, gt)
        title = f"slice {z}\n" + ("アンカー" if label == "anchor" else f"外挿 (距離{dist})")

        axes[0, col].imshow(make_overlay(ct, gt))
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(make_overlay(ct, pred_lbl))
        c2c3 = (dice["class_2"] + dice["class_3"]) / 2
        axes[1, col].set_title(
            f"Dice {dice['mean']:.3f}\n孔c2/c3 {c2c3:.3f}", fontsize=10
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("真値 (GT)", fontsize=12)
    axes[1, 0].set_ylabel("予測 (伝播)", fontsize=12)
    # ylabel は axis off で消えるのでテキストで左に追加
    fig.text(0.01, 0.72, "真値 GT", fontsize=13, rotation=90, va="center")
    fig.text(0.01, 0.28, "予測 伝播", fontsize=13, rotation=90, va="center")

    legend = "緑=椎体  赤=右椎間孔  青=左椎間孔  黄=後方要素"
    fig.suptitle(f"{sample}/{vertebra}   領域分割: 真値 vs 予測伝播\n{legend}", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  {sample}/{vertebra}: 保存 -> {out_path.name}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="予測線伝播の領域分割を可視化")
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["sample17/C1", "sample21/C2", "sample35/C3"],
                        help="可視化対象 sample/vertebra のリスト")
    parser.add_argument("--fold_search", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    out_dir = ROOT_DIR / "Unet" / "outputs" / "line_20260616" / "sig4.0_ALL(CC適用)" / "region_eval" / "viz"

    # 全 fold の予測をロードしてマージ
    all_preds = {}
    for fold in args.fold_search:
        all_preds.update(load_all_predictions(fold))

    for target in args.targets:
        sample, vertebra = target.split("/")
        key = (sample, vertebra)
        if key not in all_preds:
            print(f"  {target}: 予測データなし")
            continue
        visualize_vertebra(sample, vertebra, all_preds[key], out_dir / f"{sample}_{vertebra}.png")

    print(f"\n出力先: {out_dir}/")


if __name__ == "__main__":
    main()
