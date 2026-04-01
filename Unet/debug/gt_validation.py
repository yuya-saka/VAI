"""
GTアノテーション品質検証スクリプト

Codex推奨の品質指標に基づき、全データセットのlines.jsonを検査する。
結果はCSVと要約レポートとして出力する。

使い方:
    uv run python Unet/debug/gt_validation.py
    uv run python Unet/debug/gt_validation.py --output Unet/debug/gt_validation_results.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# -------------------------
# 定数
# -------------------------
DATASET_ROOT = Path("/mnt/nfs1/home/yamamoto-hiroto/research/VAI/dataset")
IMAGE_SIZE = 224
VERTEBRA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# 閾値（Codex推奨）
THRESH = {
    "li_warn": 0.95,
    "li_fail": 0.90,
    "rmse_perp_warn": 2.5,
    "rmse_perp_fail": 4.0,
    "kappa_warn": 10.0,
    "kappa_fail": 20.0,
    "delta_phi_warn": 8.0,
    "delta_phi_fail": 12.0,
    "delta_rho_warn": 6.0,
    "delta_rho_fail": 10.0,
    "near_dup_hard": 1.0,
    "near_dup_warn": 2.0,
}


# -------------------------
# PCA による直線パラメータ抽出（losses.py と同等）
# -------------------------
def extract_line_params_pca(pts_xy: list, image_size: int = IMAGE_SIZE):
    """
    polylineから (phi, rho, LI, RMSE_perp, lambda1, lambda2) を返す。
    無効時は全てNaN。
    """
    if pts_xy is None or len(pts_xy) < 2:
        return tuple([float("nan")] * 6)

    center = image_size / 2.0
    pts = np.array(pts_xy, dtype=np.float64)

    # 画像座標 → 数学座標（Y上向き・中心原点）
    pm = np.column_stack([pts[:, 0] - center, -(pts[:, 1] - center)])
    cen = pm.mean(axis=0)
    xc = pm - cen
    cov = (xc.T @ xc) / max(1, len(pts))

    if cov.max() < 1e-10:
        return tuple([float("nan")] * 6)

    evals, evecs = np.linalg.eigh(cov)
    lam2, lam1 = sorted(evals)  # lam1 >= lam2
    d = evecs[:, np.argmax(evals)]

    nx, ny = -d[1], d[0]
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny

    phi = math.atan2(ny, nx)
    rho = nx * cen[0] + ny * cen[1]

    # 線形性指標
    li = lam1 / (lam1 + lam2 + 1e-10)

    # 直交残差RMSE（各点からフィットした直線への距離）
    # 直線: nx*x + ny*y = rho_from_origin (centroid通過の法線形式)
    rho_abs = nx * cen[0] + ny * cen[1]
    dists = np.abs(pm @ np.array([nx, ny]) - rho_abs)
    rmse_perp = float(np.sqrt(np.mean(dists**2)))

    D = math.sqrt(image_size**2 + image_size**2)
    return float(phi), float(rho / D), float(li), rmse_perp, float(lam1), float(lam2)


# -------------------------
# 整合性チェック
# -------------------------
def check_integrity(pts_xy: list, image_size: int = IMAGE_SIZE) -> dict:
    """
    ハードフェイル・警告レベルの整合性チェック。
    返り値: {flag: str, min_seg_len: float, out_of_bounds: bool, near_dup: bool}
    """
    result = {
        "flag": "ok",
        "min_seg_len": float("inf"),
        "out_of_bounds": False,
        "near_dup_hard": False,
        "near_dup_warn": False,
    }

    if pts_xy is None or len(pts_xy) < 2:
        result["flag"] = "fail_point_count"
        return result

    pts = np.array(pts_xy, dtype=np.float64)

    # 座標範囲チェック
    if np.any(pts < 0) or np.any(pts > image_size - 1):
        result["out_of_bounds"] = True
        result["flag"] = "fail_out_of_bounds"

    # セグメント長チェック（近接点）
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if len(segs) > 0:
        min_seg = float(segs.min())
        result["min_seg_len"] = min_seg
        if min_seg < THRESH["near_dup_hard"]:
            result["near_dup_hard"] = True
            if result["flag"] == "ok":
                result["flag"] = "fail_near_dup"
        elif min_seg < THRESH["near_dup_warn"]:
            result["near_dup_warn"] = True
            if result["flag"] == "ok":
                result["flag"] = "warn_near_dup"

    return result


# -------------------------
# 折れ角の平均（kappa_bar）
# -------------------------
def compute_kappa_bar(pts_xy: list) -> float:
    """
    ポリライン内の絶対折れ角の平均（degrees）。
    点が3点未満の場合は0.0を返す（折れなし）。
    """
    if pts_xy is None or len(pts_xy) < 3:
        return 0.0

    pts = np.array(pts_xy, dtype=np.float64)
    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        # 折れ角 = π - angle（直線なら0、直角なら90度）
        angles.append(math.degrees(math.pi - math.acos(cos_a)))

    return float(np.mean(angles)) if angles else 0.0


# -------------------------
# スライス間連続性チェック（Δφ, Δρ）
# -------------------------
def compute_inter_slice_continuity(slice_params: dict) -> list:
    """
    slice_params: {slice_idx: {line_id: (phi, rho, ...)}}
    隣接スライス間のΔφ, Δρを返す。
    返り値: [{slice_from, slice_to, line_id, delta_phi_deg, delta_rho_px}]
    """
    results = []
    sorted_slices = sorted(slice_params.keys())

    for i in range(1, len(sorted_slices)):
        s_prev = sorted_slices[i - 1]
        s_curr = sorted_slices[i]

        # 連続スライスのみ（飛びは除外）
        if s_curr - s_prev > 2:
            continue

        prev = slice_params[s_prev]
        curr = slice_params[s_curr]

        for line_id in ["line_1", "line_2", "line_3", "line_4"]:
            if line_id not in prev or line_id not in curr:
                continue
            phi_p, rho_p = prev[line_id][:2]
            phi_c, rho_c = curr[line_id][:2]

            if any(math.isnan(v) for v in [phi_p, rho_p, phi_c, rho_c]):
                continue

            # φ差分（π周期対称）
            dphi = abs(phi_c - phi_p)
            dphi = min(dphi, math.pi - dphi)
            dphi_deg = math.degrees(dphi)

            # ρ差分（ピクセル換算）
            D = math.sqrt(IMAGE_SIZE**2 + IMAGE_SIZE**2)
            drho_px = abs(rho_c - rho_p) * D

            results.append(
                {
                    "slice_from": s_prev,
                    "slice_to": s_curr,
                    "line_id": line_id,
                    "delta_phi_deg": dphi_deg,
                    "delta_rho_px": drho_px,
                }
            )

    return results


# -------------------------
# メイン検証ロジック
# -------------------------
def validate_dataset(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    """
    全サンプルを検証し、アノテーション単位のレコードと
    スライス間連続性レコードを返す。
    """
    ann_records = []   # 1アノテーション（1スライス×1ライン）=1行
    cont_records = []  # スライス間連続性（1ペア×1ライン）=1行

    samples = sorted(dataset_root.iterdir())

    for sample_dir in samples:
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name

        for v_name in VERTEBRA_NAMES:
            v_dir = sample_dir / v_name
            lj = v_dir / "lines.json"
            if not v_dir.exists() or not lj.exists():
                continue

            try:
                lines_data = json.loads(lj.read_text())
            except Exception as e:
                print(f"[WARN] JSON parse error: {lj} — {e}")
                continue

            # スライス間連続性用に各スライスのパラメータを収集
            slice_params: dict[int, dict] = {}

            for slice_idx_str, lines in lines_data.items():
                slice_idx = int(slice_idx_str)

                # 画像ファイル存在確認
                img_exists = (v_dir / "images" / f"slice_{slice_idx:03d}.png").exists()
                mask_exists = (v_dir / "masks" / f"slice_{slice_idx:03d}.png").exists()

                slice_params[slice_idx] = {}

                for line_id in ["line_1", "line_2", "line_3", "line_4"]:
                    pts = lines.get(line_id)

                    # 整合性チェック
                    integrity = check_integrity(pts)

                    # PCAパラメータ
                    phi, rho, li, rmse_perp, lam1, lam2 = extract_line_params_pca(pts or [])

                    # 折れ角
                    kappa = compute_kappa_bar(pts or [])

                    # フラグ判定
                    flag = integrity["flag"]
                    if flag == "ok":
                        if li < THRESH["li_fail"] or rmse_perp > THRESH["rmse_perp_fail"] or kappa > THRESH["kappa_fail"]:
                            flag = "fail_geometry"
                        elif li < THRESH["li_warn"] or rmse_perp > THRESH["rmse_perp_warn"] or kappa > THRESH["kappa_warn"]:
                            flag = "warn_geometry"

                    n_points = len(pts) if pts else 0

                    ann_records.append(
                        {
                            "sample": sample_name,
                            "vertebra": v_name,
                            "slice_idx": slice_idx,
                            "line_id": line_id,
                            "n_points": n_points,
                            "img_exists": img_exists,
                            "mask_exists": mask_exists,
                            "flag": flag,
                            "out_of_bounds": integrity["out_of_bounds"],
                            "near_dup_hard": integrity["near_dup_hard"],
                            "near_dup_warn": integrity["near_dup_warn"],
                            "min_seg_len": round(integrity["min_seg_len"], 4),
                            "phi": round(phi, 6) if not math.isnan(phi) else float("nan"),
                            "rho": round(rho, 6) if not math.isnan(rho) else float("nan"),
                            "li": round(li, 6) if not math.isnan(li) else float("nan"),
                            "rmse_perp": round(rmse_perp, 4) if not math.isnan(rmse_perp) else float("nan"),
                            "kappa_bar_deg": round(kappa, 4),
                            "lam1": round(lam1, 4) if not math.isnan(lam1) else float("nan"),
                            "lam2": round(lam2, 4) if not math.isnan(lam2) else float("nan"),
                        }
                    )

                    # スライス間連続性用に保存
                    if not math.isnan(phi):
                        slice_params[slice_idx][line_id] = (phi, rho)

            # スライス間連続性計算
            cont = compute_inter_slice_continuity(slice_params)
            for c in cont:
                # フラグ付け
                cont_flag = "ok"
                if c["delta_phi_deg"] > THRESH["delta_phi_fail"] or c["delta_rho_px"] > THRESH["delta_rho_fail"]:
                    cont_flag = "fail_continuity"
                elif c["delta_phi_deg"] > THRESH["delta_phi_warn"] or c["delta_rho_px"] > THRESH["delta_rho_warn"]:
                    cont_flag = "warn_continuity"

                cont_records.append(
                    {
                        "sample": sample_name,
                        "vertebra": v_name,
                        "slice_from": c["slice_from"],
                        "slice_to": c["slice_to"],
                        "line_id": c["line_id"],
                        "delta_phi_deg": round(c["delta_phi_deg"], 4),
                        "delta_rho_px": round(c["delta_rho_px"], 4),
                        "flag": cont_flag,
                    }
                )

    return ann_records, cont_records


# -------------------------
# 要約レポート出力
# -------------------------
def print_summary(ann_records: list[dict], cont_records: list[dict]) -> None:
    """検証結果の要約をコンソールに出力する"""
    total = len(ann_records)
    flags = defaultdict(int)
    for r in ann_records:
        flags[r["flag"]] += 1

    print("\n" + "=" * 60)
    print("GT アノテーション品質検証レポート")
    print("=" * 60)
    print(f"\n総アノテーション数: {total}")
    print("\n--- フラグ別集計 ---")
    for flag, count in sorted(flags.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {flag:<25} : {count:>6} ({pct:.1f}%)")

    # 近接重複点の統計
    near_dup_hard = sum(1 for r in ann_records if r["near_dup_hard"])
    near_dup_warn = sum(1 for r in ann_records if r["near_dup_warn"])
    out_of_bounds = sum(1 for r in ann_records if r["out_of_bounds"])
    print(f"\n--- 整合性問題 ---")
    print(f"  座標範囲外          : {out_of_bounds}")
    print(f"  近接重複点 (< 1px)  : {near_dup_hard}")
    print(f"  近接重複点 (1-2px)  : {near_dup_warn}")

    # LI統計
    li_vals = [r["li"] for r in ann_records if not math.isnan(r["li"])]
    if li_vals:
        li_arr = np.array(li_vals)
        print(f"\n--- 線形性指標 (LI) ---")
        print(f"  mean={li_arr.mean():.4f}  median={np.median(li_arr):.4f}  min={li_arr.min():.4f}")
        print(f"  LI < 0.90 (fail): {(li_arr < 0.90).sum()}")
        print(f"  LI < 0.95 (warn): {(li_arr < 0.95).sum()}")

    # RMSE_perp統計
    rmse_vals = [r["rmse_perp"] for r in ann_records if not math.isnan(r["rmse_perp"])]
    if rmse_vals:
        rmse_arr = np.array(rmse_vals)
        print(f"\n--- 直交残差 RMSE (px) ---")
        print(f"  mean={rmse_arr.mean():.4f}  median={np.median(rmse_arr):.4f}  max={rmse_arr.max():.4f}")
        print(f"  RMSE > 4.0 (fail): {(rmse_arr > 4.0).sum()}")
        print(f"  RMSE > 2.5 (warn): {(rmse_arr > 2.5).sum()}")

    # kappa統計
    kappa_vals = [r["kappa_bar_deg"] for r in ann_records]
    if kappa_vals:
        kappa_arr = np.array(kappa_vals)
        print(f"\n--- 折れ角平均 kappa (deg) ---")
        print(f"  mean={kappa_arr.mean():.4f}  median={np.median(kappa_arr):.4f}  max={kappa_arr.max():.4f}")
        print(f"  kappa > 20 (fail): {(kappa_arr > 20).sum()}")
        print(f"  kappa > 10 (warn): {(kappa_arr > 10).sum()}")

    # スライス間連続性
    total_cont = len(cont_records)
    cont_flags = defaultdict(int)
    for r in cont_records:
        cont_flags[r["flag"]] += 1
    print(f"\n--- スライス間連続性 ({total_cont} ペア) ---")
    for flag, count in sorted(cont_flags.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_cont if total_cont > 0 else 0
        print(f"  {flag:<25} : {count:>6} ({pct:.1f}%)")

    # 椎体別の問題件数
    print(f"\n--- 椎体別 fail/warn 件数 ---")
    by_vert: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in ann_records:
        by_vert[r["vertebra"]][r["flag"]] += 1
    for v in VERTEBRA_NAMES:
        if v in by_vert:
            d = by_vert[v]
            fail = sum(v2 for k, v2 in d.items() if "fail" in k)
            warn = sum(v2 for k, v2 in d.items() if "warn" in k)
            total_v = sum(d.values())
            print(f"  {v}: total={total_v}  fail={fail}  warn={warn}")

    print("\n" + "=" * 60)


# -------------------------
# CSV保存
# -------------------------
def save_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[INFO] 保存: {path} ({len(records)} 行)")


# -------------------------
# エントリポイント
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GT アノテーション品質検証")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DATASET_ROOT,
        help="データセットのルートディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("Unet/debug/gt_validation_results"),
        help="結果CSVの出力ディレクトリ",
    )
    args = parser.parse_args()

    print(f"[INFO] データセット: {args.dataset_root}")
    print("[INFO] 検証開始...")

    ann_records, cont_records = validate_dataset(args.dataset_root)

    print_summary(ann_records, cont_records)

    save_csv(ann_records, args.output_dir / "annotations.csv")
    save_csv(cont_records, args.output_dir / "continuity.csv")

    # fail件数の多い上位20アノテーションをコンソールに表示
    fails = [r for r in ann_records if "fail" in r["flag"]]
    if fails:
        print(f"\n--- failアノテーション上位20件（li順） ---")
        fails_sorted = sorted(
            [r for r in fails if not math.isnan(r["li"])],
            key=lambda x: x["li"],
        )[:20]
        for r in fails_sorted:
            print(
                f"  {r['sample']}/{r['vertebra']}/slice_{r['slice_idx']:03d}/{r['line_id']}"
                f"  flag={r['flag']}  LI={r['li']:.3f}  RMSE={r['rmse_perp']:.2f}px"
                f"  kappa={r['kappa_bar_deg']:.1f}deg"
            )

    # 連続性failの上位20件
    cont_fails = [r for r in cont_records if "fail" in r["flag"]]
    if cont_fails:
        print(f"\n--- 連続性failペア上位20件（Δφ順） ---")
        cont_sorted = sorted(cont_fails, key=lambda x: -x["delta_phi_deg"])[:20]
        for r in cont_sorted:
            print(
                f"  {r['sample']}/{r['vertebra']}"
                f"  slice {r['slice_from']}→{r['slice_to']}/{r['line_id']}"
                f"  Δφ={r['delta_phi_deg']:.1f}°  Δρ={r['delta_rho_px']:.1f}px"
            )


if __name__ == "__main__":
    main()
