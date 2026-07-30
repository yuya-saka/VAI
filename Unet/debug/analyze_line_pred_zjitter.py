"""現行2Dモデルの予測を z 方向に平滑化したら誤差がどれだけ減るかを測る。

面（z方向に滑らかな構造）を学習することの上限効果を、
学習なしの後処理平滑化で近似的に見積もる。
"""

import csv
import sys
from collections import defaultdict

import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "Unet/outputs/line_20260616/sig4.0_ALL(CC-metrics)/vis/error_viz/line_records.csv"
)
D_NORM = np.sqrt(2) * 224


def unwrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.unwrap(phi * 2.0) / 2.0


def wrap_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """phi の [0, pi) 周期を考慮した差分（度）。"""
    d = np.degrees(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main() -> None:
    groups = defaultdict(list)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            groups[(row["sample"], row["vertebra"], row["line_name"])].append(
                (
                    int(row["slice_idx"]),
                    float(row["gt_phi"]),
                    float(row["gt_rho"]) * D_NORM,
                    float(row["pred_phi"]),
                    float(row["pred_rho"]) * D_NORM,
                )
            )

    raw_phi, raw_rho = [], []
    sm_phi = {1: [], 2: []}
    sm_rho = {1: [], 2: []}
    jitter_phi, jitter_rho = [], []

    for rows in groups.values():
        rows.sort()
        if len(rows) < 5:
            continue
        z = np.array([r[0] for r in rows], float)
        zc = z - z.mean()
        gt_phi = unwrap_phi(np.array([r[1] for r in rows]))
        gt_rho = np.array([r[2] for r in rows])
        pr_phi = unwrap_phi(np.array([r[3] for r in rows]))
        pr_rho = np.array([r[4] for r in rows])

        raw_phi.extend(wrap_diff(pr_phi, gt_phi))
        raw_rho.extend(np.abs(pr_rho - gt_rho))

        for deg in (1, 2):
            f_phi = np.polyval(np.polyfit(zc, pr_phi, deg), zc)
            f_rho = np.polyval(np.polyfit(zc, pr_rho, deg), zc)
            sm_phi[deg].extend(wrap_diff(f_phi, gt_phi))
            sm_rho[deg].extend(np.abs(f_rho - gt_rho))
            if deg == 1:
                # 予測自身の非平滑成分＝z方向のブレ
                jitter_phi.extend(np.abs(np.degrees(pr_phi - f_phi)))
                jitter_rho.extend(np.abs(pr_rho - f_rho))

    def stat(v: list, unit: str) -> str:
        a = np.array(v)
        return f"median {np.median(a):6.2f}{unit}  mean {a.mean():6.2f}{unit}  p90 {np.percentile(a, 90):6.2f}{unit}"

    print(f"対象: {len(raw_phi)} 本線（5スライス以上の椎体のみ）\n")
    print("--- 角度誤差 vs GT ---")
    print(f"  生の予測          {stat(raw_phi, ' deg')}")
    print(f"  z線形平滑化後      {stat(sm_phi[1], ' deg')}")
    print(f"  z2次平滑化後       {stat(sm_phi[2], ' deg')}")
    print("\n--- rho 誤差 vs GT ---")
    print(f"  生の予測          {stat(raw_rho, ' px')}")
    print(f"  z線形平滑化後      {stat(sm_rho[1], ' px')}")
    print(f"  z2次平滑化後       {stat(sm_rho[2], ' px')}")
    print("\n--- 予測自身の z 方向ブレ（線形からの乖離）---")
    print(f"  角度              {stat(jitter_phi, ' deg')}")
    print(f"  rho               {stat(jitter_rho, ' px')}")


if __name__ == "__main__":
    main()
