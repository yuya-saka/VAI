"""RSNA DICOMシリーズをNIfTIに一括変換（並列処理）。"""

import argparse
import gzip
import os
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
TRAIN_IMAGES_DIR = RSNA_DATA_DIR / "train_images"
NIFTI_OUTPUT_DIR = RSNA_DATA_DIR / "nifti"


def _gzip_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _cleanup_extras(output_dir: Path, study_id: str, keep: Path) -> None:
    # 本体以外のnii.gz・nii・jsonを削除
    for p in output_dir.glob(f"{study_id}*.nii.gz"):
        if p != keep:
            p.unlink()
    for p in output_dir.glob(f"{study_id}*.nii"):
        p.unlink()
    for p in output_dir.glob(f"{study_id}*.json"):
        p.unlink()


def convert_one(study_dir: Path, output_dir: Path) -> bool:
    study_id = study_dir.name
    expected_output = output_dir / f"{study_id}.nii.gz"

    if expected_output.exists():
        return True

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", study_id,
        "-o", str(output_dir),
        str(study_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # .nii.gz と .nii の両方を収集（dcm2niixが複数シリーズを生成する場合がある）
    candidates_gz = list(output_dir.glob(f"{study_id}*.nii.gz"))
    candidates_nii = list(output_dir.glob(f"{study_id}*.nii"))
    all_candidates = candidates_gz + candidates_nii

    if not all_candidates:
        print(f"  [ERROR] dcm2niix 出力なし: {result.stderr[:200]}")
        return False

    # 最大サイズのファイルをCT本体として採用（スカウト像・小シリーズを除外）
    all_candidates.sort(key=lambda f: f.stat().st_size, reverse=True)
    best = all_candidates[0]

    if best.suffix == ".gz":
        best.rename(expected_output)
    else:
        # .nii を gzip 圧縮して .nii.gz に変換
        _gzip_file(best, expected_output)

    _cleanup_extras(output_dir, study_id, expected_output)
    return expected_output.exists()


def _convert_worker(args: tuple[Path, Path]) -> tuple[str, str]:
    """並列変換ワーカー。(study_id, status) を返す。"""
    study_dir, output_dir = args
    study_id = study_dir.name
    expected = output_dir / f"{study_id}.nii.gz"

    if expected.exists():
        return (study_id, "skip")

    try:
        ok = convert_one(study_dir, output_dir)
        return (study_id, "ok" if ok else "fail")
    except Exception as e:  # noqa: BLE001
        return (study_id, f"error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA DICOM -> NIfTI 一括変換")
    parser.add_argument("--input_dir", type=Path, default=TRAIN_IMAGES_DIR)
    parser.add_argument("--output_dir", type=Path, default=NIFTI_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="テスト用：処理するStudy数の上限")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="並列プロセス数",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    study_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    if args.limit:
        study_dirs = study_dirs[: args.limit]

    total = len(study_dirs)
    tasks = [(d, args.output_dir) for d in study_dirs]

    success = 0
    skip = 0
    fail = 0
    failed_ids: list[str] = []
    t0 = time.time()

    print(f"処理対象: {total} studies")
    print(f"出力先: {args.output_dir}")
    print(f"ワーカー数: {args.workers}")
    print("-" * 60)

    with Pool(processes=args.workers) as pool:
        for i, (study_id, status) in enumerate(
            pool.imap_unordered(_convert_worker, tasks)
        ):
            done = i + 1
            if status == "ok":
                success += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failed_ids.append(study_id)
                print(f"  [FAIL] {study_id}: {status}")

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                print(
                    f"[{done}/{total}] ok={success} skip={skip} fail={fail} "
                    f"({rate:.1f}/s, 残り ~{remaining / 60:.0f}分)"
                )

    print("-" * 60)
    print(f"完了。success={success}, skipped={skip}, failed={fail}")
    if failed_ids:
        print("失敗したStudy:")
        for fid in failed_ids:
            print(f"  {fid}")


if __name__ == "__main__":
    main()
