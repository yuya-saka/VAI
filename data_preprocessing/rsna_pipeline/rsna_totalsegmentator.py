"""RSNA NIfTIファイルにTotalSegmentatorを適用しC1-C7椎体マスクを生成（並列処理）。

単一GPU内で複数ワーカープロセスを起動することでスループットを向上させる。
あるワーカーがCPUバウンドなリサンプリング・保存処理を行っている間、
別のワーカーがGPU推論を実行できるため、CPU処理とGPU処理がオーバーラップする。
"""

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path

# NFS上でPythonのmultiprocessingが一時ディレクトリを削除しようとすると
# "Device or resource busy" エラーが発生するため、/tmp にリダイレクト
os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RSNA_DATA_DIR = PROJECT_ROOT / "data" / "rsna_data"
NIFTI_DIR = RSNA_DATA_DIR / "nifti"
SEG_OUTPUT_DIR = RSNA_DATA_DIR / "segmentations"

VERTEBRAE_ROI = [
    "vertebrae_C1",
    "vertebrae_C2",
    "vertebrae_C3",
    "vertebrae_C4",
    "vertebrae_C5",
    "vertebrae_C6",
    "vertebrae_C7",
]


def is_already_done(seg_dir: Path) -> bool:
    """椎体マスクが1ファイル以上存在すれば処理済みと判定する。"""
    if not seg_dir.exists():
        return False
    existing = list(seg_dir.glob("vertebrae_C*.nii.gz"))
    return len(existing) >= 1


def _worker_loop(
    worker_id: int,
    gpu_id: int,
    nifti_paths: list[Path],
    output_dir: Path,
    threads: int,
    result_path: Path,
) -> None:
    """割り当てられたNIfTIファイルを処理し、結果をJSONファイルに書き込む。

    非daemonプロセスとして起動することで、TotalSegmentator/nnUNet が
    内部で子プロセスを生成できるようにする。
    CUDA_VISIBLE_DEVICES とスレッド数は torch/totalsegmentator の
    import より前に設定する必要がある。
    Queueによる結果受け渡しは長時間実行時にデッドロックが発生するため、
    ファイルベースの結果収集を使用する。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("TMPDIR", "/tmp")
    tempfile.tempdir = "/tmp"

    import torch
    from totalsegmentator.python_api import totalsegmentator

    torch.set_num_threads(threads)

    success = 0
    fail = 0
    failed_ids: list[str] = []
    total = len(nifti_paths)

    for i, nifti_path in enumerate(nifti_paths):
        study_id = nifti_path.name.replace(".nii.gz", "")
        seg_dir = output_dir / study_id

        if is_already_done(seg_dir):
            continue

        t0 = time.time()
        try:
            totalsegmentator(
                str(nifti_path),
                str(seg_dir),
                roi_subset=VERTEBRAE_ROI,
                fast=False,
                quiet=True,
                nr_thr_resamp=threads,
                nr_thr_saving=min(threads, 4),
            )
            n_masks = len(list(seg_dir.glob("vertebrae_C*.nii.gz")))
            elapsed = time.time() - t0

            if n_masks >= 1:
                success += 1
                print(
                    f"[w{worker_id}] ({i + 1}/{total}) {study_id} OK "
                    f"({n_masks} masks, {elapsed:.1f}s)",
                    flush=True,
                )
            else:
                fail += 1
                failed_ids.append(study_id)
                print(
                    f"[w{worker_id}] ({i + 1}/{total}) {study_id} FAILED (maskなし)",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            fail += 1
            failed_ids.append(study_id)
            print(f"[w{worker_id}] ({i + 1}/{total}) {study_id} ERROR: {e}", flush=True)
            if os.environ.get("DEBUG") == "1":
                traceback.print_exc()

    result_path.write_text(
        json.dumps({"success": success, "fail": fail, "failed_ids": failed_ids})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA TotalSegmentator 一括処理（C1-C7）")
    parser.add_argument("--nifti_dir", type=Path, default=NIFTI_DIR)
    parser.add_argument("--output_dir", type=Path, default=SEG_OUTPUT_DIR)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="使用するGPU番号（複数指定可）例: --gpus 0 1",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="各GPU上で起動する並列ワーカー数",
    )
    parser.add_argument("--limit", type=int, default=None, help="テスト用：処理するファイル数の上限")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(args.nifti_dir.glob("*.nii.gz"))
    if args.limit:
        nifti_files = nifti_files[: args.limit]

    # 処理済みをスキップしてからシャードに分配することで負荷を均等にする
    pending = [
        f for f in nifti_files
        if not is_already_done(args.output_dir / f.name.replace(".nii.gz", ""))
    ]
    skipped = len(nifti_files) - len(pending)

    # gpu_assignments[wid] = そのワーカーが使うGPU番号
    gpu_assignments: list[int] = []
    for gpu_id in args.gpus:
        gpu_assignments.extend([gpu_id] * args.workers_per_gpu)

    if pending:
        n_workers = min(len(gpu_assignments), len(pending))
    else:
        n_workers = 1
    gpu_assignments = gpu_assignments[:n_workers]

    cpu_total = os.cpu_count() or 8
    threads = max(1, cpu_total // max(1, n_workers))

    # ラウンドロビンで症例をシャードに均等分配
    shards: list[list[Path]] = [[] for _ in range(n_workers)]
    for idx, f in enumerate(pending):
        shards[idx % n_workers].append(f)

    print(f"処理対象: {len(nifti_files)} NIfTIファイル")
    print(f"処理済みスキップ: {skipped}")
    print(f"未処理: {len(pending)}")
    print(
        f"GPU: {args.gpus}, Workers/GPU: {args.workers_per_gpu}, "
        f"合計ワーカー数: {n_workers}, スレッド/ワーカー: {threads}"
    )
    print(f"ワーカー->GPU割り当て: {gpu_assignments}")
    print(f"出力先: {args.output_dir}")
    print("-" * 60)

    if not pending:
        print("処理対象なし。")
        return

    t0 = time.time()
    ctx = mp.get_context("spawn")

    result_dir = Path(tempfile.mkdtemp(prefix="rsna_seg_results_"))
    result_paths = [result_dir / f"worker_{wid}.json" for wid in range(n_workers)]

    # 非daemonプロセスとして起動（TotalSegmentatorが子プロセスを生成できるようにするため）
    procs = [
        ctx.Process(
            target=_worker_loop,
            args=(
                wid,
                gpu_assignments[wid],
                shards[wid],
                args.output_dir,
                threads,
                result_paths[wid],
            ),
            daemon=False,
        )
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    total_success = 0
    total_fail = 0
    all_failed: list[str] = []
    for rp in result_paths:
        if not rp.exists():
            print(f"[WARN] 結果ファイルが見つかりません: {rp.name}（ワーカーがクラッシュした可能性あり）")
            continue
        r = json.loads(rp.read_text())
        total_success += r["success"]
        total_fail += r["fail"]
        all_failed.extend(r["failed_ids"])

    elapsed = time.time() - t0
    print("-" * 60)
    print(
        f"完了（{elapsed / 60:.1f}分）。"
        f"success={total_success}, skipped={skipped}, failed={total_fail}"
    )
    if all_failed:
        print("失敗したStudy:")
        for fid in all_failed:
            print(f"  {fid}")


if __name__ == "__main__":
    main()
