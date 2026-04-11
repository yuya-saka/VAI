"""
画像ごとのセグメンテーション評価スクリプト

seg_only モデルと multitask モデルを fold ごとに比較し、
per-image の dice/IoU を集計・比較する。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


# -------------------------
# パス設定
# -------------------------
def _setup_sys_path() -> Path:
    """プロジェクトルートと Unet/ を sys.path に追加する"""
    script_dir = Path(__file__).resolve().parent  # Unet/
    project_root = script_dir.parent              # VAI/
    for p in [str(project_root), str(script_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return project_root


PROJECT_ROOT = _setup_sys_path()


# -------------------------
# 設定ファイル読み込み
# -------------------------
def load_yaml_config(cfg_path: str) -> dict[str, Any]:
    """YAML 設定ファイルを読み込む"""
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {cfg_path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    print(f"[INFO] 設定読込: {p.resolve()}")
    return cfg


# -------------------------
# seg_only モデルのロード
# -------------------------
def load_seg_only_model(
    cfg: dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """seg_only チェックポイントからモデルをロードする"""
    from seg_only.src.data_utils import create_model_optimizer_scheduler

    model, _, _ = create_model_optimizer_scheduler(cfg, device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] seg_only チェックポイント読込: {ckpt_path}")
    return model


# -------------------------
# multitask モデルのロード
# -------------------------
def load_multitask_model(
    cfg: dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """multitask チェックポイントからモデルをロードする"""
    from multitask.src.data_utils import create_model_optimizer_scheduler

    model, _, _ = create_model_optimizer_scheduler(cfg, device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] multitask チェックポイント読込: {ckpt_path}")
    return model


# -------------------------
# DataLoader 作成（seg_only）
# -------------------------
def create_seg_only_test_loader(
    cfg: dict[str, Any],
    fold: int,
) -> Any:
    """seg_only 用テスト DataLoader を作成する（batch_size=1 固定）"""
    from seg_only.src.data_utils import (
        create_data_loaders,
        prepare_datasets_and_splits,
    )

    # fold を上書き
    cfg_copy = {**cfg}
    cfg_copy["data"] = {**cfg.get("data", {}), "test_fold": fold}
    # バッチサイズを 1 に固定して per-image 評価を可能にする
    cfg_copy["training"] = {**cfg.get("training", {}), "batch_size": 1}

    train_s, val_s, test_s, root_dir, group, image_size, seed = prepare_datasets_and_splits(cfg_copy)
    _, _, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, seed, cfg_copy
    )
    return test_loader


# -------------------------
# DataLoader 作成（multitask）
# -------------------------
def create_multitask_test_loader(
    cfg: dict[str, Any],
    fold: int,
) -> Any:
    """multitask 用テスト DataLoader を作成する（batch_size=1 固定）"""
    from multitask.src.data_utils import (
        create_data_loaders,
        prepare_datasets_and_splits,
    )

    # fold を上書き
    cfg_copy = {**cfg}
    cfg_copy["data"] = {**cfg.get("data", {}), "test_fold": fold}
    # バッチサイズを 1 に固定して per-image 評価を可能にする
    cfg_copy["training"] = {**cfg.get("training", {}), "batch_size": 1}

    train_s, val_s, test_s, root_dir, group, image_size, sigma, seed = prepare_datasets_and_splits(cfg_copy)
    _, _, test_loader = create_data_loaders(
        train_s, val_s, test_s, root_dir, group, image_size, sigma, seed, cfg_copy
    )
    return test_loader


# -------------------------
# seg_only モデルの per-image 評価
# -------------------------
@torch.no_grad()
def eval_seg_only_per_image(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    fold: int,
) -> list[dict[str, Any]]:
    """
    seg_only モデルを画像ごとに評価して結果リストを返す

    戻り値:
        各画像の dict: sample, vertebra, slice_idx, fold, fg_mdice, fg_miou, per_class
    """
    from seg_only.utils.metrics import compute_seg_fg_metrics

    model.eval()
    results: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["image"].to(device).float()
        gt_mask = batch["gt_region_mask"].to(device).long()

        out = model(x)
        seg_logits = out["seg_logits"]

        # batch_size=1 のため 1 画像分のみ
        metrics = compute_seg_fg_metrics(seg_logits, gt_mask)

        record: dict[str, Any] = {
            "sample": batch["sample"][0],
            "vertebra": batch["vertebra"][0],
            "slice_idx": int(batch["slice_idx"][0]),
            "fold": fold,
            "seg_only_fg_mdice": metrics["fg_mdice"],
            "seg_only_fg_miou": metrics["fg_miou"],
            "seg_only_per_class": metrics["per_class"],
        }
        results.append(record)

    print(f"  [seg_only] fold={fold}: {len(results)} 画像を評価")
    return results


# -------------------------
# multitask モデルの per-image 評価
# -------------------------
@torch.no_grad()
def eval_multitask_per_image(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    fold: int,
) -> dict[str, dict[str, Any]]:
    """
    multitask モデルを画像ごとに評価し、キー付き辞書で返す

    戻り値:
        {image_key: {fg_mdice, fg_miou, per_class}} の辞書
        image_key = "{sample}_{vertebra}_{slice_idx}"
        has_seg_label=False の画像は除外される
    """
    from seg_only.utils.metrics import compute_seg_fg_metrics

    model.eval()
    results: dict[str, dict[str, Any]] = {}

    for batch in loader:
        x = batch["image"].to(device).float()

        # gt_mask と has_seg_label を取得（multitask trainer の _extract_seg_batch と同様）
        if "gt_mask" in batch:
            gt_mask = batch["gt_mask"].to(device).long()
        else:
            gt_mask = batch["gt_region_mask"].to(device).long()

        if "has_seg_label" in batch:
            has_seg_label = batch["has_seg_label"].to(device).bool()
        elif "has_gt_region_mask" in batch:
            has_seg_label = batch["has_gt_region_mask"].to(device).bool()
        else:
            # フォールバック: 全サンプルに seg ラベルありとみなす
            has_seg_label = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        # seg ラベルがない画像はスキップ
        if not has_seg_label[0].item():
            continue

        out = model(x)
        seg_logits = out["seg_logits"]

        # batch_size=1 のため 1 画像分のみ
        metrics = compute_seg_fg_metrics(seg_logits, gt_mask)

        sample_name = batch["sample"][0]
        vertebra = batch["vertebra"][0]
        slice_idx = int(batch["slice_idx"][0])
        key = f"{sample_name}_{vertebra}_{slice_idx}"

        results[key] = {
            "fg_mdice": metrics["fg_mdice"],
            "fg_miou": metrics["fg_miou"],
            "per_class": metrics["per_class"],
        }

    print(f"  [multitask] fold={fold}: {len(results)} 画像を評価（seg ラベルあり）")
    return results


# -------------------------
# 1 fold の評価
# -------------------------
def evaluate_fold(
    fold: int,
    seg_only_cfg: dict[str, Any],
    multitask_cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """
    1 つの fold について seg_only・multitask 両モデルを評価する

    戻り値:
        per-image の結果リスト（両モデルのスコアを含む）
    """
    print(f"\n{'='*60}")
    print(f"[FOLD {fold}] 評価開始")
    print(f"{'='*60}")

    # チェックポイントパス（project root からの相対パス）
    seg_exp = seg_only_cfg.get("experiment", {})
    mt_exp = multitask_cfg.get("experiment", {})

    seg_ckpt_dir = PROJECT_ROOT / "Unet" / "outputs" / seg_exp.get("phase", "seg_only_v1") / seg_exp.get("name", "baseline") / "checkpoints"
    mt_ckpt_dir = PROJECT_ROOT / "Unet" / "outputs" / mt_exp.get("phase", "multitask_seg下げる") / mt_exp.get("name", "sig3.5_alpha0.02_base") / "checkpoints"

    seg_ckpt = seg_ckpt_dir / f"best_fold{fold}.pt"
    mt_ckpt = mt_ckpt_dir / f"best_fold{fold}.pt"

    # DataLoader 作成
    print(f"  [データ] seg_only test DataLoader を作成中...")
    seg_loader = create_seg_only_test_loader(seg_only_cfg, fold)
    print(f"  [データ] multitask test DataLoader を作成中...")
    mt_loader = create_multitask_test_loader(multitask_cfg, fold)

    # モデルロード
    seg_model = load_seg_only_model(seg_only_cfg, seg_ckpt, device)
    mt_model = load_multitask_model(multitask_cfg, mt_ckpt, device)

    # per-image 評価
    print(f"  [評価] seg_only モデルを評価中...")
    seg_results = eval_seg_only_per_image(seg_model, seg_loader, device, fold)

    print(f"  [評価] multitask モデルを評価中...")
    mt_results_dict = eval_multitask_per_image(mt_model, mt_loader, device, fold)

    # 結果をマージ：seg_only の各画像に multitask のスコアを付加
    merged: list[dict[str, Any]] = []
    for rec in seg_results:
        key = f"{rec['sample']}_{rec['vertebra']}_{rec['slice_idx']}"
        mt_score = mt_results_dict.get(key)

        merged_rec = {**rec}
        if mt_score is not None:
            merged_rec["multitask_fg_mdice"] = mt_score["fg_mdice"]
            merged_rec["multitask_fg_miou"] = mt_score["fg_miou"]
            merged_rec["multitask_per_class"] = mt_score["per_class"]
        else:
            # seg ラベルなしのため multitask スコア未取得
            merged_rec["multitask_fg_mdice"] = None
            merged_rec["multitask_fg_miou"] = None
            merged_rec["multitask_per_class"] = None

        merged.append(merged_rec)

    # GPU メモリ解放
    del seg_model, mt_model
    torch.cuda.empty_cache()

    return merged


# -------------------------
# サマリー分析
# -------------------------
def analyze_hard_cases(
    all_results: list[dict[str, Any]],
    hard_thresh: float = 0.85,
) -> dict[str, Any]:
    """
    hard cases（seg_only の fg_mdice < threshold）を分析する

    引数:
        all_results: 全 fold の per-image 結果リスト
        hard_thresh: ハードケースの閾値

    戻り値:
        サマリー統計の辞書
    """
    # 両モデルのスコアがある画像のみ対象
    comparable = [r for r in all_results if r.get("multitask_fg_mdice") is not None]
    hard_cases = [r for r in comparable if r["seg_only_fg_mdice"] < hard_thresh]
    easy_cases = [r for r in comparable if r["seg_only_fg_mdice"] >= hard_thresh]

    if len(hard_cases) == 0:
        return {
            "total_images": len(all_results),
            "comparable_images": len(comparable),
            "hard_cases_count": 0,
            "hard_thresh": hard_thresh,
            "message": f"閾値 {hard_thresh} 未満のハードケースは存在しません",
        }

    # ハードケースの改善分析
    improved = [r for r in hard_cases if r["multitask_fg_mdice"] > r["seg_only_fg_mdice"]]
    degraded = [r for r in hard_cases if r["multitask_fg_mdice"] < r["seg_only_fg_mdice"]]
    unchanged = [r for r in hard_cases if r["multitask_fg_mdice"] == r["seg_only_fg_mdice"]]

    deltas = [r["multitask_fg_mdice"] - r["seg_only_fg_mdice"] for r in hard_cases]
    mean_delta = sum(deltas) / len(deltas)

    # per-class 比較（ハードケースのみ）
    class_names = ["body", "right", "left", "posterior"]
    per_class_comparison: dict[str, dict[str, float]] = {}
    for cls_name in class_names:
        seg_only_dices = [
            r["seg_only_per_class"][cls_name]["dice"]
            for r in hard_cases
            if cls_name in r.get("seg_only_per_class", {})
        ]
        mt_dices = [
            r["multitask_per_class"][cls_name]["dice"]
            for r in hard_cases
            if cls_name in r.get("multitask_per_class", {})
        ]
        if seg_only_dices and mt_dices:
            per_class_comparison[cls_name] = {
                "seg_only_mean_dice": sum(seg_only_dices) / len(seg_only_dices),
                "multitask_mean_dice": sum(mt_dices) / len(mt_dices),
                "delta": sum(mt_dices) / len(mt_dices) - sum(seg_only_dices) / len(seg_only_dices),
            }

    # easy cases の比較
    easy_deltas = [r["multitask_fg_mdice"] - r["seg_only_fg_mdice"] for r in easy_cases]
    easy_mean_delta = sum(easy_deltas) / len(easy_deltas) if easy_deltas else 0.0

    return {
        "total_images": len(all_results),
        "comparable_images": len(comparable),
        "hard_cases_count": len(hard_cases),
        "hard_thresh": hard_thresh,
        "hard_cases_improved": len(improved),
        "hard_cases_degraded": len(degraded),
        "hard_cases_unchanged": len(unchanged),
        "hard_cases_improve_rate": len(improved) / len(hard_cases),
        "hard_cases_mean_delta_fg_mdice": mean_delta,
        "hard_cases_per_class": per_class_comparison,
        "easy_cases_count": len(easy_cases),
        "easy_cases_mean_delta_fg_mdice": easy_mean_delta,
        "overall_mean_delta_fg_mdice": sum(r["multitask_fg_mdice"] - r["seg_only_fg_mdice"] for r in comparable) / len(comparable),
    }


# -------------------------
# サマリーテーブルの出力
# -------------------------
def print_summary_table(summary: dict[str, Any]) -> None:
    """サマリー統計を表形式で出力する"""
    print("\n" + "=" * 70)
    print("  Per-Image 比較サマリー")
    print("=" * 70)
    print(f"  全評価画像数:           {summary['total_images']}")
    print(f"  比較可能画像数:         {summary['comparable_images']}")
    print(f"  ハードケース閾値:       fg_mdice < {summary['hard_thresh']}")
    print(f"  ハードケース数:         {summary['hard_cases_count']}")

    if summary.get("hard_cases_count", 0) == 0:
        print(f"\n  {summary.get('message', 'ハードケースなし')}")
        print("=" * 70)
        return

    print()
    print("  [ハードケース分析]")
    print(f"  改善:     {summary['hard_cases_improved']} 画像"
          f" ({summary['hard_cases_improve_rate']*100:.1f}%)")
    print(f"  悪化:     {summary['hard_cases_degraded']} 画像")
    print(f"  変化なし: {summary['hard_cases_unchanged']} 画像")
    print(f"  平均 Δfg_mdice (ハードケース): {summary['hard_cases_mean_delta_fg_mdice']:+.4f}")

    print()
    print("  [Per-Class Dice 比較（ハードケースのみ）]")
    print(f"  {'クラス':<12} {'seg_only':>10} {'multitask':>10} {'Δ':>8}")
    print(f"  {'-'*44}")
    for cls_name, vals in summary.get("hard_cases_per_class", {}).items():
        print(
            f"  {cls_name:<12} "
            f"{vals['seg_only_mean_dice']:>10.4f} "
            f"{vals['multitask_mean_dice']:>10.4f} "
            f"{vals['delta']:>+8.4f}"
        )

    print()
    print("  [全比較可能画像の平均 Δfg_mdice]")
    print(f"  ハードケース: {summary['hard_cases_mean_delta_fg_mdice']:+.4f}")
    print(f"  イージーケース: {summary['easy_cases_mean_delta_fg_mdice']:+.4f}")
    print(f"  全体: {summary['overall_mean_delta_fg_mdice']:+.4f}")
    print("=" * 70)


# -------------------------
# コマンドライン引数
# -------------------------
def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="seg_only vs multitask の per-image セグメンテーション比較評価"
    )
    parser.add_argument(
        "--seg_only_config",
        type=str,
        default="Unet/seg_only/config/config.yaml",
        help="seg_only モデルの設定ファイルパス",
    )
    parser.add_argument(
        "--multitask_config",
        type=str,
        default="Unet/multitask/config/config.yaml",
        help="multitask モデルの設定ファイルパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Unet/outputs/eval_per_image",
        help="結果保存ディレクトリ",
    )
    parser.add_argument(
        "--hard_thresh",
        type=float,
        default=0.85,
        help="ハードケース判定閾値（fg_mdice がこの値未満をハードケースとする）",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="使用する GPU の ID",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="評価する fold のカンマ区切りリスト（例: 0,1,2,3,4）",
    )
    return parser.parse_args()


# -------------------------
# メイン処理
# -------------------------
def main() -> None:
    """メイン処理：全 fold の per-image 評価を実行しサマリーを出力する"""
    args = parse_args()

    # デバイス設定
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] 使用デバイス: {device}")

    # 設定ファイル読み込み
    seg_only_cfg = load_yaml_config(args.seg_only_config)
    multitask_cfg = load_yaml_config(args.multitask_config)

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 結果保存先: {output_dir.resolve()}")

    # 評価する fold のリスト
    folds = [int(f.strip()) for f in args.folds.split(",")]
    print(f"[INFO] 評価 fold: {folds}")

    # 全 fold を評価
    all_results: list[dict[str, Any]] = []
    for fold in folds:
        fold_results = evaluate_fold(
            fold=fold,
            seg_only_cfg=seg_only_cfg,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        all_results.extend(fold_results)
        print(f"  [完了] fold={fold}: {len(fold_results)} 画像")

    print(f"\n[INFO] 全 fold 合計: {len(all_results)} 画像を評価")

    # サマリー分析
    summary = analyze_hard_cases(all_results, hard_thresh=args.hard_thresh)

    # 結果保存
    output_data = {
        "summary": summary,
        "per_image_results": all_results,
        "config": {
            "seg_only_config": args.seg_only_config,
            "multitask_config": args.multitask_config,
            "hard_thresh": args.hard_thresh,
            "folds": folds,
        },
    }
    output_path = output_dir / "per_image_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 結果を保存しました: {output_path}")

    # サマリーテーブルを出力
    print_summary_table(summary)


if __name__ == "__main__":
    main()
