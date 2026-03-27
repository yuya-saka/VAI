"""座標系修正後の動作確認（1サンプルテスト）"""
import sys
from pathlib import Path

import torch
import yaml

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from Unet.line_only.src.trainer import predict_lines_and_eval_test
from Unet.line_only.src.data_utils import create_data_loaders, get_model


def test_one_sample():
    """1サンプルだけ評価して角度誤差を確認"""
    # Config読み込み
    cfg_path = Path("Unet/config/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # データローダー作成（test_fold=2）
    cfg["data"]["test_fold"] = 2
    train_loader, val_loader, test_loader = create_data_loaders(cfg)

    # モデル読み込み
    device = torch.device(f"cuda:{cfg['training']['gpu_id']}")
    model = get_model(cfg).to(device)

    checkpoint_path = Path(
        "Unet/outputs/checkpoints_sig2.5_ALL_+angleloss_bug修正/best_fold2.pt"
    )
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 1サンプルだけ評価
    test_loader_small = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False,
    )

    out_dir = Path("Unet/outputs/test_one_sample")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg["data"]["root_dir"])

    summary = predict_lines_and_eval_test(
        cfg, model, test_loader_small, device, dataset_root, out_dir
    )

    print("\n=== 1サンプル評価結果 ===")
    print(f"Angle error: {summary['angle_error_deg_mean']:.2f}度")
    print(f"Rho error: {summary['rho_error_px_mean']:.2f}px")
    print(f"Perpendicular dist: {summary['perpendicular_dist_px_mean']:.2f}px")
    print(f"\n可視化: {out_dir}")


if __name__ == "__main__":
    test_one_sample()
