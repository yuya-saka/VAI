# Unet/ 作業ガイド

## ドキュメント構成

| ファイル | 内容 | 更新タイミング |
|---------|------|--------------|
| **`../.claude/docs/unet-work-log.md`** | 現在の精度・設計決定・次にやること（サマリー） | 大きな変更時 |
| **`../.claude/docs/work-logs/YYYY-MM-DD.md`** | 日別の詳細作業ログ | 各セッション終了時 |
| **`../.claude/docs/DESIGN.md`** | アーキテクチャ・設計決定の詳細 | 重要な設計判断時 |
| **`config/config.yaml`** | 訓練設定、ハイパーパラメータ | 実験時 |

**セッション開始時は `unet-work-log.md` を読む。**

---

## line_only/ ディレクトリ構成

```
line_only/
├── train.py              ← 全fold実行エントリポイント
├── src/
│   ├── model.py          TinyUNet（椎体条件付きあり）
│   ├── dataset.py        PngLineDataset, データ拡張
│   ├── data_utils.py     config読込, seed, k-fold分割, DataLoader
│   └── trainer.py        訓練ループ, evaluate, train_one_fold
├── utils/
│   ├── losses.py         損失関数（L_mse + w(t)·L_line）
│   ├── metrics.py        評価メトリクス（angle, rho, perp）
│   ├── detection.py      直線検出（momentsベース, threshold=0.2）
│   └── visualization.py  描画関数
├── shim/                 ← 旧コード保管（参照用・import不可）
└── test/                 ← ユニットテスト（23ファイル）
```

---

## よく使うコマンド

```bash
# 全fold実行
uv run python Unet/line_only/train.py --config Unet/config/config.yaml

# 特定foldのみ
uv run python Unet/line_only/train.py --start_fold 0 --end_fold 0

# テスト
uv run pytest Unet/line_only/test/ -v

# 設定確認
cat Unet/config/config.yaml
```
# チェックポイント保存
 /checkpointing 
---

## このディレクトリ固有のルール

- `Unet/` 以下のコメント・docstring は **日本語** で書く（`.claude/rules/language.md` 参照）
