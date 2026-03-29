# Unet/ 作業ガイド

このディレクトリでの Claude Code との作業記録と管理方法をまとめています。

---

## 📁 ドキュメント構成

| ファイル | 内容 | 更新頻度 |
|---------|------|---------|
| **このファイル** (`Unet/CLAUDE.md`) | 作業の進め方、記録場所の案内 | セッション開始時に確認 |
| **`../.claude/docs/DESIGN.md`** | 設計決定、アーキテクチャ、TODO | 重要な設計判断時 |
| **`../.claude/docs/unet-work-log.md`** | 日々の作業記録 | セッション終了時 |
| **`config/config.yaml`** | 訓練設定、ハイパーパラメータ | 実験時 |

---

## 🔄 作業の流れ

### 1. セッション開始時

```markdown
1. `Unet/CLAUDE.md` (このファイル) を開く
2. `../.claude/docs/unet-work-log.md` で前回の続きを確認
3. Claude に「どこまでやったか覚えてる？」と聞く
4. TODO を確認して作業開始
```

### 2. 作業中

- **設計判断**: Claude が自動で `DESIGN.md` に記録 (`design-tracker` skill)
- **コード変更**: git で管理
- **実験結果**: work-log に記録

### 3. セッション終了時

```bash
# 作業ログに今日の内容を追記
# Claude に依頼: 「今日の作業を work-log に記録して」

# チェックポイント保存（オプション）
/checkpointing --full
```

---

## 📁 line_only/ ディレクトリ構成

```
line_only/
├── train.py              ← 全fold実行エントリポイント
├── __init__.py
├── src/                  ← 訓練パイプライン
│   ├── model.py          TinyUNet (DoubleConv)
│   ├── dataset.py        PngLineDataset, データ拡張
│   ├── data_utils.py     config読込, seed, k-fold分割, DataLoader
│   └── trainer.py        訓練ループ, evaluate, train_one_fold
├── utils/                ← 汎用ロジック（再利用可）
│   ├── losses.py         損失関数 (phi, rho, MSE)
│   ├── metrics.py        評価メトリクス (angle, rho, perp)
│   ├── detection.py      直線検出 (moments ベース)
│   └── visualization.py  描画関数
├── shim/                 ← 旧コード保管（参照用・import 不可）
│   ├── train_heat.py
│   ├── line_detection.py
│   ├── line_losses.py
│   └── line_metrics.py
└── test/                 ← ユニットテスト (23ファイル)
```

### 実装済み機能

- Heatmap-moment ベースの直線検出
- 幾何学的制約（phi, rho）による損失関数
- 段階的 warmup による学習
- 5-fold クロスバリデーション
- wandb ログ統合

### TODO

主要タスク：
- [ ] sigma チューニング実験の継続

---

## 🛠️ よく使うコマンド

```bash
# 全 fold 実行（line_only/ から）
uv run python Unet/line_only/train.py --config config/config.yaml

# 特定 fold のみ
uv run python Unet/line_only/train.py --start_fold 0 --end_fold 0

# 全椎体モード
uv run python Unet/line_only/train.py --all_vertebrae

# テスト
uv run pytest Unet/line_only/test/ -v

# 設定確認
cat Unet/config/config.yaml
```

---

## 💡 Tips

### Claude への依頼例

- 「前回の続きから作業して」
- 「DESIGN.md の TODO を確認して優先順位つけて」
- 「今日の作業を work-log に記録して」
- 「line_only/ の実装状況を教えて」

### 設計相談が必要なとき

Claude が自動で Codex CLI に相談します（`.claude/rules/codex-delegation.md` 参照）

### 大規模調査が必要なとき

Claude が自動で Gemini CLI に調査依頼します（`.claude/rules/gemini-delegation.md` 参照）

---

## 📊 実験管理

### 実験結果の記録先

- 訓練ログ: `Unet/logs/` (git ignore)
- チェックポイント: `Unet/checkpoints/` (git ignore)
- 実験メモ: `../.claude/docs/unet-work-log.md`

### 重要な実験は記録

```markdown
## 2026-03-13 Experiment

**Setup:**
- Loss: heatmap MSE + line geometry (phi, rho)
- Warmup: 10 epochs

**Results:**
- Angle error: XX.X degrees
- Notes: ...
```

---

## 🔗 関連リンク

- [プロジェクト全体のガイド](../CLAUDE.md)
- [設計ドキュメント](../.claude/docs/DESIGN.md)
- [開発環境](../.claude/rules/dev-environment.md)
- [コーディング規約](../.claude/rules/coding-principles.md)
- [言語規約](../.claude/rules/language.md)
- [matplotlibでの日本語使用規約](../.claude/rules/matplotlib-japanese.md)