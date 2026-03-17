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

## 📝 現在の作業内容

### 実装中の機能

**line_only/ ディレクトリ**
- Heatmap-moment ベースの直線検出
- 幾何学的制約（phi, rho）による損失関数
- 段階的 warmup による学習

主要ファイル：
- `line_only/line_detection.py` - 直線検出ロジック
- `line_only/line_losses.py` - 損失関数
- `line_only/line_metrics.py` - 評価メトリクス
- `line_only/train_heat.py` - 訓練スクリプト

### 最近の変更

- `train_heat.py` を削除 → `line_only/train_heat.py` に移行
- `run_all_folds.py` を `train.py` にリネーム
- matplotlib の日本語対応仕様決定

### TODO (詳細は DESIGN.md 参照)

主要タスク：
- [ ] 訓練/評価パイプラインのリファクタリング
- [ ] 評価ユーティリティの抽出
- [ ] コメントの英語化

---

## 🛠️ よく使うコマンド

```bash
# 訓練実行
uv run python Unet/line_only/train_heat.py

# 全 fold 実行
uv run python Unet/train.py

# テスト
uv run pytest Unet/line_only/test_line_losses.py -v

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
