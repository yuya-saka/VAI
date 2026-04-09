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

## プロジェクト構成

| ディレクトリ | 概要 | 状態 |
|-------------|------|------|
| `line_only/` | 直線検出のみ（TinyUNet）、baseline | ✅ 完成・実験済み |
| `multitask/` | 直線検出 + セグメンテーション（ResUNet）、現在のメイン | ✅ 実装完了・実験中 |

---

## line_only/

```
line_only/
├── train.py              ← 全fold実行エントリポイント
├── src/   model.py / dataset.py / data_utils.py / trainer.py
├── utils/ losses.py / metrics.py / detection.py / visualization.py
└── test/  ユニットテスト
```

```bash
uv run python Unet/line_only/train.py --config Unet/config/config.yaml
uv run pytest Unet/line_only/test/ -v
```

## multitask/

```
multitask/
├── train.py              ← 全fold実行エントリポイント
├── config/config.yaml    ← multitask専用config
├── src/   model.py / dataset.py / data_utils.py / trainer.py
├── utils/ losses.py / metrics.py / detection.py / visualization.py
└── test/  test_model.py / test_losses.py（13/13 pass）
```

```bash
uv run python Unet/multitask/train.py --config Unet/multitask/config/config.yaml
uv run pytest Unet/multitask/test/ -v
```
# チェックポイント保存
 /checkpointing 
---

## このディレクトリ固有のルール

- `Unet/` 以下のコメント・docstring は **日本語** で書く（`.claude/rules/language.md` 参照）
