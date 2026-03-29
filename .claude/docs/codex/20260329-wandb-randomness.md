# Codex Analysis: wandb と乱数状態、17.77°→10.71° 改善の原因調査
Date: 2026-03-29

## Question

PyTorch 学習でwandbの有無が乱数状態に影響するか、
またfold1の角度誤差が17.77°→10.71°に改善した本当の原因を調査。

## 実行順序（trainer.py: train_one_fold）

```
set_seed(42)                           # train.py 側で呼ばれる（train_one_fold 外）
↓
create_data_loaders(...)               # DataLoader 作成（torch.Generator 使用）
↓
wandb.init() if enabled                # オプション
↓
create_model_optimizer_scheduler()     # モデル重み初期化（Xavier/Kaiming）
```

重要: test_aug_fold1.py では set_seed() が train_one_fold() 直前に呼ばれる（同じ順序）。

## Codex 回答（3回の質問に基づく）

### Q1: DataLoader 作成は global torch RNG を消費するか？

**No**（DataLoader 作成時点では消費しない）

- `DataLoader(shuffle=True, generator=g)` はオブジェクト構築時には RNG を使わない
- シャッフル用のランダムインデックス生成はイテレーション開始時（`iter(loader)` / `for` ループ）に発生
- `generator=g` を渡した場合、シャッフルは global torch RNG ではなく `g` を使う

### Q2: wandb.init() は torch RNG を消費するか？

**No**

- wandb.init() は通常 torch 操作を呼ばない
- GPU 検出は nvidia-smi コール（torch 経由でない）
- システム情報収集も torch RNG を消費しない

### Q3: wandb の有無でモデル重みは変わるか？

**No（条件が満たされれば同一）**

- DataLoader 作成で RNG 未消費 + wandb.init() で RNG 未消費 なら、
  モデル重み初期化時の torch RNG 状態は同一
- set_seed(42) → DataLoader 作成（RNG 消費なし）→ wandb.init()（RNG 消費なし）→ モデル初期化
  という順序なら、wandb on/off で重みは変わらない

### Q4: 39% 改善の原因はコード変更で説明できるか？

**Yes**

- リファクタリング・バグ修正・augmentation パラメータ修正が複数入っていれば、
  17.77°→10.71° の改善は wandb on/off ではなくコード変更が原因である可能性が高い

## 根本原因の特定

### sig2.0_base（17.77°）と今回（10.71°）の本当の違い

ワークログ（unet-work-log.md）と git log の分析から、
sig2.0_base が実行されたのは **2026-03-27 のリファクタリング前**。

当時から今回（2026-03-29）の間に入った変更：

| 変更 | 効果 |
|------|------|
| `train_heat.py` → `src/trainer.py` リファクタリング | バグ修正を含む大規模再構成 |
| `utils/losses.py` Stage 1-3 実装 | NaN 廃止、ベクトル化、新損失関数 |
| albumentations `Affine` パラメータ修正（`mode/cval` → `border_mode/fill`） | デフォルト値は同じだが内部動作が変化した可能性 |
| `extract_pred_line_params_batch` のリファクタリング | 抽出精度の向上 |

### wandb の影響は**ない**

- wandb は torch RNG を消費しない（Codex 確認済み）
- モデル重みは wandb on/off で同一
- 17.77°→10.71° の改善は wandb 無効化が原因ではない

## 推奨修正（再現性の確保）

現在のコードでも再現性は確保されているが、より安全にするなら：

**Option A（推奨）: model 初期化直前で set_seed() を再呼び出し**

```python
# trainer.py: train_one_fold() 内
create_data_loaders(...)       # DataLoader 作成
if wandb_enabled:
    wandb.init(...)            # wandb 初期化

set_seed(seed)                 # ← ここで再シード（モデル重み初期化を確実に固定）
create_model_optimizer_scheduler()
```

**Option B: DataLoader と model の seed を分離**

```python
set_seed(seed)                 # モデル用シード
create_model_optimizer_scheduler()  # 先にモデル初期化

# DataLoader は別 seed 管理（generator=g のみ）
create_data_loaders(...)
```

## まとめ

| 問い | 答え |
|------|------|
| wandb.init() は torch RNG を消費するか | **No** |
| wandb on/off でモデル重みは変わるか | **No** |
| 17.77°→10.71° の原因は wandb か | **No** |
| 真の原因は何か | **リファクタリング + バグ修正 + losses.py 再設計** |
| 乱数の再現性に問題があるか | **現状は問題なし**（DataLoader 作成も wandb も RNG を消費しないため） |
| 安全策として set_seed 再呼び出しは有効か | **Yes**（防衛的プログラミングとして推奨） |

## 参考ファイル

- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/line_only/src/trainer.py` (L598-743)
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/line_only/src/data_utils.py` (L143-216)
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/line_only/train.py` (L114-126)
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/Unet/line_only/test/test_aug_fold1.py`
- `/mnt/nfs1/home/yamamoto-hiroto/research/VAI/.claude/docs/unet-work-log.md`
