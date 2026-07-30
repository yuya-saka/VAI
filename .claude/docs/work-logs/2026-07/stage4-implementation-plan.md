# Stage4 実装計画書

作成日: 2026-07-29
このファイルの位置づけ: **実行可能なタスクリスト**。設計の根拠・実測値・Codexの分析は転記しない。
参照する上位文書:

- `.claude/docs/stage4-implementation-design.md` — 設計根拠・実測値・Codex決着事項（**必ず先に読む**）
- `.claude/docs/stage4-design-proposal.md` — 評価プロトコルの詳細
- `.claude/docs/codex/20260729-stage4-mixed-supervision-design.md` — Codex分析全文
- `data/rsna_data/stage4_folds.csv` — 構築済みの層別5-fold（`study_id,fold`）

**使い方**: 上から順に着手し、依存関係を守ること。各 Phase の末尾に完了条件（Definition of Done）を書いてある。
実装で迷ったら「なぜ」は上位文書を参照し、このファイルは更新しない（更新するのは上位文書側）。

---

## Phase 0: 前提の解消

### 0-1. 未commitの解消
`data_preprocessing/rsna_pipeline/process_dataset.py` / `process_study.py` の `--no-bbox` フラグ（07-29実装）を commit する。

### 0-2. 着手前に確定させる4点
`stage4-implementation-design.md` §11 の未確定事項のうち、実装の分岐点になる2つは**着手前にユーザーに確認する**。

1. **ディレクトリ構成**: `train_models/stage4/` を新設するか、`train_models/stage3/` に config フラグで拡張するか
   → 本計画は暫定的に **Stage3 拡張**（差分がデータ層＋損失1項＋サンプラのみのため）を前提に書く。
   新設と決まった場合は Phase 1〜5 のファイルパスを `stage3/` → `stage4/` に読み替える
2. **固定 epoch 数**: 早期停止を切るため事前に必要（Phase 5-1）。Weak-only を1 fold だけ動かして学習曲線を見て決め、以降変更しない

残り2点（level-only・Weak-only-size-matched を正式armに含めるか、C2除外macro-APを正式に報告するか）は
Phase 3 / Phase 7 着手前まで保留してよい。

**Definition of Done**: 未commit解消。ディレクトリ構成と固定epoch数の暫定値が決まっている。

---

## Phase 1: データ層

### 1-1. 領域ラベルの読み込み・結合

新規モジュール `train_models/stage3/src/region_labels.py`:

```python
def load_region_labels(csv_path: Path) -> dict[tuple[str, str], np.ndarray]:
    """(study_id, level) -> 4次元 int8 [R1,R2,R3,R4]。複数 run は OR 集約。"""

def region_supervision_of(
    label: int, key: tuple[str, str], region_labels: dict[...],
) -> Literal["strong", "weak", "negative"]:
    """label==0 -> negative / label==1 かつ key in region_labels -> strong / それ以外 -> weak"""
```

`collect_items`（`train_models/stage2/src/data_utils.py:112`、Stage2/3で共有）に2フィールド追加:

```python
item["region_label"] = region_labels.get(key, np.zeros(4, dtype=np.int8))  # weak/negativeは未使用
item["region_supervision"] = region_supervision_of(...)
```

### 1-2. fold読み込みの切り替え

新関数（`data_utils.py` か新規 `stage4_folds.py`）:

```python
def load_stage4_fold_map(csv_path: Path) -> dict[str, int]:
    """study_id -> fold。sha256 をログ出力し、既知hashと不一致なら例外。"""

def split_by_stage4_fold(
    items: list[dict], fold_map: dict[str, int], val_fold: int,
) -> tuple[list[dict], list[dict]]:
    """既存の split_test_holdout / split_items_cv は使わない（Stage4はholdoutなし・独自fold）。"""
```

起動時に `stage4_folds.csv` の sha256 を出力し、`stage4-implementation-design.md` 記載時点のhashと突き合わせる
（**split を後から作り直さない**ことをコードでも担保する）。

### 1-3. 陰性サンプラ

新規 `NegativeRegionSampler`（1:1、毎epoch再抽出、レベル完全一致、`seed=42+epoch`）:

```python
class NegativeRegionSampler:
    def sample(self, epoch: int) -> list[dict]:
        """
        - strong items のレベル分布と完全一致するよう negative から抽出
        - 可能な限り 1 patient から 1 bag。不足レベルのみ全患者一巡後に2 bag目許可
        - seed = 42 + epoch, 復元抽出しない
        - 選択した bag ID を manifest として保存（outputs/.../fold{f}/negative_manifest_epoch{e}.json）
        """
```

**禁止**: run全体で固定した陰性 subset を使い回すこと。

### 1-4. テスト

- `test_region_labels.py`: OR集約の正しさ、strong/weak/negative分類、268bag全数で教師が付くこと
- `test_negative_sampler.py`: 1:1比率、レベル分布のstrongとの一致、epoch間で抽出内容が変わること、manifest記録

**Definition of Done**: `collect_items` が `region_label` / `region_supervision` を返す。fold読み込みが `stage4_folds.csv` ベース。陰性サンプラがテスト込みで動く。

---

## Phase 2: augmentation の修正（最優先・ここを飛ばすと全実験が壊れる）

### 2-1. flip検出を region_label の swap に伝播

対象: `train_models/stage2/src/dataset.py`。

現状（確認済み）:
- `REGION_REMAP_HORIZONTAL = [0, 1, 3, 2, 4]`（17行）でマスクの ID 2↔3 は既に正しく swap される
- `_augment_volume`（158–194行）は `(images, regions)` しか返さず、flip適用の有無が呼び出し元に漏れない
- `RSNARegionDataset.__getitem__`（129行〜）は `region_label` を知らない（Stage4で新規に持ち込む）

変更方針:
1. `_augment_volume` の戻り値を `(images, regions, flip_applied: bool)` に拡張する
   （`replay_applied_horizontal_flip(augmented["replay"])` の結果を返すだけ）
2. Stage4用データセット（`RSNAStage3Dataset` と同様に `RSNARegionDataset` を継承する新クラス、
   例 `RSNAStage4Dataset`）で `__getitem__` を override し、`flip_applied` が True のとき

   ```python
   region_label = item["region_label"][[0, 2, 1, 3]]  # R2<->R3 (index 1<->2)
   ```

   を出力テンソルに反映する

**明確化（設計書の記述に対する実装上の補足）**:
上位文書は「region-valid フラグ R2↔R3 もswap」と書いているが、本実装では**別途のフラグは不要**と判断する。
`region_valid` はモデル側 `Stage3Model._region_features`（`train_models/stage3/src/model.py`）が
region_mask のピクセルから forward 時に都度計算しており、mask 自体は既存の
`remap_regions_after_horizontal_flip` で正しく反転済みなので自動的に整合する。
swap が必要なのは新規に持ち込む `region_label`（人手ラベルの4次元ベクトル）だけである。

R1・R4 は swap しない。vertical flip / transpose は対処不要（既存のまま）。

### 2-2. 必須 unit test（Codex 指定、3本とも実装。1本でも欠けたら Phase 3 に進まない）

1. **double flip** で画像・マスク・`region_label` が完全復元される
2. **R2のみ陽性の合成サンプル**が flip 後に **R3のみ陽性**になる
3. 学習サンプル100件で **mask ID と `region_label` の対応不一致が0件**
   （1件でも検出したらテストを失敗させ、学習を止める設計にする）

**Definition of Done**: 3 unit test 全て pass。`region_supervision="strong"` のサンプルを100件抽出し、
flip適用前後でマスクの左右領域と `region_label` の左右が一致することを目視でも1回確認する。

---

## Phase 3: level-only 対照（最初の実験・CPUのみ・数分）

目的: 合格ライン（no-skill 0.342 ではなく **level-only 0.458**）を先に確定させる。

- 学習不要。fold の学習側から **レベル別領域陽性率**を計算し、検証側に適用するだけ
- 新規スクリプト `train_models/stage3/scripts/stage4_level_only_baseline.py`
- 出力: 全268bag と C2除く231bag、両方の pooled OOF macro-AP

**Definition of Done**: 出力が `stage4-implementation-design.md` §7.4 の実測値（0.458 / 0.345 付近）と一致することを確認。
一致しなければ pooled OOF の実装自体にバグがあるので、Phase 6 に進む前にここで気づける。

---

## Phase 4: 損失

### 4-1. `L_region`

`train_models/stage3/utils/losses.py` に追加:

```python
def compute_region_pos_weight(
    strong_labels: np.ndarray, n_negative_sampled: int,
) -> Tensor:
    """w_r = min((2*N_A - P_r) / P_r, 8.0)。fold の学習側 strong から都度計算する。"""

def region_loss(
    region_logits: Tensor,      # q[r], [B,4]
    region_target: Tensor,      # [B,4]
    supervision_mask: Tensor,   # [B] bool, strong or サンプル済negative
    pos_weight: Tensor,         # [4]
) -> Tensor:
    """4領域で平均 -> 教師付きbagで平均。weak(mask=False)には教師を与えない。"""
```

既存 `stage3_loss`（`losses.py:118`）はそのまま使い、`L_total` の組み立て側で加算する。

```
L_total = L_vertebra + lambda_region(epoch) * L_region + lambda_neg * L_negative
```

`lambda_neg` の config デフォルトを 0.1 → **0.05** に変更。

### 4-2. λ スケジュール

```python
def lambda_region_schedule(epoch: int) -> float:
    return 0.25 + 0.75 * min(epoch / 4, 1.0)
```

trainer が epoch 番号を loss 計算箇所に渡せるよう軽微に修正する。

### 4-3. 診断ロギング

新規 `train_models/stage3/utils/diagnostics.py`。**100 optimizer step ごと**に記録:

- 共有 encoder 最終層での `cos(g_vertebra, g_region)`
- 勾配ノルム比 `λ_region · ‖g_region‖ / ‖g_vertebra‖`
- smooth-max 最大領域の bag 間分布（同一領域が陽性bagの何%で最大か）
- 1領域への pooling weight 集中度（`>0.95` の bag 割合）

閾値と警告条件は `stage4-implementation-design.md` §8.3 の表のとおり。wandb にログする。

**Definition of Done**: `region_loss` の unit test（strong/weak/negative の扱いが仕様どおりか）。
診断ログが1 epoch分、閾値判定込みで出力される。

---

## Phase 5: 学習ループの調整

### 5-1. 早期停止を無効化し epoch 数を固定

config に `fixed_epochs: N` を追加し、`early_stopping_patience` 依存の分岐を Stage4 では通さない。
N は Phase 0-2 で暫定決定した値。**Weak-only と Mixed で同じ N を使う**（confirmatory の条件6）。

### 5-2. バッチ構成

strong / weak / negative を一定比で混ぜる層別サンプラ。
`L_vertebra` は層別サンプリングで歪むので、既存 Stage3 の事前確率補正ロジック
（`(2N⁺·mean⁺ + N⁻·mean⁻)/(2N⁺+N⁻)`、07-28 Codex指摘）が Stage4 でも効いているか確認し、
無ければ移植する。

**Definition of Done**: 1 epoch を通しで動かし、`L_vertebra` / `L_region` / `L_negative` の3値と
診断ログが揃って出力される（実際の学習結果の良し悪しはまだ問わない）。

---

## Phase 6: Weak-only と Mixed-from-scratch（中心対比）

- 5-fold × 5-seed（`[42, 43, 44, 45, 46]`）
- **2 arm の差は `λ_region` の有無だけ**にする。fold・seed・初期値・sampler・陰性ID・augmentation乱数・
  更新回数は完全に同一（`stage4-implementation-design.md` §8.2 条件4-5）
- config例: `train_models/stage3/config/stage4_weak_only.yaml`、`stage4_mixed.yaml`
- **outer validation fold を checkpoint 選択や早期停止に使わない**（Phase 5-1 で担保済みのはず。ここで再確認）

**Definition of Done**: 5-fold × 5-seed = 25 run が Weak-only / Mixed それぞれ完走。
各 run の config diff が `λ_region` 関連パラメータ以外に無いことを diff で確認する。

---

## Phase 7: 評価

### 7-1. pooled OOF

- 268 bag全部の検証予測を5fold分連結
- 5 seed は **bag ごとに確率を平均してから** AP を1回計算（seedを5倍の観測にしない）

### 7-2. 患者単位 paired bootstrap（主仮説）

```
Δ = macro-AP(Mixed) − macro-AP(Weak-only)
判定: 10,000回、fold内で患者を復元抽出、95% CI下限 > 0 なら優越
```

### 7-3. 椎体 safety gate

```
ΔAUROC = AUROC(Mixed) − AUROC(Weak-only)
Pass: 患者 paired 95% CI下限 > -0.010
補助: AUPRC の Δ ≥ -0.020
```

Failした場合は独立椎体ヘッドの追加を検討するが、それは**このconfirmatory armの修正ではなく新しい実験**として扱う
（設計書§8.3）。

### 7-4. レポート

- 全268bag と C2除く231bag、両方の macro-AP を出す
- fold別APは診断としてのみ付記（判断には使わない）
- fold内percentile rankプールの感度解析も出し、raw pooledとの差が0.03以上なら
  「fold-scale sensitive」と明記する

**Definition of Done**: 主仮説・safety gate・両方のmacro-APが1本のレポートとして出力される。

---

## Phase 8: 追加arm（Phase 6-7 が確定してから着手）

- **Detail-only**: strong 214 + 陰性214。Mixedとdetail bag提示回数を完全一致させる
- **Weak-only-size-matched**: strong 214 bagを弱ラベル扱い + 陰性214（Detail-onlyとの三角測量用、Phase0-2で正式arm化を判断していれば実施）
- **Pretrain-to-Mixed**: fold毎にWeak-only事前学習 → mixed fine-tune

各armの詳細は `stage4-implementation-design.md` §6, §8.4 を参照。

---

## 全体の依存関係

```
Phase 0 ─┬─→ Phase 1 ─┬─→ Phase 2 ─→ Phase 3
         │            │              │
         │            └─→ Phase 4 ───┤
         │                           ↓
         └───────────────────→ Phase 5 ─→ Phase 6 ─→ Phase 7 ─→ Phase 8
```

Phase 2（augmentation修正）は Phase 3 より前に完了させること。**ここが唯一、静かに壊れる箇所**であるため、
テストなしで先に進んではならない。
