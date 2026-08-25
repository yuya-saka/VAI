# fracture_detection 進捗

最終更新: 2026-08-24

## 現在の主軸

`fracture_detection/baseline0/` の椎体単位骨折分類と、その教師モデルを使う疑似ラベル生成を主軸として扱う。

- 入力: 15面固定の2.5D CT 5ch + 椎体全体mask 1ch
- モデル: EfficientNetV2-S + BiLSTM
- 出力: 面ごとの骨折logitをmean-sigmoidでbag確率へ集約
- 評価: patient-grouped nested 5-fold
- データ: 品質除外済み13,432 bag
- 疑似ラベル: fold対応教師のGrad-CAMを4解剖領域へ集約し、CAM監査を通して生成

学習済み5-fold成果物は
`fracture_detection/baseline0/outputs/08_19/baseline0_shared_core/` に保持する。

疑似ラベルは2026-08-24に全量再生成済み。

- 出力: `fracture_detection/baseline0/outputs/08_19/pseudo_labels/`
- score: 40,296行、13,432一意bag、各bag 3 teacher
- temperature: 5 teacher × 4領域 = 20行
- provenance・checkpoint hash・出力hashは独立再計算で確認済み

## 現在の構成

```text
fracture_detection/
├── PROGRESS.md
└── baseline0/
    ├── cli/          # train / evaluate / attention / CAM audit / pseudo-label generation
    ├── config/       # schema / YAML
    ├── data/         # dataset / staging / split / sampling / constants
    ├── modeling/     # model / loss
    ├── training/     # trainer / optimizer / experiment management
    ├── evaluation/   # metrics
    ├── pseudo_labeling/ # Grad-CAM / CAM audit / score / report
    ├── resources/
    └── tests/
```

## 整理方針

- 失敗した MTL、Proposed、Type2 は再利用しない。
- 疑似ラベル生成とCAM監査は現行の主要機能として維持する。
- 新しい手法ごとにトップレベルdirectoryを増やさない。
- Baseline 0から派生する検討は、採用が決まるまで文書または小さなablationとして扱う。
- コードは責務別directoryへ置き、`baseline0/`直下へ実装fileを増やさない。
- 各directory内では、役割が1ファイルで収まる限り過剰に階層化しない。
- 過去の実装や判断が必要な場合はGit履歴を参照し、現行treeへarchiveを置かない。

## 次の作業

再生成済み疑似ラベルを使うpair構築・student学習へ進む。再生成が必要な場合は、復元済みCLIと同じfold・checkpoint provenanceを維持する。
