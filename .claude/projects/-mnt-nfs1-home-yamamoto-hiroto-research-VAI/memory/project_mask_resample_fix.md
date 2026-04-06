---
name: CT-mask alignment bug fixed in convert_to_png.py
description: CTとmaskのaffineが異なる問題を修正し、dataset/を再生成した
type: project
---

`Unet/preprocessing/convert_to_png.py` に CT-mask アライメントバグが存在した。

SlicerでCTをcropして保存したが、maskは保存し直していなかったため、affineが最大25.6mmずれていた。
修正前は同じslice indexで切っていたため、PNGのCTとmaskがずれていた。

**Fix:** `nibabel.processing.resample_from_to(mask_nii, ct_nii, order=0)` でmaskをCT空間にリサンプリングする処理を追加（L309-311）。affineかshapeが異なる場合のみ実行。

**Why:** CT・mask affineの不一致を検出・補正しないまま同一slice indexで切るとずれる。

**How to apply:** dataset/を使う際はこの修正済みスクリプトで生成されたものを使う。再生成日: 2026-04-06。成功椎骨135件、総スライス910枚。
