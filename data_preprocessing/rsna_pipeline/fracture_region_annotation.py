"""骨折領域アノテーションツール（Marimo）

465件の (study, level) について、15プレーンを確認しながら
4領域のどれに骨折があるかをラベル付けする。

起動方法:
    uv run marimo edit data_preprocessing/rsna_pipeline/fracture_region_annotation.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _imports():
    import json
    import os
    from io import BytesIO
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib_fontja  # noqa: F401
    import numpy as np
    import pandas as pd
    from PIL import Image

    return BytesIO, Image, Path, json, mo, np, os, pd, plt


@app.cell
def _setup(Path, json, np, pd):
    """定数とデータ読み込み"""
    DATA_DIR = Path("data/rsna_data")
    FRACTURE_DATASET_DIR = DATA_DIR / "fracture_dataset"
    META_DIR = DATA_DIR / "processing_metadata"
    BBOX_CSV = DATA_DIR / "fracture_bbox_planes.csv"
    TRAIN_CSV = DATA_DIR / "train.csv"
    LABEL_CSV = DATA_DIR / "fracture_region_labels.csv"

    SAMPLING_PS = 0.4
    OUTPUT_SIZE = 224
    HALF_PX = (OUTPUT_SIZE - 1) / 2.0

    REGION_COLORS = {
        1: np.array([100, 149, 237]) / 255,   # cornflower blue
        2: np.array([50, 205, 50]) / 255,     # lime green
        3: np.array([220, 80, 80]) / 255,     # tomato
        4: np.array([255, 215, 0]) / 255,     # gold
    }
    REGION_NAMES = {1: "R1（椎体）", 2: "R2（右横突孔）", 3: "R3（左横突孔）", 4: "R4（後方要素）"}

    # 対象リスト構築
    bbox_df = pd.read_csv(BBOX_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    level_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    studies_with_bbox = set(bbox_df["study_id"].unique())
    levels_with_bbox = set(
        zip(bbox_df["study_id"], bbox_df["level"], strict=True)
    )

    targets = []
    for _, row in train_df.iterrows():
        sid = row["StudyInstanceUID"]
        if sid not in studies_with_bbox:
            continue
        for level in level_cols:
            if row[level] != 1:
                continue
            if (sid, level) not in levels_with_bbox:
                continue
            if not (FRACTURE_DATASET_DIR / sid / level / "region_4class.npy").exists():
                continue
            targets.append({"study_id": sid, "level": level})

    targets_df = pd.DataFrame(targets)

    # 既存ラベルの読み込み
    if LABEL_CSV.exists():
        existing = pd.read_csv(LABEL_CSV)
    else:
        existing = pd.DataFrame(columns=["study_id", "level", "region_1", "region_2", "region_3", "region_4"])

    # 未完了のインデックスを特定
    done_set = set(zip(existing["study_id"], existing["level"], strict=True))
    remaining_indices = [
        i for i, row in targets_df.iterrows()
        if (row["study_id"], row["level"]) not in done_set
    ]

    return (
        BBOX_CSV, DATA_DIR, FRACTURE_DATASET_DIR, HALF_PX, LABEL_CSV,
        META_DIR, OUTPUT_SIZE, REGION_COLORS, REGION_NAMES, SAMPLING_PS,
        bbox_df, done_set, existing, level_cols, remaining_indices,
        targets_df, train_df,
    )


@app.cell
def _navigator(mo, remaining_indices, targets_df, done_set):
    """ナビゲーション UI"""
    total = len(targets_df)
    done_count = len(done_set)

    progress_text = mo.md(f"**進捗: {done_count} / {total} 件完了**")

    if not remaining_indices:
        mo.stop(True, mo.md("## ✅ 全件アノテーション完了！"))

    # 未完了の最初のインデックスを初期値に
    init_idx = remaining_indices[0]

    idx_slider = mo.ui.slider(
        start=0,
        stop=total - 1,
        value=init_idx,
        label="件番号",
        full_width=True,
    )

    return done_count, idx_slider, init_idx, progress_text, total


@app.cell
def _show_info(mo, idx_slider, targets_df, done_set):
    """現在の件情報表示"""
    idx = idx_slider.value
    row = targets_df.iloc[idx]
    study_id = row["study_id"]
    level = row["level"]
    status = "✅ 完了" if (study_id, level) in done_set else "⏳ 未完了"

    info = mo.md(
        f"**[{idx + 1}/{len(targets_df)}]** "
        f"`{study_id.split('.')[-1]}` / **{level}** — {status}"
    )
    return idx, info, level, status, study_id


@app.cell
def _render_image(
    BytesIO, BBOX_CSV, FRACTURE_DATASET_DIR, HALF_PX, Image, META_DIR,
    OUTPUT_SIZE, REGION_COLORS, SAMPLING_PS, bbox_df, idx, json, level,
    mo, np, pd, plt, study_id, Path,
):
    """15プレーン画像を生成して表示"""
    import matplotlib.patches as mpatches

    level_dir = FRACTURE_DATASET_DIR / study_id / level
    ct = np.load(level_dir / "ct.npy")               # (15, 5, 224, 224)
    vmask = np.load(level_dir / "vertebra_mask.npy")  # (15, 224, 224)
    region = np.load(level_dir / "region_4class.npy") # (15, 224, 224)

    level_bboxes = bbox_df[
        (bbox_df["study_id"] == study_id) & (bbox_df["level"] == level)
    ]

    # plane → bboxリスト
    plane_bbox_list: dict = {i: [] for i in range(15)}
    for _, bbox in level_bboxes.iterrows():
        plane_bbox_list[int(bbox["plane_index"])].append(
            (
                float(bbox["row_min"]),
                float(bbox["col_min"]),
                float(bbox["row_max"]),
                float(bbox["col_max"]),
            )
        )

    # 3行×5列のグリッド
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.suptitle(f"{study_id.split('.')[-1]} / {level}", fontsize=13, fontweight="bold")

    for pi in range(15):
        ax = axes[pi // 5][pi % 5]
        ct_plane = ct[pi, 2]  # 中央チャンネル
        rgb = np.stack([ct_plane] * 3, axis=-1).astype(float) / 255.0

        # region overlay
        overlay = rgb.copy()
        for r, color in REGION_COLORS.items():
            mask = (region[pi] == r) & (vmask[pi] > 0)
            overlay[mask] = color
        blended = 0.6 * rgb + 0.4 * overlay
        ax.imshow(np.clip(blended, 0, 1))
        ax.set_title(f"p{pi}", fontsize=8, pad=2)
        ax.axis("off")

        # bbox
        for r0, c0, r1, c1 in plane_bbox_list[pi]:
            r0c, c0c = max(0, int(r0)), max(0, int(c0))
            r1c, c1c = min(OUTPUT_SIZE, int(r1)), min(OUTPUT_SIZE, int(c1))
            rect = mpatches.Rectangle(
                (c0c, r0c), c1c - c0c, r1c - r0c,
                linewidth=1.5, edgecolor="white", facecolor="none",
            )
            ax.add_patch(rect)

    # 凡例
    legend_patches = [
        mpatches.Patch(color=c, label=f"R{r}") for r, c in REGION_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)

    img_display = mo.image(pil_img, width="100%")
    return img_display, plane_bbox_list


@app.cell
def _label_input(mo, REGION_NAMES, existing, study_id, level):
    """ラベル選択 UI"""
    # 既存ラベルがあれば初期値に設定
    ex_row = existing[(existing["study_id"] == study_id) & (existing["level"] == level)]

    def _init(r):
        if len(ex_row) > 0:
            return bool(ex_row.iloc[0][f"region_{r}"])
        return False

    r1 = mo.ui.switch(label=REGION_NAMES[1], value=_init(1))
    r2 = mo.ui.switch(label=REGION_NAMES[2], value=_init(2))
    r3 = mo.ui.switch(label=REGION_NAMES[3], value=_init(3))
    r4 = mo.ui.switch(label=REGION_NAMES[4], value=_init(4))

    label_ui = mo.hstack([r1, r2, r3, r4], gap=2, justify="center")
    return label_ui, r1, r2, r3, r4


@app.cell
def _save_button(mo):
    save_btn = mo.ui.button(label="💾 保存", kind="success")
    return (save_btn,)


@app.cell
def _do_save(
    LABEL_CSV, existing, level, mo, pd, r1, r2, r3, r4,
    save_btn, status, study_id,
):
    """保存処理"""
    mo.stop(save_btn.value == 0)

    new_row = pd.DataFrame([{
        "study_id": study_id,
        "level": level,
        "region_1": int(r1.value),
        "region_2": int(r2.value),
        "region_3": int(r3.value),
        "region_4": int(r4.value),
    }])

    # 既存行を上書き or 追記
    updated = existing[~((existing["study_id"] == study_id) & (existing["level"] == level))]
    updated = pd.concat([updated, new_row], ignore_index=True)
    updated = updated.sort_values(["study_id", "level"]).reset_index(drop=True)
    updated.to_csv(LABEL_CSV, index=False)

    result = mo.callout(
        mo.md(f"✅ 保存しました: `{study_id.split('.')[-1]}` / {level} → R1={int(r1.value)} R2={int(r2.value)} R3={int(r3.value)} R4={int(r4.value)}"),
        kind="success",
    )
    return (result, updated)


@app.cell
def _layout(
    done_count, idx_slider, img_display, info, label_ui,
    mo, progress_text, save_btn, total,
):
    """レイアウト組み立て"""
    mo.vstack([
        mo.hstack([progress_text, mo.md(f"残り {total - done_count} 件")], justify="space-between"),
        idx_slider,
        info,
        img_display,
        mo.md("---"),
        mo.md("**骨折がある領域を選択してください:**"),
        label_ui,
        mo.hstack([save_btn], justify="center"),
    ])
