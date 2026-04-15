"""データセットとデータ拡張の定義（SDF 補助タスク版）"""

import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..src.model import VERTEBRA_TO_IDX


# -------------------------
# ポリライン前処理
# -------------------------
def preprocess_polyline(pts_xy: list, tau_dup: float = 1.0) -> list:
    """
    GT ポリラインの前処理：近接重複点の除去のみ

    引数:
        pts_xy: [[x, y], ...] の点リスト（最低2点）
        tau_dup: 近接重複点の除去閾値 (px)

    戻り値:
        前処理済みの点リスト（最低2点は保持）
    """
    if pts_xy is None or len(pts_xy) < 2:
        return pts_xy

    pts = np.array(pts_xy, dtype=np.float64)

    keep = [0]
    for i in range(1, len(pts) - 1):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= tau_dup:
            keep.append(i)
    keep.append(len(pts) - 1)
    pts = pts[keep]

    if len(pts) < 2:
        return pts_xy

    return pts.tolist()


# -------------------------
# 椎体グループ展開
# -------------------------
def vertebra_names_from_group(group: str):
    """椎体グループ名から椎体名リストを返す"""
    if group == "C1":
        return ["C1"]
    if group == "C2":
        return ["C2"]
    if group == "C3_C7":
        return ["C3", "C4", "C5", "C6", "C7"]
    if group == "ALL":
        return ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    raise ValueError(f"Unknown group: {group}")


# -------------------------
# サンプル有効性チェック
# -------------------------
def _is_sample_valid_png(sample_dir: Path, vertebra_group: str) -> bool:
    """
    PNG判定:
    - sample_dir/Cx/lines.json が存在
    - lines.json内で、4本線が揃っていて各lineが2点以上のsliceが1枚でもある
    - images/masks が存在している slice が1枚でもある
    """
    vertebra_names = vertebra_names_from_group(vertebra_group)

    for v_name in vertebra_names:
        v_dir = sample_dir / v_name
        if not v_dir.exists():
            continue

        lj = v_dir / "lines.json"
        if not lj.exists():
            continue

        try:
            lines_data = json.loads(lj.read_text())
        except Exception:
            continue

        for slice_idx_str, lines in lines_data.items():
            ok = True
            for k in ["line_1", "line_2", "line_3", "line_4"]:
                if k not in lines or lines[k] is None or len(lines[k]) < 2:
                    ok = False
                    break
            if not ok:
                continue

            slice_idx = int(slice_idx_str)
            ip = v_dir / "images" / f"slice_{slice_idx:03d}.png"
            mp = v_dir / "masks" / f"slice_{slice_idx:03d}.png"
            if ip.exists() and mp.exists():
                return True

    return False


# -------------------------
# SDF 生成関数
# -------------------------
def generate_truncated_sdf(
    polyline_points_list: list[list],
    vertebra_mask: np.ndarray,
    image_size: int,
    tau: float = 24.0,
) -> np.ndarray:
    """
    GT ポリラインと椎体マスクから 4ch Truncated SDF を生成する

    各境界 i に対して半平面符号付き距離を計算し、truncate する:
        s_i(x, y) = clip(d_i(x, y) / tau, -1, 1)
    d_i は PCA 法で推定した境界直線に対する符号付き垂直距離。
    符号は椎体マスクの重心が正になる方向に統一する（内側=正）。

    引数:
        polyline_points_list: 4本分のポリライン [[upper], [lower], [left], [right]]
                              各要素は [[x, y], ...] の点リスト
        vertebra_mask: 椎体シルエットマスク (H, W) float32 [0, 1]（0.5以上が椎体内部）
        image_size: 画像サイズ（正方形を仮定）
        tau: truncation 閾値（px 単位）

    戻り値:
        sdf: (4, H, W) float32、値域 [-1, 1]
    """
    H = W = image_size
    sdf = np.zeros((4, H, W), dtype=np.float32)

    # 座標グリッド（画像座標系: x=col, y=row）
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)  # 各 (H, W)

    # 椎体マスク重心（符号判定用: 内側=正 を保証するため）
    mask_bin = (vertebra_mask > 0.5).astype(np.float64)
    mask_sum = mask_bin.sum()
    if mask_sum > 0:
        mask_cy = float((mask_bin * Y).sum() / mask_sum)
        mask_cx = float((mask_bin * X).sum() / mask_sum)
    else:
        # マスクがない場合は画像中心を使う
        mask_cx = W / 2.0
        mask_cy = H / 2.0

    for ch_idx, pts_xy in enumerate(polyline_points_list):
        if pts_xy is None or len(pts_xy) < 2:
            # ポリラインなし: ゼロマップ
            sdf[ch_idx] = 0.0
            continue

        pts = np.array(pts_xy, dtype=np.float64)  # (N, 2) [x, y]

        # 直線の重心
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()

        # PCA で主軸方向を取得
        xc = pts - np.array([cx, cy])
        cov = (xc.T @ xc) / max(1, len(pts))

        if cov.max() < 1e-10:
            sdf[ch_idx] = 0.0
            continue

        evals, evecs = np.linalg.eigh(cov)
        # 主軸方向（最大固有値の固有ベクトル）
        d = evecs[:, np.argmax(evals)]  # [dx, dy]

        # 法線ベクトル（主軸を 90 度回転）
        nx = -d[1]
        ny = d[0]

        # 符号付き距離場: d_i(x, y) = nx*(x - cx) + ny*(y - cy)
        dist = nx * (X - cx) + ny * (Y - cy)

        # 椎体マスク重心での距離が正になるよう符号を修正（内側=正）
        centroid_val = nx * (mask_cx - cx) + ny * (mask_cy - cy)
        if centroid_val < 0:
            dist = -dist

        # Truncated SDF: clip(d / tau, -1, 1)
        sdf[ch_idx] = np.clip(dist / tau, -1.0, 1.0).astype(np.float32)

    return sdf


# -------------------------
# データ拡張（オンライン）
# -------------------------
def get_transforms(
    phase: str = "train", cfg_aug: dict | None = None
) -> A.Compose | A.ReplayCompose | None:
    """
    CT（画像）、mask（マスク）、SDF チャンネル（画像として扱う）に同じ幾何変換を適用
    """
    if phase != "train":
        return None
    cfg_aug = cfg_aug or {}

    ts = []

    # 幾何変換
    if cfg_aug.get("rotation", False):
        ts.append(A.Rotate(limit=float(cfg_aug.get("rotation_limit", 20)), p=0.5))

    if cfg_aug.get("scale", False):
        ts.append(
            A.Affine(
                scale=(
                    1.0 - float(cfg_aug.get("scale_limit", 0.1)),
                    1.0 + float(cfg_aug.get("scale_limit", 0.1)),
                ),
                translate_percent=0.0,
                rotate=0,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                fill_mask=0.0,
            )
        )

    # 左右反転
    if cfg_aug.get("horizontal_flip", False):
        ts.append(A.HorizontalFlip(p=float(cfg_aug.get("horizontal_flip_prob", 0.1))))

    # 上下反転
    if cfg_aug.get("vertical_flip", False):
        ts.append(A.VerticalFlip(p=float(cfg_aug.get("vertical_flip_prob", 0.1))))

    # 輝度コントラスト
    if cfg_aug.get("brightness_contrast", False):
        ts.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(cfg_aug.get("brightness_limit", 0.2)),
                contrast_limit=float(cfg_aug.get("contrast_limit", 0.2)),
                p=0.5,
            )
        )

    # ガウシアンノイズ
    if cfg_aug.get("gaussian_noise", False):
        var_lim = cfg_aug.get("noise_var_limit", [10, 50])
        ts.append(A.GaussNoise(var_limit=tuple(var_lim), p=0.3))

    # SDF チャンネルは bilinear 補間（image 扱い）
    additional_targets = {
        "mask": "mask",    # 椎体マスク: nearest 補間
        "gt_mask": "mask", # 領域ラベル: nearest 補間
        "sdf1": "image",   # SDF ch0 (upper): bilinear 補間
        "sdf2": "image",   # SDF ch1 (lower): bilinear 補間
        "sdf3": "image",   # SDF ch2 (left): bilinear 補間
        "sdf4": "image",   # SDF ch3 (right): bilinear 補間
    }

    # flip を使う場合は ReplayCompose でフリップの有無を記録
    use_flip = cfg_aug.get("horizontal_flip", False) or cfg_aug.get("vertical_flip", False)
    if use_flip:
        return A.ReplayCompose(ts, additional_targets=additional_targets)
    return A.Compose(ts, additional_targets=additional_targets)


# -------------------------
# データセット（PNG）: SDF 補助タスク版
# -------------------------
class PngSdfDataset(Dataset):
    """
    SDF 補助タスク用データセット

    dataset/
      sampleXX/
        C3/
          images/slice_000.png
          masks/slice_000.png
          lines.json

    SDF は aug 前に生成し、空間拡張時は bilinear 補間で変形する。
    水平反転時は left/right チャンネル（ch2, ch3）を入れ替える。
    垂直反転時は upper/lower チャンネル（ch0, ch1）を入れ替える。
    """

    def __init__(
        self,
        root_dir: Path,
        sample_names,
        group: str = "C3_C7",
        image_size: int = 224,
        sdf_tau: float = 24.0,
        transform=None,
        cfg_aug: dict | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.sample_names = list(sample_names)
        self.group = group
        self.image_size = int(image_size)
        self.sdf_tau = float(sdf_tau)
        self.vertebra_names = vertebra_names_from_group(group)

        self.transform = transform
        self.cfg_aug = cfg_aug or {}

        self._bad_slices = self._load_bad_slices()

        self.items = []
        self._build_index()
        print(f"[INFO] PngSdfDataset: {len(self.items)} slices")

    def _load_bad_slices(self) -> set:
        """bad_slices_all.json を読み込み、除外スライスのセットを返す"""
        bad_json = self.root_dir / "bad_slices_all.json"
        if not bad_json.exists():
            return set()
        try:
            data = json.loads(bad_json.read_text())
            entries = data if isinstance(data, list) else data.get("bad_slices", [])
            result = set()
            for entry in entries:
                slice_val = entry.get("slice_idx", entry.get("slice"))
                if slice_val is None:
                    continue
                result.add((entry["sample"], entry["vertebra"], int(slice_val)))
            if result:
                print(f"[INFO] bad_slices: {len(result)} スライスを除外")
            return result
        except Exception as e:
            print(f"[WARN] bad_slices_all.json の読み込みに失敗: {e}")
            return set()

    def _load_qc_excludes(self, vertebra_dir: Path) -> frozenset[int]:
        """qc_scores.json を読み込み、label=='exclude' の slice_idx 集合を返す"""
        qc_json = vertebra_dir / "qc_scores.json"
        if not qc_json.exists():
            return frozenset()
        try:
            data = json.loads(qc_json.read_text())
            excludes = {
                int(entry["slice_idx"])
                for entry in data
                if entry.get("label") == "exclude"
            }
            return frozenset(excludes)
        except Exception:
            return frozenset()

    def _build_index(self) -> None:
        """有効なスライスをインデックスに追加"""
        for s in self.sample_names:
            sd = self.root_dir / s
            if not sd.exists():
                continue

            for v in self.vertebra_names:
                vd = sd / v
                lj = vd / "lines.json"
                if not vd.exists() or not lj.exists():
                    continue

                try:
                    lines_data = json.loads(lj.read_text())
                except Exception:
                    continue

                qc_excludes = self._load_qc_excludes(vd)

                for slice_idx_str, lines in lines_data.items():
                    ok = True
                    for k in ["line_1", "line_2", "line_3", "line_4"]:
                        if k not in lines or lines[k] is None or len(lines[k]) < 2:
                            ok = False
                            break
                    if not ok:
                        continue

                    slice_idx = int(slice_idx_str)

                    if (s, v, slice_idx) in self._bad_slices:
                        continue

                    if slice_idx in qc_excludes:
                        continue

                    ip = vd / "images" / f"slice_{slice_idx:03d}.png"
                    mp = vd / "masks" / f"slice_{slice_idx:03d}.png"
                    if not ip.exists() or not mp.exists():
                        continue

                    gp = vd / "gt_masks" / f"slice_{slice_idx:03d}.png"
                    gt_mask_path = gp if gp.exists() else None

                    self.items.append(
                        {
                            "sample": s,
                            "vertebra": v,
                            "slice_idx": slice_idx,
                            "img_path": ip,
                            "mask_path": mp,
                            "gt_mask_path": gt_mask_path,
                            "lines": lines,
                        }
                    )

    def _load_gt_mask(self, gt_mask_path: Path | None) -> tuple[np.ndarray, bool]:
        """gt_masks を読み込み、欠損時はゼロマスクを返す"""
        empty_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if gt_mask_path is None:
            return empty_mask, False

        try:
            gt_mask = np.array(Image.open(gt_mask_path), dtype=np.uint8)
        except Exception as e:
            print(f"[WARN] gt_mask の読み込みに失敗: {gt_mask_path} ({e})")
            return empty_mask, False

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]

        if gt_mask.shape != (self.image_size, self.image_size):
            gt_mask = cv2.resize(
                gt_mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_mask = np.clip(gt_mask, 0, 4).astype(np.uint8)
        return gt_mask, True

    def _did_apply_flip(
        self, replay: dict[str, Any] | None, flip_class: str
    ) -> bool:
        """ReplayCompose の記録から指定クラスの反転有無を判定する

        引数:
            replay: albumentations の replay 辞書
            flip_class: 'HorizontalFlip' または 'VerticalFlip'
        """
        if replay is None:
            return False
        for tr in replay.get("transforms", []):
            if not tr.get("__class_fullname__", "").endswith(flip_class):
                continue
            if tr.get("applied", False):
                return True
        return False

    def _swap_gt_mask_left_right(self, gt_mask: np.ndarray) -> np.ndarray:
        """水平反転後に right/left ラベルを入れ替える"""
        if gt_mask.size == 0:
            return gt_mask
        label_map = np.array([0, 1, 3, 2, 4], dtype=np.uint8)
        return label_map[gt_mask]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        it = self.items[i]

        # 画像・マスク読み込み
        ct = np.array(Image.open(it["img_path"]).convert("L"), np.float32) / 255.0
        mk = np.array(Image.open(it["mask_path"]).convert("L"), np.float32) / 255.0
        gt_mask, has_gt_mask = self._load_gt_mask(it.get("gt_mask_path"))

        if ct.shape != (self.image_size, self.image_size):
            ct = cv2.resize(
                ct, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
        if mk.shape != (self.image_size, self.image_size):
            mk = cv2.resize(
                mk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
            )

        # SDF 生成（aug 前）: [upper=line_1, lower=line_2, left=line_3, right=line_4]
        polylines = [
            preprocess_polyline(it["lines"][k])
            for k in ["line_1", "line_2", "line_3", "line_4"]
        ]
        sdf = generate_truncated_sdf(
            polylines, mk, self.image_size, tau=self.sdf_tau
        )  # (4, H, W)

        # オンライン拡張（訓練時のみ）
        if self.transform is not None:
            out = self.transform(
                image=ct,
                mask=mk,
                gt_mask=gt_mask,
                sdf1=sdf[0],
                sdf2=sdf[1],
                sdf3=sdf[2],
                sdf4=sdf[3],
            )
            ct = out["image"]
            mk = out["mask"]
            gt_mask = out["gt_mask"].astype(np.uint8)
            sdf = np.stack(
                [out["sdf1"], out["sdf2"], out["sdf3"], out["sdf4"]], axis=0
            ).astype(np.float32)

            # flip 時のチャンネル入れ替え
            if isinstance(self.transform, A.ReplayCompose):
                replay = out.get("replay")
                # 水平反転: left(ch2) <-> right(ch3)
                if self.cfg_aug.get("horizontal_flip", False):
                    if self._did_apply_flip(replay, "HorizontalFlip"):
                        sdf = sdf[[0, 1, 3, 2]]
                        gt_mask = self._swap_gt_mask_left_right(gt_mask)
                # 垂直反転: upper(ch0) <-> lower(ch1)
                if self.cfg_aug.get("vertical_flip", False):
                    if self._did_apply_flip(replay, "VerticalFlip"):
                        sdf = sdf[[1, 0, 2, 3]]

        ct = np.clip(ct, 0.0, 1.0).astype(np.float32)
        mk = np.clip(mk, 0.0, 1.0).astype(np.float32)
        sdf = np.clip(sdf, -1.0, 1.0).astype(np.float32)
        gt_mask = np.clip(gt_mask, 0, 4).astype(np.uint8)

        # 椎体インデックス
        vertebra_idx = VERTEBRA_TO_IDX.get(it["vertebra"], 0)

        x = np.stack([ct, mk], 0).astype(np.float32)  # (2, H, W)

        return {
            "image": torch.from_numpy(x),
            "sdf_field": torch.from_numpy(sdf),
            "gt_region_mask": torch.from_numpy(gt_mask.astype(np.int64)),
            "has_gt_region_mask": torch.tensor(has_gt_mask, dtype=torch.bool),
            "vertebra_idx": torch.tensor(vertebra_idx, dtype=torch.long),
            "sample": it["sample"],
            "vertebra": it["vertebra"],
            "slice_idx": it["slice_idx"],
        }


# 後方互換エイリアス
PngLineDataset = PngSdfDataset
