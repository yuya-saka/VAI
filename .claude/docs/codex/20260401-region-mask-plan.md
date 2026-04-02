# Codex Analysis: Region Mask Creation from Line Annotations
Date: 2026-04-01

## Question

We have a medical image segmentation problem. We have 4 polyline annotations per cervical vertebra CT slice (224x224 pixels) that define boundaries between anatomical regions. We need to convert these 4 polylines into a 5-channel segmentation mask (4 regions + background).

Data:
- 4 polylines (line_1 to line_4) per slice, each with multiple points
- Binary vertebra mask (which pixels are inside the vertebra)
- Target: assign each vertebra pixel to one of 4 anatomical regions

The 4 lines roughly divide the vertebra cross-section like this:
- line_1 (right-center vertical boundary): separates center body from right transverse foramen
- line_2 (right horizontal boundary): separates right transverse foramen from posterior elements
- line_3 (left-center vertical boundary): separates center body from left transverse foramen
- line_4 (left horizontal boundary): separates left transverse foramen from posterior elements

## Codex Response

**1. 最も堅牢なアルゴリズム**

4本を「4本の無限直線」として扱うより、**左右それぞれの T-junction を持つ 4 本の separator ray** として扱うのが一番安定です。

手順はこれです。

1. 各 polyline を前処理する。
   近接重複点除去、画像外クリップ、`len<2` は invalid。

2. 各 polyline に対して直線近似を作る。
   基本は **TLS/PCA**。強く曲がる場合は全点 PCA ではなく、junction 側の数点で局所接線を取る。

3. 右ペア `(line_1, line_2)`、左ペア `(line_3, line_4)` の junction を強制する。
   交点 `J_R`, `J_L` を求め、実アノテーションが少し離れていても同一点にスナップする。

4. 各 line は junction から**外側 1 方向にだけ延長**する。
   ここが重要です。
   `line_1/line_3` を画像全体に無限延長すると posterior を余計に切るので、**ray** として延長します。
   `line_1`: `J_R` から body/foramen を分ける方向へ
   `line_2`: `J_R` から foramen/posterior を分ける方向へ
   `line_3`: `J_L` から body/foramen を分ける方向へ
   `line_4`: `J_L` から foramen/posterior を分ける方向へ

5. 4 本の ray を vertebra mask 内で barrier として rasterize する。
   `cv2.line` で 1-2 px、必要なら 1 px dilation/closing。

6. `vertebra_mask - barrier` を連結成分分解する。
   正常なら 4 領域に分かれる。

7. 4 成分を anatomical label に割り当てる。
   これは centroid と line-side 判定で決める。

8. 最後に 5ch one-hot にする。
   例: `[background, body, right_foramen, left_foramen, posterior]`

**2. 線がきれいに交わらず gap が残る場合**

gap は half-plane の理屈で埋めるより、**先に topology を修復**した方が安全です。

推奨順はこれです。

1. `line_1-line_2`, `line_3-line_4` を同一 junction にスナップする。
2. junction 近傍の最近 endpoint 同士を短い bridge で結ぶ。
3. barrier を 1 px dilate / close する。
4. それでも 4 成分にならなければ、marker-based watershed か geodesic fill を使う。
5. posterior は half-plane で直接決めず、**残差領域**として取る。

要点は、**曖昧 pixel を separator に最近い方で決めない**ことです。
最近傍だと junction gap で body/posterior が簡単に混ざります。

**3. 正しい topology**

side 名を固定して扱うのが安全です。

- `body_side(line_1)`: `line_1` の body 側
- `foramen_side(line_1)`: `line_1` の右 transverse foramen 側
- `body_side(line_3)`: `line_3` の body 側
- `foramen_side(line_3)`: `line_3` の左 transverse foramen 側
- `posterior_side(line_2)`: `line_2` の posterior 側
- `nonposterior_side(line_2)`: `line_2` の foramen/body 側
- `posterior_side(line_4)`: `line_4` の posterior 側
- `nonposterior_side(line_4)`: `line_4` の foramen/body 側

そのうえで:

- `body = V ∩ body_side(line_1) ∩ body_side(line_3) ∩ nonposterior_side(line_2) ∩ nonposterior_side(line_4)`
- `right_foramen = V ∩ foramen_side(line_1) ∩ nonposterior_side(line_2)`
- `left_foramen = V ∩ foramen_side(line_3) ∩ nonposterior_side(line_4)`
- `posterior = V \ (body ∪ right_foramen ∪ left_foramen)`

ここで `V = vertebra_mask` です。
**posterior を residual にする**のが一番壊れにくいです。

**4. コード outline**

```python
from dataclasses import dataclass
import numpy as np
import cv2

EPS = 1e-6

@dataclass
class FittedLine:
    point: np.ndarray      # (2,)
    direction: np.ndarray  # unit vector (2,)

def preprocess_polyline(points: np.ndarray, tau_dup: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts) - 1):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= tau_dup:
            keep.append(i)
    keep.append(len(pts) - 1)
    return pts[keep]

def fit_tls_line(points: np.ndarray) -> FittedLine:
    pts = np.asarray(points, dtype=np.float32)
    c = pts.mean(axis=0)
    x = pts - c
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    d = vt[0]
    d = d / (np.linalg.norm(d) + EPS)
    return FittedLine(point=c, direction=d)

def line_intersection(a: FittedLine, b: FittedLine) -> np.ndarray | None:
    A = np.stack([a.direction, -b.direction], axis=1)
    if abs(np.linalg.det(A)) < 1e-4:
        return None
    t = np.linalg.solve(A, b.point - a.point)[0]
    return a.point + t * a.direction

def nearest_endpoint(points: np.ndarray, junction: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points - junction[None, :], axis=1)
    return points[np.argmax(d)]  # junction から遠い端を outward 側とみなす

def extend_ray_to_mask(
    junction: np.ndarray,
    outward_point: np.ndarray,
    vertebra_mask: np.ndarray,
    max_len: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    u = outward_point - junction
    u = u / (np.linalg.norm(u) + EPS)

    last_inside = junction.copy()
    H, W = vertebra_mask.shape
    for t in np.linspace(0, max_len, 2048):
        p = junction + t * u
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or x >= W or y < 0 or y >= H:
            break
        if vertebra_mask[y, x] > 0:
            last_inside = np.array([x, y], dtype=np.float32)
    return junction.astype(np.float32), last_inside

def draw_barrier(mask_shape: tuple[int, int], segments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    barrier = np.zeros(mask_shape, dtype=np.uint8)
    for p0, p1 in segments:
        cv2.line(
            barrier,
            tuple(np.round(p0).astype(int)),
            tuple(np.round(p1).astype(int)),
            color=1,
            thickness=2,
        )
    kernel = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel)
    return barrier

def split_regions(vertebra_mask: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    free = ((vertebra_mask > 0) & (barrier == 0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(free)
    return labels

def signed_side(points_xy: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    v = p1 - p0
    w = points_xy - p0[None, :]
    return v[0] * w[:, 1] - v[1] * w[:, 0]

def generate_region_mask(
    line_1, line_2, line_3, line_4, vertebra_mask: np.ndarray
) -> np.ndarray:
    H, W = vertebra_mask.shape
    pts1 = preprocess_polyline(np.asarray(line_1))
    pts2 = preprocess_polyline(np.asarray(line_2))
    pts3 = preprocess_polyline(np.asarray(line_3))
    pts4 = preprocess_polyline(np.asarray(line_4))

    L1 = fit_tls_line(pts1)
    L2 = fit_tls_line(pts2)
    L3 = fit_tls_line(pts3)
    L4 = fit_tls_line(pts4)

    JR = line_intersection(L1, L2)
    JL = line_intersection(L3, L4)

    if JR is None:
        JR = 0.5 * (pts1[-1] + pts2[0])  # fallback
    if JL is None:
        JL = 0.5 * (pts3[-1] + pts4[0])  # fallback

    s1 = extend_ray_to_mask(JR, nearest_endpoint(pts1, JR), vertebra_mask)
    s2 = extend_ray_to_mask(JR, nearest_endpoint(pts2, JR), vertebra_mask)
    s3 = extend_ray_to_mask(JL, nearest_endpoint(pts3, JL), vertebra_mask)
    s4 = extend_ray_to_mask(JL, nearest_endpoint(pts4, JL), vertebra_mask)

    barrier = draw_barrier((H, W), [s1, s2, s3, s4])
    labels = split_regions(vertebra_mask, barrier)

    # labels -> anatomical labels:
    # 1. component centroid を出す
    # 2. line-side 条件で body / right_foramen / left_foramen / posterior に対応付け
    # 3. posterior は residual にする

    seg = np.zeros((5, H, W), dtype=np.uint8)
    seg[0] = (vertebra_mask == 0).astype(np.uint8)  # background
    # seg[1] = body
    # seg[2] = right_foramen
    # seg[3] = left_foramen
    # seg[4] = posterior
    return seg
```

**5. edge case と fallback**

- `line_i` が 2 点未満
  その slice は invalid 扱い。無理に埋めない。

- `line_1-line_2` または `line_3-line_4` がほぼ平行
  交点ではなく最近 endpoint の midpoint を junction にする。

- polyline が強く曲がる
  全点 PCA ではなく junction 側 3-5 点で局所接線を使う。

- junction が vertebra mask の外に飛ぶ
  junction を mask 内最近点へ clamp する。

- barrier 後の connected component 数が `!= 4`
  まず bridge と closing を強める。
  それでもだめなら markers を置いて watershed。

- 小さなゴミ成分が出る
  面積閾値で最近接大成分へ merge。

- 左右 line 名が入れ替わった annotation
  centroid の左右位置と expected anatomy で検出して flag を立てる。

**6. 検証方法**

最低限、毎 slice でこれを確認します。

- one-hot 性
  全 pixel で `sum(channels) == 1`

- vertebra 内完全被覆
  `vertebra_mask == 1` の pixel は必ず 4 region のどれか

- 背景整合
  `vertebra_mask == 0` の pixel は background のみ

- 4 region の存在
  各 region 面積 `> 0`。ゼロなら fail

- topology 整合
  `right_foramen` は `line_1` の foramen 側かつ `line_2` の non-posterior 側
  `left_foramen` は `line_3` の foramen 側かつ `line_4` の non-posterior 側
  `body` は両 vertical の body 側
  `posterior` は residual

- 可視化 QA
  CT 上に 4 色 region overlay + 4 separator を重ねてランダム 50-100 slice を目視

- dataset 統計
  vertebra level ごとの各 region 面積比を出し、極端値を flag

**結論**:「4 本の無限直線で直接塗る」のではなく、「2 つの T-junction に topology を強制して barrier 化し、最後は component/marker ベースで埋める」のが最も壊れにくいです。
