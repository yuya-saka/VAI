"""RSNA データセットに対する 4 領域マスク生成パイプライン。

line_only モデルの推論結果を用いて椎体を 4 領域（椎体・右椎間孔・左椎間孔・後方要素）に
分割するスクリプト群を格納する。
"""

import sys
from pathlib import Path

_unet = Path(__file__).resolve().parents[2]  # Unet/
_root = _unet.parent  # VAI/
for _p in (str(_unet), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
