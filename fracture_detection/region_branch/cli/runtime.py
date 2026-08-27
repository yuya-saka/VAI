"""region_branch CLIの実行環境設定。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_local_temp_dir(base_dir: Path = Path("/tmp")) -> Path:
    """multiprocessing・TorchInductor cacheをNFS外のローカル領域へ配置する。"""
    local_temp_dir = base_dir / f"vai-region-branch-{os.getuid()}"
    local_temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(local_temp_dir)
    inductor_cache_dir = local_temp_dir / "torchinductor-cache"
    inductor_cache_dir.mkdir(mode=0o700, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache_dir)
    tempfile.tempdir = str(local_temp_dir)
    return local_temp_dir
