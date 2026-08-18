from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from fracture_detection.baseline0.cli.train import configure_local_temp_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = PROJECT_ROOT / "fracture_detection" / "baseline0" / "cli" / "train.py"


def test_train_script_direct_invocation_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Baseline 0のnested 5-fold学習" in result.stdout


def test_configure_local_temp_dir_overrides_nfs_temp_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", None)

    local_temp_dir = configure_local_temp_dir(tmp_path)

    assert local_temp_dir == tmp_path / f"vai-baseline0-{os.getuid()}"
    assert local_temp_dir.is_dir()
    assert tempfile.tempdir == str(local_temp_dir)
    assert all(
        os.environ[variable] == str(local_temp_dir)
        for variable in ("TMPDIR", "TEMP", "TMP")
    )
