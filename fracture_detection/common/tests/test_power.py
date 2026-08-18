from __future__ import annotations

import pytest

from fracture_detection.common.power import paired_normal_mde


def test_paired_normal_mde_decreases_with_correlation() -> None:
    low_correlation = paired_normal_mde(0.03, correlation=0.5)
    high_correlation = paired_normal_mde(0.03, correlation=0.9)

    assert high_correlation < low_correlation
    assert paired_normal_mde(0.0, correlation=0.9) == pytest.approx(0.0)
