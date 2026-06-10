"""
metrics.py のユニットテスト。

sklearn との突合で P/R/F1・最適しきい値を検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from learning.utils.metrics import (
    compute_level_metrics,
    compute_oof_metrics,
    find_optimal_threshold,
)

Y_TRUE = np.array([1, 1, 0, 0, 1, 0])
Y_PROB = np.array([0.8, 0.6, 0.4, 0.3, 0.7, 0.9])


class TestFindOptimalThreshold:
    def test_matches_sklearn_argmax(self):
        """sklearn の precision_recall_curve から手動で求めた最適しきい値と一致する。"""
        precision, recall, thresholds = precision_recall_curve(Y_TRUE, Y_PROB)
        f1 = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
            0.0,
        )
        expected = float(thresholds[np.argmax(f1)])
        assert find_optimal_threshold(Y_TRUE, Y_PROB) == pytest.approx(expected, abs=1e-6)

    def test_all_positive_returns_05(self):
        assert find_optimal_threshold(np.ones(5), np.rand(5) if False else np.array([0.9, 0.8, 0.7, 0.6, 0.5])) == pytest.approx(0.5)

    def test_all_negative_returns_05(self):
        assert find_optimal_threshold(np.zeros(5), np.array([0.9, 0.8, 0.7, 0.6, 0.5])) == pytest.approx(0.5)


class TestComputeOofMetrics:
    def test_auroc_matches_sklearn(self):
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        expected = roc_auc_score(Y_TRUE, Y_PROB)
        assert result["auroc"] == pytest.approx(expected, abs=1e-6)

    def test_auprc_matches_sklearn(self):
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        expected = average_precision_score(Y_TRUE, Y_PROB)
        assert result["auprc"] == pytest.approx(expected, abs=1e-6)

    def test_at_05_matches_sklearn(self):
        """threshold=0.5 での P/R/F1 が sklearn と一致する。"""
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        y_pred = (Y_PROB >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            Y_TRUE, y_pred, average="binary", zero_division=0
        )
        assert result["at_05"]["precision"] == pytest.approx(float(p), abs=1e-6)
        assert result["at_05"]["recall"] == pytest.approx(float(r), abs=1e-6)
        assert result["at_05"]["f1"] == pytest.approx(float(f), abs=1e-6)

    def test_at_opt_threshold_maximizes_f1(self):
        """最適しきい値での F1 が 0.5 固定より高いか同等。"""
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        assert result["at_opt"]["f1"] >= result["at_05"]["f1"] - 1e-6

    def test_prevalence(self):
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        assert result["prevalence"] == pytest.approx(0.5, abs=1e-6)

    def test_bootstrap_ci_keys(self):
        groups = np.array(["s1", "s2", "s3", "s1", "s2", "s3"])
        result = compute_oof_metrics(Y_TRUE, Y_PROB, groups=groups)
        ci = result["bootstrap_ci"]
        assert "auroc_lo" in ci and "auroc_hi" in ci
        assert "auprc_lo" in ci and "auprc_hi" in ci

    def test_no_groups_no_ci(self):
        result = compute_oof_metrics(Y_TRUE, Y_PROB)
        assert "bootstrap_ci" not in result


class TestComputeLevelMetrics:
    def setup_method(self):
        # C1×2, C2×2, C3×2 の合計6サンプル
        self.y_true = np.array([1, 0, 1, 0, 1, 0])
        self.y_prob = np.array([0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
        self.levels = np.array(["C1", "C1", "C2", "C2", "C3", "C3"])

    def test_level_keys(self):
        result = compute_level_metrics(self.y_true, self.y_prob, self.levels)
        assert "C1" in result
        assert "C2" in result
        assert "C3" in result
        assert "C3-7" not in result
        assert "C4" not in result  # データなし

    def test_n_pos(self):
        result = compute_level_metrics(self.y_true, self.y_prob, self.levels)
        assert result["C1"]["n_pos"] == 1
        assert result["C2"]["n_pos"] == 1

    def test_prf_matches_sklearn(self):
        result = compute_level_metrics(self.y_true, self.y_prob, self.levels)
        # C1 のみ検証
        mask = self.levels == "C1"
        y_pred = (self.y_prob[mask] >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            self.y_true[mask], y_pred, average="binary", zero_division=0
        )
        assert result["C1"]["precision"] == pytest.approx(float(p), abs=1e-6)
        assert result["C1"]["recall"] == pytest.approx(float(r), abs=1e-6)
        assert result["C1"]["f1"] == pytest.approx(float(f), abs=1e-6)
