"""Tests for src/ml/metrics.py"""
import numpy as np
import pytest
from src.ml.metrics import compute_metrics, accuracy, FairnessMetrics


class TestComputeMetrics:
    def test_perfect_fairness(self):
        """When both groups have identical TPR, EOD should be 0."""
        y_true    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred, sensitive, priv_value=1)
        assert abs(m.eod) < 1e-9
        assert abs(m.spd) < 1e-9

    def test_eod_known_value(self):
        """Manually computed EOD should match.
        priv group  (sensitive=1): y_true=[1,1,0,0], y_pred=[1,0,0,0] → TPR=0.5
        unpriv group(sensitive=0): y_true=[1,1,0,0], y_pred=[1,1,0,0] → TPR=1.0
        EOD = unpriv_TPR - priv_TPR = 1.0 - 0.5 = 0.5
        """
        y_true    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred    = np.array([1, 0, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred, sensitive, priv_value=1)
        assert abs(m.eod - 0.5) < 1e-6

    def test_returns_fairness_metrics_dataclass(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([1, 1, 0, 0])
        m = compute_metrics(y, y, s)
        assert isinstance(m, FairnessMetrics)
        assert hasattr(m, "eod")
        assert hasattr(m, "aod")
        assert hasattr(m, "dir")
        assert hasattr(m, "spd")

    def test_passes_eod_threshold(self):
        """Perfect predictions → EOD=0 → should pass threshold."""
        y = np.ones(8, dtype=int)
        s = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y, y, s)
        assert m.passes_eod_threshold()

    def test_fails_eod_threshold(self):
        """EOD of 0.5 should fail the default ±0.05 threshold."""
        y_true    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred    = np.array([1, 0, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred, sensitive, priv_value=1)
        assert not m.passes_eod_threshold()

    def test_passes_four_fifths_rule(self):
        """Equal predictions for both groups → DIR=1.0 → passes."""
        y_true    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred, sensitive, priv_value=1)
        assert m.passes_four_fifths_rule()

    def test_accuracy_metric(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        assert accuracy(y_true, y_pred) == 0.75

    def test_str_representation(self):
        m = FairnessMetrics(eod=0.03, aod=0.02, dir=0.95, spd=-0.01)
        s = str(m)
        assert "EOD" in s
        assert "DIR" in s


class TestAccuracy:
    def test_all_correct(self):
        y = np.array([0, 1, 0, 1])
        assert accuracy(y, y) == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy(y_true, y_pred) == 0.0

    def test_partial(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        assert accuracy(y_true, y_pred) == 0.75


class TestFairnessMetricsDataclass:
    def test_to_dict(self):
        m = FairnessMetrics(eod=0.01, aod=0.02, dir=0.90, spd=-0.01)
        d = m.to_dict()
        assert d["eod"] == 0.01
        assert d["dir"] == 0.90

    def test_passes_aod_threshold(self):
        m = FairnessMetrics(eod=0.01, aod=0.02, dir=0.90, spd=-0.01)
        assert m.passes_aod_threshold()

    def test_fails_aod_threshold(self):
        m = FairnessMetrics(eod=0.01, aod=0.10, dir=0.90, spd=-0.01)
        assert not m.passes_aod_threshold()
