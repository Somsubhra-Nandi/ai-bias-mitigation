"""Tests for src/ml/mitigators.py"""
import numpy as np
import pytest
from src.ml.mitigators import KamiranCaldersReweighing, ThresholdOptimizer, get_mitigator


class TestKamiranCaldersReweighing:
    def setup_method(self):
        np.random.seed(42)
        n = 200
        self.sensitive = np.random.choice([0, 1], n, p=[0.4, 0.6])
        self.y         = np.random.choice([0, 1], n)
        self.X         = np.random.randn(n, 5)
        self.reweigher = KamiranCaldersReweighing(sensitive_col="Gender", priv_value=1)

    def test_fit_returns_self(self):
        result = self.reweigher.fit(self.X, self.y, self.sensitive)
        assert result is self.reweigher

    def test_weights_positive(self):
        self.reweigher.fit(self.X, self.y, self.sensitive)
        weights = self.reweigher.transform(self.sensitive, self.y)
        assert np.all(weights > 0)

    def test_weights_length_matches_input(self):
        self.reweigher.fit(self.X, self.y, self.sensitive)
        weights = self.reweigher.transform(self.sensitive, self.y)
        assert len(weights) == len(self.y)

    def test_weights_are_floats(self):
        self.reweigher.fit(self.X, self.y, self.sensitive)
        weights = self.reweigher.transform(self.sensitive, self.y)
        assert weights.dtype in [np.float32, np.float64, float]

    def test_weights_computed_after_fit(self):
        """_weights dict should be populated after fit."""
        self.reweigher.fit(self.X, self.y, self.sensitive)
        assert len(self.reweigher._weights) > 0

    def test_uniform_weights_when_groups_and_labels_independent(self):
        """When sensitive attr and label are independent, all weights should be equal."""
        n = 200
        # Alternate sensitive: [1,1,0,0,1,1,0,0,...] — 50% each group
        # Alternate y:         [1,0,1,0,1,0,1,0,...] — 50% each label, same in each group
        sensitive = np.array([1, 1, 0, 0] * (n // 4))
        y         = np.array([1, 0, 1, 0] * (n // 4))
        X         = np.zeros((n, 2))
        rw = KamiranCaldersReweighing(sensitive_col="Gender", priv_value=1)
        rw.fit(X, y, sensitive)
        weights = rw.transform(sensitive, y)
        # Reweigher guarantees: positive weights of correct length
        assert np.all(weights > 0)
        assert len(weights) == n
        # All weights should be equal when distribution is already fair
        assert np.allclose(weights, weights[0], atol=1e-6)

    def test_small_dataset(self):
        """Should work on a minimal dataset without crashing."""
        sensitive = np.array([1, 1, 0, 0])
        y         = np.array([1, 0, 1, 0])
        X         = np.zeros((4, 2))
        rw = KamiranCaldersReweighing(sensitive_col="Gender", priv_value=1)
        rw.fit(X, y, sensitive)
        weights = rw.transform(sensitive, y)
        assert len(weights) == 4
        assert np.all(weights > 0)


class TestThresholdOptimizer:
    def setup_method(self):
        np.random.seed(42)
        n = 100
        self.sensitive = np.random.choice([0, 1], n, p=[0.4, 0.6])
        self.y_true    = np.random.choice([0, 1], n, p=[0.3, 0.7])
        self.y_prob    = np.clip(
            self.y_true * 0.6 + np.random.uniform(0, 0.4, n), 0, 1
        )
        self.optimizer = ThresholdOptimizer(
            sensitive_col="Gender",
            priv_value=1,
            n_thresholds=11,   # small grid for fast tests
        )

    def test_fit_returns_self(self):
        result = self.optimizer.fit(
            self.y_true, self.y_prob, self.sensitive,
            baseline_accuracy=0.7
        )
        assert result is self.optimizer

    def test_thresholds_populated_after_fit(self):
        self.optimizer.fit(
            self.y_true, self.y_prob, self.sensitive,
            baseline_accuracy=0.7
        )
        assert len(self.optimizer.thresholds_) > 0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            self.optimizer.predict(self.y_prob, self.sensitive)

    def test_predict_returns_binary(self):
        self.optimizer.fit(
            self.y_true, self.y_prob, self.sensitive,
            baseline_accuracy=0.7
        )
        preds = self.optimizer.predict(self.y_prob, self.sensitive)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_length_matches_input(self):
        self.optimizer.fit(
            self.y_true, self.y_prob, self.sensitive,
            baseline_accuracy=0.7
        )
        preds = self.optimizer.predict(self.y_prob, self.sensitive)
        assert len(preds) == len(self.y_true)


class TestGetMitigator:
    def test_returns_reweighing(self):
        m = get_mitigator("reweighing", sensitive_col="Gender", priv_value=1)
        assert isinstance(m, KamiranCaldersReweighing)

    def test_returns_threshold_optimizer(self):
        m = get_mitigator("threshold_optimizer", sensitive_col="Gender", priv_value=1)
        assert isinstance(m, ThresholdOptimizer)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown mitigation method"):
            get_mitigator("nonexistent_method", sensitive_col="Gender", priv_value=1)