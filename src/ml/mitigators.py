"""
ml/mitigators.py
─────────────────
Algorithmic fairness mitigation implementations.

Currently implements:
  • KamiranCaldersReweighing — pre-processing sample-weight assignment
  • ThresholdOptimizer        — post-processing per-group decision thresholds

Both mitigators follow a fit / transform (or predict) interface so they
can be swapped transparently based on the mitigation_plan.json.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing: Reweighing (Kamiran & Calders, 2012)
# ─────────────────────────────────────────────────────────────────────────────

class KamiranCaldersReweighing:
    """
    Assigns instance weights so that the joint distribution of
    (sensitive_attribute, label) is independent under the weighted
    distribution.

    Reference: Kamiran & Calders (2012), "Data preprocessing techniques
    for classification without discrimination."
    """

    def __init__(self, sensitive_col: str, priv_value, pos_label=1):
        self.sensitive_col = sensitive_col
        self.priv_value    = priv_value
        self.pos_label     = pos_label
        self._weights: Dict[Tuple, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> "KamiranCaldersReweighing":
        """Compute per-(group, label) weights from the training distribution."""
        n = len(y)
        groups = [self.priv_value, self._unpriv_value(sensitive)]
        labels = list(np.unique(y))

        for g in groups:
            for lbl in labels:
                p_g   = np.mean(sensitive == g)
                p_lbl = np.mean(y == lbl)
                p_g_lbl = np.mean((sensitive == g) & (y == lbl))
                # Expected weight = P(g)*P(lbl) / P(g, lbl)
                weight = (p_g * p_lbl) / p_g_lbl if p_g_lbl > 0 else 1.0
                self._weights[(g, lbl)] = weight

        logger.info("Reweighing weights computed: %s", self._weights)
        return self

    def transform(self, sensitive: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a sample-weight array for every training instance."""
        weights = np.ones(len(y))
        for i, (g, lbl) in enumerate(zip(sensitive, y)):
            key = (g, lbl) if (g, lbl) in self._weights else (self.priv_value, lbl)
            weights[i] = self._weights.get(key, 1.0)
        return weights

    def _unpriv_value(self, sensitive: np.ndarray):
        unique = np.unique(sensitive)
        others = [v for v in unique if v != self.priv_value]
        return others[0] if others else self.priv_value


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: Per-group Threshold Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdOptimizer:
    """
    Applies different decision thresholds for each demographic group so
    that Equal Opportunity is equalised across groups at inference time.

    Implements a grid search over the threshold space to minimise |EOD|
    subject to a maximum accuracy drop constraint.
    """

    def __init__(
        self,
        sensitive_col:      str,
        priv_value,
        pos_label:          int = 1,
        max_accuracy_drop:  float = 0.02,
        n_thresholds:       int = 101,
    ):
        self.sensitive_col     = sensitive_col
        self.priv_value        = priv_value
        self.pos_label         = pos_label
        self.max_accuracy_drop = max_accuracy_drop
        self.n_thresholds      = n_thresholds
        self.thresholds_: Dict = {}

    def fit(
        self,
        y_true:    np.ndarray,
        y_prob:    np.ndarray,
        sensitive: np.ndarray,
        baseline_accuracy: float,
    ) -> "ThresholdOptimizer":
        """
        Grid-search per-group thresholds to equalise TPR (Equal Opportunity).

        y_prob : probability of positive class for each instance.
        """
        from src.ml.metrics import compute_metrics, accuracy

        grid = np.linspace(0.0, 1.0, self.n_thresholds)
        groups = np.unique(sensitive)

        # Start with threshold=0.5 for each group
        best_thresholds = {g: 0.5 for g in groups}
        best_eod_abs    = float("inf")

        for t_priv in grid:
            for t_unpriv in grid:
                candidate = {self.priv_value: t_priv}
                for g in groups:
                    if g != self.priv_value:
                        candidate[g] = t_unpriv

                y_pred = self._apply(y_prob, sensitive, candidate)
                acc    = accuracy(y_true, y_pred)
                if acc < baseline_accuracy - self.max_accuracy_drop:
                    continue

                metrics = compute_metrics(
                    y_true, y_pred, sensitive,
                    priv_value=self.priv_value,
                    pos_label=self.pos_label,
                )
                if abs(metrics.eod) < best_eod_abs:
                    best_eod_abs    = abs(metrics.eod)
                    best_thresholds = candidate

        self.thresholds_ = best_thresholds
        logger.info(
            "ThresholdOptimizer fitted: thresholds=%s  best|EOD|=%.4f",
            best_thresholds, best_eod_abs,
        )
        return self

    def predict(self, y_prob: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        if not self.thresholds_:
            raise RuntimeError("Call fit() before predict().")
        return self._apply(y_prob, sensitive, self.thresholds_)

    def _apply(
        self,
        y_prob:     np.ndarray,
        sensitive:  np.ndarray,
        thresholds: Dict,
    ) -> np.ndarray:
        y_pred = np.zeros(len(y_prob), dtype=int)
        for g, t in thresholds.items():
            mask = sensitive == g
            y_pred[mask] = (y_prob[mask] >= t).astype(int)
        # fallback for unseen groups
        unhandled = ~np.isin(sensitive, list(thresholds.keys()))
        if unhandled.any():
            y_pred[unhandled] = (y_prob[unhandled] >= 0.5).astype(int)
        return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_mitigator(method: str, **kwargs):
    """Return the correct mitigator instance by name."""
    registry = {
        "reweighing":         KamiranCaldersReweighing,
        "threshold_optimizer":ThresholdOptimizer,
    }
    if method not in registry:
        raise ValueError(
            f"Unknown mitigation method '{method}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[method](**kwargs)
