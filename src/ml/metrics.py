"""
ml/metrics.py
──────────────
Fairness metric computation: EOD, AOD, DIR, SPD.

All metrics follow the AIF360 / Fairlearn convention.
Every function accepts plain numpy arrays so the module has no
dependency on any specific ML framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    """Container for the four standard group fairness metrics."""
    eod: float   # Equal Opportunity Difference
    aod: float   # Average Odds Difference
    dir: float   # Disparate Impact Ratio
    spd: float   # Statistical Parity Difference

    def to_dict(self) -> dict:
        return asdict(self)

    def passes_eod_threshold(self, low: float = -0.05, high: float = 0.05) -> bool:
        return low <= self.eod <= high

    def passes_aod_threshold(self, low: float = -0.07, high: float = 0.07) -> bool:
        return low <= self.aod <= high

    def passes_four_fifths_rule(self, threshold: float = 0.80) -> bool:
        """DIR ≥ 0.80 is the EEOC four-fifths rule."""
        return self.dir >= threshold

    def __str__(self) -> str:
        return (
            f"EOD={self.eod:+.4f}  AOD={self.aod:+.4f}  "
            f"DIR={self.dir:.4f}  SPD={self.spd:+.4f}"
        )


def _safe_rate(positives: int, total: int) -> float:
    """True positive or positive prediction rate, guarded against div/0."""
    return positives / total if total > 0 else 0.0


def compute_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    sensitive:  np.ndarray,
    priv_value: int | str = 1,
    pos_label:  int | str = 1,
) -> FairnessMetrics:
    """
    Compute group fairness metrics for a binary classifier.

    Parameters
    ----------
    y_true :
        Ground-truth labels (0/1 or matching pos_label).
    y_pred :
        Model predictions (0/1).
    sensitive :
        Sensitive attribute column (same length as y_true).
    priv_value :
        Value in `sensitive` that denotes the privileged group.
    pos_label :
        The positive class label.

    Returns
    -------
    FairnessMetrics
    """
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    priv_mask  = sensitive == priv_value
    unpriv_mask = ~priv_mask

    # Helper masks
    y_pos  = y_true  == pos_label
    yh_pos = y_pred  == pos_label

    # ── True Positive Rates ───────────────────────────────────────────────────
    priv_tp_total   = np.sum(y_pos  & priv_mask)
    unpriv_tp_total = np.sum(y_pos  & unpriv_mask)
    priv_tpr  = _safe_rate(int(np.sum(yh_pos & y_pos & priv_mask)),   priv_tp_total)
    unpriv_tpr= _safe_rate(int(np.sum(yh_pos & y_pos & unpriv_mask)), unpriv_tp_total)

    # ── False Positive Rates ──────────────────────────────────────────────────
    y_neg  = ~y_pos
    priv_fp_total   = np.sum(y_neg  & priv_mask)
    unpriv_fp_total = np.sum(y_neg  & unpriv_mask)
    priv_fpr  = _safe_rate(int(np.sum(yh_pos & y_neg & priv_mask)),   priv_fp_total)
    unpriv_fpr= _safe_rate(int(np.sum(yh_pos & y_neg & unpriv_mask)), unpriv_fp_total)

    # ── Positive Prediction Rates (for SPD & DIR) ─────────────────────────────
    priv_ppr  = _safe_rate(int(np.sum(yh_pos & priv_mask)),   int(np.sum(priv_mask)))
    unpriv_ppr= _safe_rate(int(np.sum(yh_pos & unpriv_mask)), int(np.sum(unpriv_mask)))

    # ── Metric Computations ───────────────────────────────────────────────────
    eod = unpriv_tpr - priv_tpr                              # should be ≈ 0
    aod = 0.5 * ((unpriv_fpr - priv_fpr) + (unpriv_tpr - priv_tpr))
    spd = unpriv_ppr - priv_ppr
    dir = _safe_rate(int(unpriv_ppr * 1000), int(priv_ppr * 1000))   # ratio

    metrics = FairnessMetrics(eod=eod, aod=aod, dir=dir, spd=spd)
    logger.info("Fairness metrics: %s", metrics)
    return metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence = (0, 1),
) -> dict:
    """Lightweight classification report (no sklearn dependency)."""
    report = {}
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        report[str(label)] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1-score":  round(f1, 4),
            "support":   int(np.sum(y_true == label)),
        }
    report["accuracy"] = round(accuracy(y_true, y_pred), 4)
    return report
