"""
src/ml/human_impact.py
───────────────────────
Phase 1 & 2 — Degenerate Detector + Human Impact Translation.

Computes the confusion matrix on mitigated predictions, flags degenerate
models (predict-majority-class exploitation), and translates raw numbers
into human-readable cost estimates that the debate agents can argue over.

Called from local_demo.py and task.py immediately after mitigation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# ── Tuneable thresholds ───────────────────────────────────────────────────────
DEGENERATE_THRESHOLD = 0.90   # flag if model says YES to >90% of test set
DIR_FAILURE_THRESHOLD = 0.60  # DIR below this = four-fifths rule catastrophically failed

# ── Human cost constants (adjustable per deployment context) ──────────────────
# These represent realistic estimates for a liver disease screening context.
BIOPSY_COST_INR        = 45_000   # cost of one unnecessary liver biopsy (INR)
MISSED_DIAGNOSIS_COST  = "Critical care escalation / potential fatality"


def compute_human_impact(
    y_true:           np.ndarray,
    y_pred_mitigated: np.ndarray,
    mitigated_dir:    float,
    output_path:      Optional[str] = "local_artifacts/human_impact.json",
) -> dict:
    """
    Compute confusion matrix, degenerate flag, and human cost translation.

    Parameters
    ----------
    y_true            : Ground truth labels (0/1)
    y_pred_mitigated  : Mitigated model's predictions (0/1)
    mitigated_dir     : Disparate Impact Ratio from fairness metrics
    output_path       : Where to save human_impact.json (None = skip save)

    Returns
    -------
    dict — the full human_impact payload, also saved to output_path
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_mitigated)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    # sklearn returns [[TN, FP], [FN, TP]] for binary classification
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    test_set_size    = len(y_true)
    predicted_yes    = int(np.sum(y_pred == 1))
    predicted_yes_ratio = round(predicted_yes / test_set_size, 6)

    # ── Degenerate detection ──────────────────────────────────────────────────
    is_degenerate = predicted_yes_ratio > 0.90 or predicted_yes_ratio < 0.10
    dir_failure   = mitigated_dir < DIR_FAILURE_THRESHOLD

    if is_degenerate:
        logger.warning(
            "🚨 DEGENERATE MODEL DETECTED — predicted YES for %.1f%% of test set. "
            "The model is exploiting the majority class to game fairness metrics.",
            predicted_yes_ratio * 100,
        )
    else:
        logger.info(
            "✅ Degenerate check passed — predicted YES ratio: %.1f%%",
            predicted_yes_ratio * 100,
        )

    # ── Precision / Recall (real performance picture) ─────────────────────────
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(
        2 * precision * recall / (precision + recall), 4
    ) if (precision + recall) > 0 else 0.0
    false_positive_rate = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

    # ── Phase 2: Human Impact Translation ────────────────────────────────────
    # False Negatives → missed diagnoses (patient harm)
    # False Positives → unnecessary biopsies (financial + psychological harm)
    unnecessary_biopsy_cost_inr = fp * BIOPSY_COST_INR
    unnecessary_biopsy_cost_usd = round(unnecessary_biopsy_cost_inr / 83, 2)  # ~83 INR/USD

    human_impact = {
        # ── Raw confusion matrix ──────────────────────────────────────────────
        "test_set_size":         test_set_size,
        "true_positives":        tp,
        "true_negatives":        tn,
        "false_positives":       fp,
        "false_negatives":       fn,

        # ── Prediction distribution ───────────────────────────────────────────
        "predicted_yes_count":   predicted_yes,
        "predicted_yes_ratio":   predicted_yes_ratio,
        "predicted_no_count":    test_set_size - predicted_yes,

        # ── Degenerate flags ──────────────────────────────────────────────────
        "is_degenerate":         is_degenerate,
        "degenerate_threshold":  DEGENERATE_THRESHOLD,
        "dir_failure":           dir_failure,
        "dir_failure_threshold": DIR_FAILURE_THRESHOLD,
        "mitigated_dir":         round(mitigated_dir, 4),

        # ── Classification performance ────────────────────────────────────────
        "precision":             precision,
        "recall":                recall,
        "f1_score":              f1,
        "false_positive_rate":   false_positive_rate,

        # ── Phase 2: Human cost translation ──────────────────────────────────
        "human_cost": {
            "false_negatives_interpretation": (
                f"{fn} patients with liver disease were MISSED by the model. "
                f"Each represents a potential delayed diagnosis, critical care "
                f"escalation, or fatality."
            ),
            "false_positives_interpretation": (
                f"{fp} healthy patients were falsely flagged as diseased. "
                f"Each faces an unnecessary liver biopsy, patient trauma, and "
                f"wasted clinical resources."
            ),
            "unnecessary_biopsy_cost_inr":    unnecessary_biopsy_cost_inr,
            "unnecessary_biopsy_cost_usd":    unnecessary_biopsy_cost_usd,
            "missed_diagnoses_risk":          MISSED_DIAGNOSIS_COST,
            "patient_trauma_count":           fp,
            "lives_at_risk_from_fn":          fn,
        },

        # ── Deployment recommendation ─────────────────────────────────────────
        "deployment_recommendation": (
            "REJECT — Model collapse detected. Predicting majority class."
            if is_degenerate else
            "REJECT — DIR critically below four-fifths rule threshold."
            if dir_failure else
            "CONDITIONAL APPROVE — Pending human reviewer sign-off."
        ),
    }

    # ── Save artifact ─────────────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(human_impact, indent=2))
        logger.info("Human impact artifact saved → %s", output_path)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info(
        "\n"
        "┌─────────────────────────────────────────────────────┐\n"
        "│              CONFUSION MATRIX ANALYSIS              │\n"
        "├────────────────────┬────────────────────────────────┤\n"
        "│  True  Positives   │  %-30d  │\n"
        "│  True  Negatives   │  %-30d  │\n"
        "│  False Positives   │  %-30d  │\n"
        "│  False Negatives   │  %-30d  │\n"
        "├────────────────────┼────────────────────────────────┤\n"
        "│  Precision         │  %-30.4f  │\n"
        "│  Recall            │  %-30.4f  │\n"
        "│  F1 Score          │  %-30.4f  │\n"
        "│  Predicted YES %%   │  %-30s  │\n"
        "│  Is Degenerate     │  %-30s  │\n"
        "│  Biopsy Cost (INR) │  %-30s  │\n"
        "└────────────────────┴────────────────────────────────┘",
        tp, tn, fp, fn,
        precision, recall, f1,
        f"{predicted_yes_ratio*100:.1f}%",
        f"🚨 YES" if is_degenerate else "✅ NO",
        f"₹{unnecessary_biopsy_cost_inr:,}",
    )

    return human_impact