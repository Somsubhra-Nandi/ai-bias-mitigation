"""
pipeline/components/evaluate_and_register.py
─────────────────────────────────────────────
Phase 4 — Evaluation Gate (Hard Stop #2) + Model Registry push.

Checks:
  • Mitigated EOD within ±0.05 threshold
  • Accuracy drop ≤ 2% vs baseline

If either check fails → pipeline dies.
If both pass → model pushed to Vertex AI Model Registry as a CANDIDATE.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SERVING_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
)


def run_evaluate_and_register(
    project:          str,
    location:         str,
    version_tag:      str,
    result_gcs_uri:   str,
    plan_gcs_uri:     str,
    model_gcs_uri:    str,
    artifacts_bucket: str,
) -> dict:
    """
    Evaluation gate + conditional Model Registry push.

    Returns:
      - gate_passed        : bool
      - model_resource     : Vertex AI Model resource name (if passed)
      - gate_report_uri    : gs:// path to gate_decision.json
    """
    from src.shared.gcp_utils import gcs_read_string, gcs_upload_string
    from src.shared.contracts import (
        TrainingResult, MitigationPlan,
        GateDecision, PipelineStatus, save_model,
    )
    import google.cloud.aiplatform as aiplatform

    # ── Load artefacts ────────────────────────────────────────────────────────
    result = TrainingResult.model_validate_json(gcs_read_string(result_gcs_uri))
    plan   = MitigationPlan.model_validate_json(gcs_read_string(plan_gcs_uri))

    mitigated_eod     = result.mitigated_bias.equal_opportunity_diff
    accuracy_drop_pct = (result.baseline_accuracy - result.mitigated_accuracy) * 100

    # ── Derive thresholds from plan ───────────────────────────────────────────
    eod_min = -0.05
    eod_max =  0.05
    if plan.fairness_thresholds:
        t = plan.fairness_thresholds[0]
        eod_min, eod_max = t.min_value, t.max_value

    max_drop = plan.max_accuracy_drop_pct

    # ── Run checks ────────────────────────────────────────────────────────────
    checks: dict[str, bool] = {
        f"eod_within_[{eod_min},{eod_max}]":          eod_min <= mitigated_eod <= eod_max,
        f"accuracy_drop_leq_{max_drop}pct":           accuracy_drop_pct <= max_drop,
        "dir_above_four_fifths_rule":                  result.mitigated_bias.disparate_impact_ratio >= 0.80,
    }
    failure_reasons = [name for name, passed in checks.items() if not passed]
    gate_passed = len(failure_reasons) == 0

    logger.info(
        "Evaluation Gate — EOD=%.4f  drop=%.3f%%  passed=%s",
        mitigated_eod, accuracy_drop_pct, gate_passed,
    )

    # ── Build gate decision ───────────────────────────────────────────────────
    model_resource: Optional[str] = None

    if gate_passed:
        aiplatform.init(project=project, location=location)

        # Model artifact dir is the parent of model.joblib
        model_artifact_dir = str(Path(model_gcs_uri).parent)

        model = aiplatform.Model.upload(
            display_name=f"fairguard-candidate-{version_tag}",
            artifact_uri=model_artifact_dir,
            serving_container_image_uri=SERVING_CONTAINER,
            description=(
                f"FairGuard debiased model | version={version_tag} | "
                f"EOD={mitigated_eod:.4f} | run={result.run_id}"
            ),
            labels={
                "stage":       "candidate",
                "version_tag": version_tag.replace("_", "-"),
                "run_id":      result.run_id[:20],
            },
        )
        model_resource = model.resource_name
        logger.info("Model registered (CANDIDATE): %s", model_resource)
    else:
        logger.error(
            "Evaluation Gate FAILED — pipeline halted.\nFailed checks: %s",
            failure_reasons,
        )

    gate = GateDecision(
        gate_name="evaluation_gate_phase4",
        passed=gate_passed,
        checks=checks,
        failure_reasons=failure_reasons,
        model_registry_resource=model_resource,
    )

    gate_json = gate.model_dump_json(indent=2)
    gate_gcs  = f"gs://{artifacts_bucket}/reports/{version_tag}/gate_decision.json"
    gcs_upload_string(gate_json, gate_gcs, "application/json")

    if not gate_passed:
        raise RuntimeError(
            f"Hard Stop #2 — Evaluation Gate failed: {failure_reasons}. "
            f"See {gate_gcs} for details."
        )

    return {
        "gate_passed":    True,
        "model_resource": model_resource,
        "gate_report_uri":gate_gcs,
    }
