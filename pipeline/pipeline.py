"""
pipeline/pipeline.py
─────────────────────
The @dsl.pipeline definition — the single file that stitches all 5 phases
into one Vertex AI Pipeline DAG.

Each phase calls the corresponding component function.  Hard stops are
implemented as Python exceptions; KFP will mark the run as FAILED and
halt downstream steps automatically.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All knobs in one place — passed to run_pipeline()."""
    project:             str
    location:            str             = "us-central1"
    version_tag:         str             = "ilpd_v1"
    raw_csv_gcs_uri:     str             = ""          # must be set before run
    data_bucket:         str             = ""          # {project}-data
    artifacts_bucket:    str             = ""          # {project}-artifacts
    pipeline_root:       str             = ""          # gs://{project}-pipeline-root
    pubsub_topic:        str             = "fairguard-hitl-alerts"
    sa_ingestion:        str             = ""
    sa_training:         str             = ""
    sa_deployment:       str             = ""
    ethics_policy_path:  Optional[str]   = None
    historical_skew:     Optional[dict]  = None
    endpoint_display:    str             = "fairguard-endpoint"
    canary_pct:          int             = 10
    machine_type:        str             = "n1-standard-8"
    approval_webhook:    Optional[str]   = None
    null_threshold:      float           = 0.30
    leakage_threshold:   float           = 0.95
    run_name:            Optional[str]   = None

    def __post_init__(self):
        # Auto-derive bucket names from project if not set
        if not self.data_bucket:
            self.data_bucket = f"{self.project}-data"
        if not self.artifacts_bucket:
            self.artifacts_bucket = f"{self.project}-artifacts"
        if not self.pipeline_root:
            self.pipeline_root = f"gs://{self.project}-pipeline-root"
        if not self.sa_ingestion:
            self.sa_ingestion = f"sa-ingestion@{self.project}.iam.gserviceaccount.com"
        if not self.sa_training:
            self.sa_training = f"sa-training@{self.project}.iam.gserviceaccount.com"
        if not self.sa_deployment:
            self.sa_deployment = f"sa-deployment@{self.project}.iam.gserviceaccount.com"
        if not self.raw_csv_gcs_uri:
            self.raw_csv_gcs_uri = (
                f"gs://{self.data_bucket}/uploads/{self.version_tag}/data.csv"
            )


def run_pipeline(cfg: PipelineConfig) -> dict:
    """
    Orchestrate all 5 phases sequentially.

    This function is the "detonator" — it runs each phase in order,
    passing outputs from one phase as inputs to the next.
    Each phase raises on failure (hard stop).

    Returns a summary dict with all artefact URIs.
    """
    from pipeline.components.validate_data       import run_validate_data
    from pipeline.components.generate_strategy   import run_generate_strategy
    from pipeline.components.train_and_mitigate  import run_train_and_mitigate
    from pipeline.components.evaluate_and_register import run_evaluate_and_register
    from pipeline.components.deploy_endpoint     import run_deploy_endpoint
    from pipeline.components.generate_reports    import run_generate_reports

    summary: dict = {"version_tag": cfg.version_tag, "config": cfg.__dict__.copy()}

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Secure Ingestion & Validation Gate
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 1 — INGESTION  ━━━━━━━━━━━━━━━━━━━━")
    phase1 = run_validate_data(
        raw_csv_gcs_uri=cfg.raw_csv_gcs_uri,
        project=cfg.project,
        version_tag=cfg.version_tag,
        data_bucket=cfg.data_bucket,
        artifacts_bucket=cfg.artifacts_bucket,
        ethics_policy_path=cfg.ethics_policy_path,
        null_threshold=cfg.null_threshold,
        leakage_threshold=cfg.leakage_threshold,
    )
    if phase1["status"] != "passed":
        raise RuntimeError(
            f"HARD STOP #1 — Validation failed. "
            f"Failure report: {phase1.get('failure_report')}"
        )
    summary["phase1"] = phase1
    logger.info("Phase 1 ✅  validated_uri=%s", phase1["validated_uri"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Ethics Strategy
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 2 — ETHICS STRATEGY  ━━━━━━━━━━━━━━")
    phase2 = run_generate_strategy(
        contract_gcs_uri=phase1["contract_uri"],
        artifacts_bucket=cfg.artifacts_bucket,
        version_tag=cfg.version_tag,
        ethics_policy_path=cfg.ethics_policy_path,
        historical_skew=cfg.historical_skew,
    )
    summary["phase2"] = phase2
    logger.info("Phase 2 ✅  method=%s", phase2["method"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Reproducible Cloud Training
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 3 — TRAINING  ━━━━━━━━━━━━━━━━━━━━━")
    phase3 = run_train_and_mitigate(
        project=cfg.project,
        location=cfg.location,
        version_tag=cfg.version_tag,
        validated_gcs_uri=phase1["validated_uri"],
        contract_gcs_uri=phase1["contract_uri"],
        plan_gcs_uri=phase2["plan_uri"],
        artifacts_bucket=cfg.artifacts_bucket,
        pipeline_root=cfg.pipeline_root,
        sa_training_email=cfg.sa_training,
        machine_type=cfg.machine_type,
        run_name=cfg.run_name,
    )
    summary["phase3"] = phase3
    logger.info("Phase 3 ✅  run_id=%s", phase3["run_id"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Evaluation Gate + Registry
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 4 — EVALUATION GATE  ━━━━━━━━━━━━━━")
    phase4 = run_evaluate_and_register(
        project=cfg.project,
        location=cfg.location,
        version_tag=cfg.version_tag,
        result_gcs_uri=phase3["result_uri"],
        plan_gcs_uri=phase2["plan_uri"],
        model_gcs_uri=phase3["model_uri"],
        artifacts_bucket=cfg.artifacts_bucket,
    )
    summary["phase4"] = phase4
    logger.info("Phase 4 ✅  model_resource=%s", phase4["model_resource"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5a — Generate Reports (before deploy so scorecard is ready)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 5a — REPORTS  ━━━━━━━━━━━━━━━━━━━━━")

    # We need an endpoint name for the notebook even before deploy;
    # use a placeholder that will be updated after deploy.
    placeholder_endpoint = (
        f"projects/{cfg.project}/locations/{cfg.location}/"
        f"endpoints/{cfg.endpoint_display}"
    )
    phase5a = run_generate_reports(
        project=cfg.project,
        location=cfg.location,
        version_tag=cfg.version_tag,
        contract_gcs_uri=phase1["contract_uri"],
        plan_gcs_uri=phase2["plan_uri"],
        result_gcs_uri=phase3["result_uri"],
        endpoint_resource=placeholder_endpoint,
        gate_passed=phase4["gate_passed"],
        artifacts_bucket=cfg.artifacts_bucket,
    )
    summary["phase5a"] = phase5a
    logger.info("Phase 5a ✅  scorecard=%s", phase5a["scorecard_uri"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5b — Canary Deploy + HITL Pause
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━━━━━━━  PHASE 5b — CANARY DEPLOY  ━━━━━━━━━━━━━━━")
    phase5b = run_deploy_endpoint(
        project=cfg.project,
        location=cfg.location,
        version_tag=cfg.version_tag,
        model_resource=phase4["model_resource"],
        artifacts_bucket=cfg.artifacts_bucket,
        pubsub_topic=cfg.pubsub_topic,
        sa_deployment=cfg.sa_deployment,
        endpoint_display=cfg.endpoint_display,
        canary_pct=cfg.canary_pct,
        machine_type="n1-standard-4",
        approval_webhook=cfg.approval_webhook,
    )
    summary["phase5b"] = phase5b
    logger.info(
        "Phase 5b ✅  endpoint=%s  HITL msg=%s",
        phase5b["endpoint_resource"],
        phase5b["pubsub_message_id"],
    )

    logger.info(
        "\n\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║            🛡️  FAIRGUARD PIPELINE COMPLETE                   ║\n"
        "║                                                              ║\n"
        "║  Pipeline is now SUSPENDED pending human approval.           ║\n"
        "║  The Compliance Officer has been notified via Pub/Sub.       ║\n"
        "║                                                              ║\n"
        "║  Artefacts:                                                  ║\n"
        f"║  • Scorecard : {phase5a['scorecard_uri'][:46]:<46} ║\n"
        f"║  • WIT NB    : {phase5a['wit_notebook_uri'][:46]:<46} ║\n"
        f"║  • Gate rpt  : {phase4['gate_report_uri'][:46]:<46} ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    return summary
