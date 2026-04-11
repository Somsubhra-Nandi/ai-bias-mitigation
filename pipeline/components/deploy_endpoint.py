"""
pipeline/components/deploy_endpoint.py
────────────────────────────────────────
Phase 5.1 — Canary Deployment + Human-in-the-Loop (HITL) Pause.

  • Creates / updates a Vertex AI Endpoint.
  • Routes 10% of traffic to the newly debiased model.
  • Publishes a Pub/Sub alert to the Compliance Officer.
  • Intentionally suspends itself — pipeline only resumes on webhook approval.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import google.cloud.aiplatform as aiplatform

logger = logging.getLogger(__name__)


def run_deploy_endpoint(
    project:          str,
    location:         str,
    version_tag:      str,
    model_resource:   str,
    artifacts_bucket: str,
    pubsub_topic:     str,
    sa_deployment:    str,
    endpoint_display: str = "fairguard-endpoint",
    canary_pct:       int = 10,
    machine_type:     str = "n1-standard-4",
    approval_webhook: Optional[str] = None,
) -> dict:
    """
    Deploy the candidate model with a canary split and fire HITL alert.

    Returns:
      - endpoint_resource : Vertex AI Endpoint resource name
      - deployed_model_id : ID of the newly deployed model
      - pubsub_message_id : Pub/Sub message ID for the HITL alert
      - manifest_uri      : gs:// path to deployment_manifest.json
    """
    from src.shared.gcp_utils import (
        get_or_create_endpoint, publish_hitl_alert, gcs_upload_string,
    )
    from src.shared.contracts import DeploymentManifest, PipelineStatus

    aiplatform.init(project=project, location=location)

    # ── Get or create endpoint ────────────────────────────────────────────────
    endpoint = get_or_create_endpoint(
        display_name=endpoint_display,
        project=project,
        location=location,
    )

    # ── Canary deploy ─────────────────────────────────────────────────────────
    model = aiplatform.Model(model_name=model_resource)

    logger.info(
        "Deploying %s to endpoint %s at %d%% traffic…",
        model_resource, endpoint.resource_name, canary_pct,
    )

    endpoint.deploy(
        model=model,
        traffic_percentage=canary_pct,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=3,
        deployed_model_display_name=f"fairguard-canary-{version_tag}",
        service_account=sa_deployment,
        sync=True,
    )

    # Retrieve deployed model id (last deployed)
    endpoint_obj = aiplatform.Endpoint(endpoint.resource_name)
    deployed_model_id = endpoint_obj.gca_resource.deployed_models[-1].id
    logger.info("Canary live — deployed_model_id=%s", deployed_model_id)

    # ── Build HITL alert payload ──────────────────────────────────────────────
    scorecard_uri = (
        f"gs://{artifacts_bucket}/reports/{version_tag}/fairness_scorecard.md"
    )
    wit_uri = (
        f"gs://{artifacts_bucket}/notebooks/{version_tag}/FairGuard_WIT_Demo.ipynb"
    )

    alert_payload = {
        "event":              "fairguard.hitl.review_required",
        "version_tag":        version_tag,
        "endpoint":           endpoint.resource_name,
        "deployed_model_id":  deployed_model_id,
        "canary_traffic_pct": canary_pct,
        "fairness_scorecard": scorecard_uri,
        "wit_notebook":       wit_uri,
        "approval_webhook":   approval_webhook or "NOT_CONFIGURED",
        "instructions": (
            "Review the fairness scorecard and WIT notebook linked above. "
            "Then POST {\"approved\": true} to the approval_webhook URL to "
            "promote the model to 100% traffic, or {\"approved\": false} to "
            "roll it back."
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    message_id = publish_hitl_alert(
        project=project,
        topic=pubsub_topic,
        payload=alert_payload,
    )
    logger.info("HITL alert published — message_id=%s", message_id)

    # ── Persist deployment manifest ───────────────────────────────────────────
    manifest = DeploymentManifest(
        endpoint_resource_name=endpoint.resource_name,
        deployed_model_id=deployed_model_id,
        canary_traffic_pct=canary_pct,
        baseline_traffic_pct=100 - canary_pct,
        hitl_pubsub_message_id=message_id,
        status=PipelineStatus.SUSPENDED,
        approval_webhook_url=approval_webhook,
    )

    manifest_gcs = (
        f"gs://{artifacts_bucket}/deployments/{version_tag}/deployment_manifest.json"
    )
    gcs_upload_string(manifest.model_dump_json(indent=2), manifest_gcs, "application/json")
    logger.info("Deployment manifest → %s", manifest_gcs)

    logger.info(
        "Pipeline SUSPENDED — awaiting human approval.\n"
        "Compliance Officer has been notified via Pub/Sub topic '%s'.",
        pubsub_topic,
    )

    return {
        "endpoint_resource":  endpoint.resource_name,
        "deployed_model_id":  deployed_model_id,
        "pubsub_message_id":  message_id,
        "manifest_uri":       manifest_gcs,
    }
