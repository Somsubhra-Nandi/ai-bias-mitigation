"""
pipeline/components/train_and_mitigate.py
──────────────────────────────────────────
Phase 3 — Reproducible Cloud Execution.

Launches a Vertex AI CustomTrainingJob that runs src/ml/task.py inside
Google's pre-built Scikit-learn container.  The local machine does ZERO
compute — it only submits the job and polls for completion.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import google.cloud.aiplatform as aiplatform

logger = logging.getLogger(__name__)

# Google's managed Scikit-learn training container
SKLEARN_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest"
)


def run_train_and_mitigate(
    project:           str,
    location:          str,
    version_tag:       str,
    validated_gcs_uri: str,
    contract_gcs_uri:  str,
    plan_gcs_uri:      str,
    artifacts_bucket:  str,
    pipeline_root:     str,
    sa_training_email: str,
    machine_type:      str = "n1-standard-8",
    run_name:          Optional[str] = None,
) -> dict:
    """
    Submit and await a Vertex AI CustomTrainingJob.

    Returns:
      - result_uri      : gs:// path to training_result.json
      - model_uri       : gs:// path to model.joblib
      - run_id          : MLflow / Vertex Experiments run ID
    """
    aiplatform.init(project=project, location=location)

    output_gcs = f"gs://{artifacts_bucket}/models/{version_tag}"
    display_name = f"fairguard-train-{version_tag}"

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": SKLEARN_CONTAINER,
                "command":   ["python", "-m", "src.ml.task"],
                "args": [
                    "--project",          project,
                    "--location",         location,
                    "--data-gcs-uri",     validated_gcs_uri,
                    "--contract-gcs-uri", contract_gcs_uri,
                    "--plan-gcs-uri",     plan_gcs_uri,
                    "--output-gcs-uri",   output_gcs,
                    "--run-name",         run_name or f"fairguard-{version_tag}",
                ],
                "env": [
                    {
                        "name":  "GITHUB_PAT",
                        "value": os.environ.get("GITHUB_PAT", ""),
                    }
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=output_gcs,
        project=project,
        location=location,
    )

    logger.info("Submitting CustomTrainingJob: %s", display_name)
    job.run(
        service_account=sa_training_email,
        timeout=7200,        # 2-hour hard limit
        restart_job_on_worker_restart=False,
        sync=True,           # Block until complete
    )
    logger.info("Training job finished with state: %s", job.state)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Training job {display_name} ended in state {job.state.name}. "
            "Check Vertex AI console for logs."
        )

    result_uri = f"{output_gcs}/training_result.json"
    model_uri  = f"{output_gcs}/model.joblib"

    # Extract run_id from the result artifact
    from src.shared.gcp_utils import gcs_read_string
    import json
    result_json = gcs_read_string(result_uri)
    run_id = json.loads(result_json).get("run_id", "unknown")

    return {
        "result_uri": result_uri,
        "model_uri":  model_uri,
        "run_id":     run_id,
        "output_dir": output_gcs,
    }
