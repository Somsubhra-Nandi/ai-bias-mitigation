"""
shared/gcp_utils.py
────────────────────
Thin wrappers around google-cloud-storage, google-cloud-aiplatform,
and google-cloud-pubsub so the rest of the codebase stays free of
boiler-plate.  All functions raise on failure so callers can let
exceptions bubble up to the pipeline error handler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from google.cloud import pubsub_v1, storage
from google.cloud.aiplatform import gapic as aip_gapic
from google.protobuf import json_format
import google.cloud.aiplatform as aiplatform

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def init_vertex(project: str, location: str = "us-central1") -> None:
    """Must be called once per process before any Vertex SDK calls."""
    aiplatform.init(project=project, location=location)
    logger.info("Vertex AI SDK initialised — project=%s location=%s", project, location)


# ─────────────────────────────────────────────────────────────────────────────
# Cloud Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def gcs_upload(local_path: str | Path, gcs_uri: str) -> None:
    """Upload a local file to gs://bucket/object."""
    gcs_uri = gcs_uri.removeprefix("gs://")
    bucket_name, _, blob_name = gcs_uri.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    logger.info("Uploaded %s → gs://%s/%s", local_path, bucket_name, blob_name)


def gcs_download(gcs_uri: str, local_path: str | Path) -> Path:
    """Download gs://bucket/object to a local path."""
    gcs_uri = gcs_uri.removeprefix("gs://")
    bucket_name, _, blob_name = gcs_uri.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded gs://%s/%s → %s", bucket_name, blob_name, local_path)
    return local_path


def gcs_upload_string(content: str, gcs_uri: str, content_type: str = "text/plain") -> None:
    """Upload a string directly to GCS without touching disk."""
    gcs_uri = gcs_uri.removeprefix("gs://")
    bucket_name, _, blob_name = gcs_uri.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type=content_type)
    logger.info("Uploaded string → gs://%s/%s", bucket_name, blob_name)


def gcs_read_string(gcs_uri: str) -> str:
    """Read a GCS object as a UTF-8 string."""
    gcs_uri = gcs_uri.removeprefix("gs://")
    bucket_name, _, blob_name = gcs_uri.partition("/")
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).download_as_text()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Versioning
# ─────────────────────────────────────────────────────────────────────────────

def log_dataset_version(
    project: str,
    dataset_hash: str,
    version_tag:  str,
    gcs_raw_path: str,
    lineage_gcs_uri: str,
) -> None:
    """Append an entry to dataset_versions.json in GCS."""
    try:
        existing_json = gcs_read_string(lineage_gcs_uri)
        versions: list = json.loads(existing_json)
    except Exception:
        versions = []

    entry = {
        "version_tag":   version_tag,
        "dataset_hash":  dataset_hash,
        "gcs_raw_path":  gcs_raw_path,
        "logged_at":     datetime.utcnow().isoformat() + "Z",
    }
    versions.append(entry)
    gcs_upload_string(
        json.dumps(versions, indent=2),
        lineage_gcs_uri,
        content_type="application/json",
    )
    logger.info("Dataset version logged: %s (%s)", version_tag, dataset_hash[:12])


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def register_model(
    display_name:        str,
    artifact_uri:        str,
    serving_container:   str,
    project:             str,
    location:            str = "us-central1",
    labels:              Optional[Dict[str, str]] = None,
    description:         str = "",
) -> str:
    """Upload model to Vertex AI Model Registry and return resource name."""
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container,
        description=description,
        labels=labels or {},
    )
    logger.info("Model registered: %s", model.resource_name)
    return model.resource_name


def get_model_by_resource(resource_name: str) -> aiplatform.Model:
    return aiplatform.Model(model_name=resource_name)


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_endpoint(
    display_name: str,
    project: str,
    location: str = "us-central1",
) -> aiplatform.Endpoint:
    """Return an existing endpoint with this display_name, or create one."""
    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{display_name}"',
        project=project,
        location=location,
    )
    if existing:
        logger.info("Found existing endpoint: %s", existing[0].resource_name)
        return existing[0]
    endpoint = aiplatform.Endpoint.create(display_name=display_name)
    logger.info("Created new endpoint: %s", endpoint.resource_name)
    return endpoint


def deploy_canary(
    endpoint:          aiplatform.Endpoint,
    model_resource:    str,
    canary_traffic:    int = 10,
    machine_type:      str = "n1-standard-4",
) -> str:
    """
    Deploy a new model to the endpoint with canary_traffic% of requests.
    Returns the deployed_model_id.
    """
    model = aiplatform.Model(model_name=model_resource)
    deployed = endpoint.deploy(
        model=model,
        traffic_percentage=canary_traffic,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=3,
        deployed_model_display_name="fairguard-canary",
    )
    logger.info("Canary deployed at %d%% traffic — id=%s", canary_traffic, deployed)
    return str(deployed)


# ─────────────────────────────────────────────────────────────────────────────
# Pub/Sub — HITL Alert
# ─────────────────────────────────────────────────────────────────────────────

def publish_hitl_alert(
    project:      str,
    topic:        str,
    payload:      Dict[str, Any],
) -> str:
    """Publish a JSON message to the HITL Pub/Sub topic. Returns message_id."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic)
    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(topic_path, data=data)
    message_id = future.result(timeout=30)
    logger.info("HITL alert published → %s (msg_id=%s)", topic_path, message_id)
    return message_id


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Experiments (MLflow backend)
# ─────────────────────────────────────────────────────────────────────────────

def get_experiment_run(
    experiment_name: str,
    run_name: str,
    project: str,
    location: str = "us-central1",
) -> Any:
    """Retrieve a Vertex Experiment run object for metric inspection."""
    experiment = aiplatform.Experiment(
        experiment_name=experiment_name,
        project=project,
        location=location,
    )
    runs = aiplatform.ExperimentRun.list(experiment=experiment)
    for run in runs:
        if run.name == run_name:
            return run
    raise ValueError(f"Run '{run_name}' not found in experiment '{experiment_name}'")
