"""
pipeline/components/validate_data.py
──────────────────────────────────────
Phase 1 — Ingestion, Profiling, Schema Agent, Validation Gate.

This KFP component:
  1.1  Hashes the raw CSV and versions it.
  1.2  Runs deterministic statistical profiling (no raw rows).
  1.3  Calls schema_agent → DatasetContract.
  1.4  Enforces null checks, type checks, and leakage (Pearson ≥ 0.95).
       Hard stop on failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import numpy as np
from kfp.v2.dsl import component, Output, Artifact, Dataset

logger = logging.getLogger(__name__)

# ── Standalone (non-KFP) entrypoint used by submit_pipeline.py ───────────────

def run_validate_data(
    raw_csv_gcs_uri:    str,
    project:            str,
    version_tag:        str,
    data_bucket:        str,
    artifacts_bucket:   str,
    ethics_policy_path: str | None = None,
    null_threshold:     float = 0.30,
    leakage_threshold:  float = 0.95,
) -> dict:
    """
    Orchestrate Phase 1 end-to-end.

    Returns a dict with:
      - status          : "passed" | "failed"
      - validated_uri   : gs:// path to clean parquet
      - contract_uri    : gs:// path to dataset_contract.json
      - failure_report  : gs:// path to failure_report.json (if failed)
    """
    from src.shared.gcp_utils import (
        gcs_download, gcs_upload, gcs_upload_string,
        sha256_file, log_dataset_version,
    )
    from src.agents.schema_agent import run_schema_agent
    from src.shared.contracts import ValidationReport, LeakageFlag, PipelineStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1.1 Download & hash ───────────────────────────────────────────────
        raw_path = gcs_download(raw_csv_gcs_uri, tmpdir / "raw.csv")
        dataset_hash = sha256_file(raw_path)
        logger.info("Dataset hash: %s  tag: %s", dataset_hash, version_tag)

        versioned_gcs = f"gs://{data_bucket}/raw/{version_tag}/data.csv"
        gcs_upload(raw_path, versioned_gcs)
        log_dataset_version(
            project=project,
            dataset_hash=dataset_hash,
            version_tag=version_tag,
            gcs_raw_path=versioned_gcs,
            lineage_gcs_uri=f"gs://{artifacts_bucket}/lineage/dataset_versions.json",
        )

        # ── 1.2 Statistical profiling (metadata only) ─────────────────────────
        df = pd.read_csv(raw_path)
        profile = _build_statistical_profile(df)
        logger.info("Statistical profile built for %d columns", len(profile["columns"]))

        # ── 1.3 Schema Agent ──────────────────────────────────────────────────
        contract_local = tmpdir / "dataset_contract.json"
        contract = run_schema_agent(
            statistical_profile=profile,
            dataset_name=version_tag.split("_")[0],
            dataset_hash=dataset_hash,
            version_tag=version_tag,
            output_path=str(contract_local),
        )
        contract_gcs = f"gs://{artifacts_bucket}/contracts/{version_tag}/dataset_contract.json"
        gcs_upload(contract_local, contract_gcs)

        # ── 1.4 Validation Gate ───────────────────────────────────────────────
        null_checks:    dict  = {}
        type_checks:    dict  = {}
        leakage_flags:  list  = []
        failure_reasons:list  = []

        # Null check
        for col in df.columns:
            pct = float(df[col].isna().mean())
            null_checks[col] = pct
            if pct > null_threshold:
                failure_reasons.append(
                    f"Column '{col}' has {pct:.1%} missing values (threshold: {null_threshold:.0%})."
                )

        # Type check (protected attrs must be categorical / object)
        for attr in contract.protected_attributes:
            is_cat = df[attr].dtype in (object, "category") or df[attr].nunique() < 20
            type_checks[attr] = is_cat
            if not is_cat:
                failure_reasons.append(
                    f"Protected attribute '{attr}' appears numeric (cardinality {df[attr].nunique()})."
                )

        # Leakage check (Pearson with target)
        target_col = contract.target_variable
        if target_col in df.columns:
            y_enc = pd.factorize(df[target_col])[0]
            for col in df.select_dtypes(include=[np.number]).columns:
                if col == target_col:
                    continue
                r = abs(float(np.corrcoef(df[col].fillna(0), y_enc)[0, 1]))
                if r >= leakage_threshold:
                    flag = LeakageFlag(
                        feature=col, correlation=r, threshold=leakage_threshold
                    )
                    leakage_flags.append(flag)
                    failure_reasons.append(
                        f"Leakage detected: '{col}' correlates {r:.3f} with target."
                    )

        status = PipelineStatus.PASSED if not failure_reasons else PipelineStatus.FAILED

        report = ValidationReport(
            status=status,
            dataset_hash=dataset_hash,
            version_tag=version_tag,
            null_checks=null_checks,
            type_checks=type_checks,
            leakage_flags=leakage_flags,
            failure_reasons=failure_reasons,
        )

        report_name = "validation_report.json" if status == PipelineStatus.PASSED else "failure_report.json"
        report_gcs  = f"gs://{artifacts_bucket}/reports/{version_tag}/{report_name}"
        gcs_upload_string(report.model_dump_json(indent=2), report_gcs, "application/json")

        if status == PipelineStatus.FAILED:
            logger.error("Validation FAILED — pipeline halted. Reasons:\n%s",
                         "\n".join(failure_reasons))
            return {"status": "failed", "failure_report": report_gcs}

        # Move validated data
        validated_gcs = f"gs://{data_bucket}/validated/{version_tag}/data.csv"
        gcs_upload(raw_path, validated_gcs)
        logger.info("Validation PASSED — data at %s", validated_gcs)

        return {
            "status":        "passed",
            "validated_uri": validated_gcs,
            "contract_uri":  contract_gcs,
            "dataset_hash":  dataset_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical profiler (inspects metadata, NOT raw sensitive rows)
# ─────────────────────────────────────────────────────────────────────────────

def _build_statistical_profile(df: pd.DataFrame) -> dict:
    columns = []
    for col in df.columns:
        series = df[col]
        col_info: dict = {
            "name":        col,
            "dtype":       str(series.dtype),
            "cardinality": int(series.nunique()),
            "missing_pct": round(float(series.isna().mean()), 4),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info["stats"] = {
                "min":  round(float(desc["min"]),  4),
                "max":  round(float(desc["max"]),  4),
                "mean": round(float(desc["mean"]), 4),
                "std":  round(float(desc["std"]),  4),
            }
        else:
            # Only expose top-5 category names (no actual sensitive values)
            col_info["top_categories"] = list(
                series.value_counts().head(5).index.astype(str)
            )

        columns.append(col_info)

    return {
        "row_count":    len(df),
        "column_count": len(df.columns),
        "columns":      columns,
        "profiled_at":  datetime.utcnow().isoformat() + "Z",
    }
