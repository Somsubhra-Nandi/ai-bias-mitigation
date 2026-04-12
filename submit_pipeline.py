#!/usr/bin/env python3
"""
submit_pipeline.py
───────────────────
The detonator script.  Run this locally to kick off the entire FairGuard
pipeline.  Your machine does only orchestration; all heavy compute runs
inside Vertex AI on Google infrastructure.

Usage
─────
    export GITHUB_PAT="ghp_your_token_here"
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-ingestion-key.json"

    python submit_pipeline.py \
        --project       my-gcp-project \
        --version-tag   ilpd_v1 \
        --raw-csv       gs://my-gcp-project-data/uploads/ilpd_v1/data.csv

Optional flags (all have sensible defaults):
    --location          GCP region              [us-central1]
    --canary-pct        Canary traffic %         [10]
    --machine-type      Vertex training VM       [n1-standard-8]
    --ethics-policy     Path to ethics_policy.json
    --approval-webhook  HTTPS webhook for HITL approval
    --run-name          MLflow experiment run name
    --dry-run           Validate config only; do not submit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"fairguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("fairguard.submit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FairGuard Enterprise Pipeline — Submit Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",          required=True,  help="GCP project ID")
    p.add_argument("--version-tag",      default="ilpd_v1", help="Dataset version tag")
    p.add_argument("--raw-csv",          default="",     help="gs:// URI of raw CSV")
    p.add_argument("--location",         default="us-central1")
    p.add_argument("--canary-pct",       type=int,   default=10)
    p.add_argument("--machine-type",     default="n1-standard-8")
    p.add_argument("--ethics-policy",    default=None,   help="Local path to ethics_policy.json")
    p.add_argument("--approval-webhook", default=None,   help="HTTPS webhook URL for HITL approval")
    p.add_argument("--run-name",         default=None,   help="MLflow experiment run name")
    p.add_argument("--null-threshold",   type=float, default=0.30)
    p.add_argument("--leakage-threshold",type=float, default=0.95)
    p.add_argument("--dry-run",          action="store_true",
                   help="Validate config and env vars only; do not submit")
    return p.parse_args()


def validate_environment() -> list[str]:
    """Return a list of missing/invalid environment issues."""
    issues: list[str] = []

    if not os.environ.get("GITHUB_PAT"):
        issues.append(
            "GITHUB_PAT environment variable is not set. "
            "Required by Schema, Ethics, and Storyteller agents (GitHub Models API). "
            "Get a token at https://github.com/settings/tokens"
        )

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        # Try Application Default Credentials
        try:
            import google.auth
            google.auth.default()
        except Exception:
            issues.append(
                "No GCP credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "or run `gcloud auth application-default login`."
            )

    return issues


def main() -> None:
    args = parse_args()

    logger.info("═══════════════════════════════════════════════")
    logger.info("   🛡️  FairGuard Enterprise Pipeline Launcher  ")
    logger.info("═══════════════════════════════════════════════")
    logger.info("Project     : %s", args.project)
    logger.info("Version tag : %s", args.version_tag)
    logger.info("Location    : %s", args.location)
    logger.info("Canary %%    : %d%%", args.canary_pct)
    logger.info("Machine     : %s", args.machine_type)
    logger.info("Dry run     : %s", args.dry_run)

    # ── Environment checks ────────────────────────────────────────────────────
    env_issues = validate_environment()
    if env_issues:
        for issue in env_issues:
            logger.error("❌  %s", issue)
        sys.exit(1)
    logger.info("✅  Environment checks passed.")

    # ── Build config ──────────────────────────────────────────────────────────
    from pipeline.pipeline import PipelineConfig

    cfg = PipelineConfig(
        project=args.project,
        location=args.location,
        version_tag=args.version_tag,
        raw_csv_gcs_uri=args.raw_csv,
        canary_pct=args.canary_pct,
        machine_type=args.machine_type,
        ethics_policy_path=args.ethics_policy,
        approval_webhook=args.approval_webhook,
        run_name=args.run_name or f"fairguard-{args.version_tag}",
        null_threshold=args.null_threshold,
        leakage_threshold=args.leakage_threshold,
    )

    logger.info("Pipeline config:")
    for k, v in cfg.__dict__.items():
        logger.info("  %-22s: %s", k, v)

    if args.dry_run:
        logger.info("DRY RUN — configuration is valid. Exiting without submitting.")
        sys.exit(0)

    # ── Confirm before launch ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  About to launch FairGuard pipeline on project: {args.project}")
    print(f"  This will create GCS objects, Vertex AI jobs,")
    print(f"  and a Vertex AI Endpoint. Costs will be incurred.")
    print("─" * 60)
    confirm = input("  Type 'yes' to continue: ").strip().lower()
    if confirm != "yes":
        logger.info("Aborted by user.")
        sys.exit(0)

    # ── Run the pipeline ──────────────────────────────────────────────────────
    from pipeline.pipeline import run_pipeline

    start = datetime.utcnow()
    try:
        summary = run_pipeline(cfg)
    except RuntimeError as exc:
        logger.error("Pipeline FAILED: %s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(3)

    elapsed = (datetime.utcnow() - start).total_seconds()

    # ── Save run summary ──────────────────────────────────────────────────────
    summary_path = Path(f"run_summary_{args.version_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Run summary saved → %s", summary_path)

    logger.info(
        "Pipeline completed in %.1f seconds. "
        "Awaiting human approval via Pub/Sub.",
        elapsed,
    )


if __name__ == "__main__":
    main()
