"""
pipeline/components/generate_strategy.py
──────────────────────────────────────────
Phase 2 — Policy-Constrained Ethics Strategy.

Invokes the Ethics Strategist Agent to produce:
  • mitigation_plan.json
  • ethics_decision_log.md
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run_generate_strategy(
    contract_gcs_uri:    str,
    artifacts_bucket:    str,
    version_tag:         str,
    ethics_policy_path:  str | None = None,
    historical_skew:     dict | None = None,
) -> dict:
    """
    Orchestrate Phase 2.

    Returns:
      - plan_uri        : gs:// path to mitigation_plan.json
      - decision_log_uri: gs:// path to ethics_decision_log.md
    """
    from src.shared.gcp_utils import gcs_download, gcs_upload
    from src.shared.contracts import load_contract
    from src.agents.ethics_agent import run_ethics_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download contract
        contract_local = gcs_download(contract_gcs_uri, tmpdir / "dataset_contract.json")
        contract = load_contract(contract_local)

        plan_local = tmpdir / "mitigation_plan.json"
        log_local  = tmpdir / "ethics_decision_log.md"

        plan, log_md = run_ethics_agent(
            contract=contract,
            policy_path=ethics_policy_path,
            historical_skew=historical_skew,
            plan_output_path=str(plan_local),
            log_output_path=str(log_local),
        )

        plan_gcs = f"gs://{artifacts_bucket}/plans/{version_tag}/mitigation_plan.json"
        log_gcs  = f"gs://{artifacts_bucket}/plans/{version_tag}/ethics_decision_log.md"

        gcs_upload(plan_local, plan_gcs)
        gcs_upload(log_local,  log_gcs)

        logger.info("Mitigation plan → %s", plan_gcs)
        logger.info("Ethics log       → %s", log_gcs)

        return {
            "plan_uri":         plan_gcs,
            "decision_log_uri": log_gcs,
            "method":           plan.method,
        }
