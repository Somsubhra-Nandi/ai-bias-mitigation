"""
agents/storyteller_agent.py
────────────────────────────
Phase 5.2 — The Storyteller Agent.

Takes raw bias/accuracy numbers and translates them into a "Lives
Impacted" narrative that non-technical stakeholders (compliance officers,
board members) can read and act on.  Output is fairness_scorecard.md.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from src.shared.contracts import DatasetContract, MitigationPlan, TrainingResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are the FairGuard Storyteller — a compliance narrative writer.

You receive technical fairness metrics and must produce a fairness_scorecard.md
that a non-technical compliance officer or board member can understand in under
3 minutes.

Structure your output EXACTLY as:

# 🛡️ FairGuard Fairness Scorecard
## Executive Summary (2-3 sentences max)
## The Bias We Found (plain English, no jargon)
## What We Did About It (plain English)
## Lives Impacted: Before & After
A markdown table:
| Group         | Metric        | Before | After  | Change |
|---------------|---------------|--------|--------|--------|
| ...           | ...           | ...    | ...    | ✅/⚠️  |
## Accuracy Trade-off
(Plain sentence: "We reduced bias by X% while only sacrificing Y% accuracy.")
## Residual Risk & Monitoring
(What remains imperfect and how Vertex Model Monitoring catches drift.)
## Compliance Checklist
- [ ] Equal Opportunity Difference within ±0.05 threshold
- [ ] Disparate Impact Ratio > 0.80 (four-fifths rule)
- [ ] Human reviewer approved deployment
- [ ] Canary rollout active (10% traffic)
- [ ] Vertex Model Monitoring configured

Use ✅ for passing thresholds and ⚠️ for items needing attention.
Use concrete, human language. Avoid metric names in the narrative; use
plain descriptions like "the model was less likely to approve liver-disease
patients from Group X than Group Y."
"""


def run_storyteller_agent(
    contract:         DatasetContract,
    plan:             MitigationPlan,
    result:           TrainingResult,
    gate_passed:      bool,
    endpoint_name:    str,
    output_path:      Optional[str] = None,
    model:            str = "claude-sonnet-4-20250514",
) -> str:
    """
    Generate fairness_scorecard.md.

    Returns the markdown string.
    """
    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise ValueError("GITHUB_PAT environment variable is missing.")

    # 1. Build a structured summary for the agent FIRST
    summary = {
        "dataset":          contract.dataset_name,
        "version_tag":      contract.version_tag,
        "protected_attr":   plan.protected_attribute,
        "privileged_group": plan.privileged_group,
        "unprivileged_group": plan.unprivileged_group,
        "mitigation_method": plan.method,
        "baseline": {
            "accuracy": round(result.baseline_accuracy * 100, 2),
            "eod":  result.baseline_bias.equal_opportunity_diff,
            "aod":  result.baseline_bias.average_odds_diff,
            "dir":  result.baseline_bias.disparate_impact_ratio,
            "spd":  result.baseline_bias.statistical_parity_diff,
        },
        "mitigated": {
            "accuracy": round(result.mitigated_accuracy * 100, 2),
            "eod":  result.mitigated_bias.equal_opportunity_diff,
            "aod":  result.mitigated_bias.average_odds_diff,
            "dir":  result.mitigated_bias.disparate_impact_ratio,
            "spd":  result.mitigated_bias.statistical_parity_diff,
        },
        "accuracy_drop_pct": round(
            (result.baseline_accuracy - result.mitigated_accuracy) * 100, 3
        ),
        "eod_threshold": {
            "min": plan.fairness_thresholds[0].min_value if plan.fairness_thresholds else -0.05,
            "max": plan.fairness_thresholds[0].max_value if plan.fairness_thresholds else  0.05,
        },
        "gate_passed":   gate_passed,
        "endpoint_name": endpoint_name,
        "run_id":        result.run_id,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
    }

    user_message = (
        "Generate the fairness scorecard for this model run:\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```"
    )

    # 2. Setup the API call
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {pat}",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    
    # 3. Execute
    logger.info("Storyteller Agent — calling Claude via GitHub Models...")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    markdown = resp.json()["choices"][0]["message"]["content"].strip()

    # Prepend auto-generated banner
    banner = (
        f"\n"
        f"\n"
        f"\n\n"
    )
    full_doc = banner + markdown

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(full_doc, encoding="utf-8")
        logger.info("Fairness scorecard written → %s", output_path)

    return full_doc


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.shared.contracts import load_contract, load_mitigation_plan

    contract = load_contract("dataset_contract.json")
    plan     = load_mitigation_plan("mitigation_plan.json")
    result   = TrainingResult.model_validate_json(
        Path("training_result.json").read_text()
    )
    md = run_storyteller_agent(
        contract=contract,
        plan=plan,
        result=result,
        gate_passed=True,
        endpoint_name="fairguard-endpoint",
        output_path="fairness_scorecard.md",
    )
    print(md)