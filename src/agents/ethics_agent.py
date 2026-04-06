"""
agents/ethics_agent.py
───────────────────────
Phase 2 — The Policy-Constrained Ethics Strategist Agent.

The agent is BOUNDED by ethics_policy.json.  It cannot choose a
mitigation method that is not on the allowlist, and it cannot propose
an accuracy drop larger than the policy maximum.  All decisions are
logged in a human-readable ethics_decision_log.md.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.shared.contracts import (
    DatasetContract,
    MitigationMethod,
    MitigationPlan,
    save_model,
)

logger = logging.getLogger(__name__)

# ── Default ethics policy (overridable via JSON file) ────────────────────────
DEFAULT_ETHICS_POLICY: Dict[str, Any] = {
    "policy_version": "ethics_policy_v1",
    "allowed_mitigation_methods": [
        "reweighing",
        "threshold_optimizer",
        "calibrated_eq_odds",
    ],
    "max_accuracy_drop_pct": 2.0,
    "fairness_metric": "eod",
    "target_eod_range": [-0.05, 0.05],
    "target_aod_range": [-0.07, 0.07],
    "prohibited_features": ["name", "id", "ssn", "national_id"],
}


_SYSTEM_PROMPT = """
You are the FairGuard Ethics Strategist — a policy-constrained JSON emitter.

You receive:
  1. A DatasetContract describing the validated dataset.
  2. An ethics_policy defining hard constraints you MUST obey.
  3. Historical bias skew metrics (may be null on first run).

Your job is to output TWO artefacts:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTEFACT 1 — mitigation_plan (JSON block between <<<JSON and >>>)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schema:
{
  "plan_version":          "1.0",
  "dataset_version_tag":   "<tag>",
  "method":                "<one of the allowed_mitigation_methods>",
  "protected_attribute":   "<column name>",
  "privileged_group":      "<group label>",
  "unprivileged_group":    "<group label>",
  "fairness_thresholds": [
    {"metric": "eod", "min_value": <float>, "max_value": <float>}
  ],
  "max_accuracy_drop_pct": <float>,
  "hyperparameters":       {},
  "policy_version":        "<policy version string>",
  "rationale_summary":     "<≤3 sentence explanation>"
}

RULES:
- method MUST be in the policy's allowed_mitigation_methods list.
- max_accuracy_drop_pct MUST NOT exceed the policy's max_accuracy_drop_pct.
- Fairness thresholds MUST be within the policy's target ranges.
- Do NOT include any field not in the schema above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTEFACT 2 — decision_log (Markdown block between <<<MD and >>>)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A concise (~300 word) Markdown document with:
- ## Why this mitigation method?
- ## Why these fairness thresholds?
- ## Trade-offs acknowledged
- ## Policy constraints applied

Format your entire response as:
<<<JSON
{ ... }
>>>
<<<MD
# FairGuard Ethics Decision Log
...
>>>
"""


def run_ethics_agent(
    contract:           DatasetContract,
    policy_path:        Optional[str]          = None,
    historical_skew:    Optional[Dict[str, Any]] = None,
    plan_output_path:   Optional[str]          = None,
    log_output_path:    Optional[str]          = None,
    model:              str                    = "claude-sonnet-4-20250514",
) -> tuple[MitigationPlan, str]:
    """
    Run the ethics agent.

    Returns
    -------
    (MitigationPlan, decision_log_markdown)
    """
    # Load policy
    if policy_path and Path(policy_path).exists():
        policy = json.loads(Path(policy_path).read_text())
        logger.info("Loaded ethics policy from %s", policy_path)
    else:
        policy = DEFAULT_ETHICS_POLICY
        logger.warning("No policy file found — using DEFAULT_ETHICS_POLICY.")

    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise ValueError("GITHUB_PAT environment variable is missing.")

    # 1. Define the user message FIRST
    user_message = (
        f"## DatasetContract\n```json\n{contract.model_dump_json(indent=2)}\n```\n\n"
        f"## Ethics Policy\n```json\n{json.dumps(policy, indent=2)}\n```\n\n"
        f"## Historical Bias Skew\n```json\n{json.dumps(historical_skew or {}, indent=2)}\n```\n\n"
        "Generate both artefacts now."
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
    logger.info("Ethics Agent — calling Claude via GitHub Models...")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    logger.debug("Ethics Agent raw output:\n%s", raw)

    # ── Parse both blocks ────────────────────────────────────────────────────
    json_block, md_block = _parse_dual_output(raw)

    # ── Validate JSON against policy ─────────────────────────────────────────
    plan_dict = json.loads(json_block)

    allowed = policy.get("allowed_mitigation_methods", [])
    if plan_dict.get("method") not in allowed:
        raise ValueError(
            f"Agent proposed disallowed method '{plan_dict['method']}'. "
            f"Allowed: {allowed}"
        )

    max_drop = policy.get("max_accuracy_drop_pct", 2.0)
    if plan_dict.get("max_accuracy_drop_pct", 0) > max_drop:
        raise ValueError(
            f"Agent proposed accuracy drop {plan_dict['max_accuracy_drop_pct']}% "
            f"> policy max {max_drop}%."
        )

    plan_dict["policy_version"] = policy["policy_version"]
    plan = MitigationPlan.model_validate(plan_dict)

    if plan_output_path:
        save_model(plan, plan_output_path)
        logger.info("MitigationPlan written → %s", plan_output_path)

    if log_output_path:
        # Prepend metadata header
        header = (
            f"\n"
            f"\n"
            f"\n\n"
        )
        Path(log_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_output_path).write_text(header + md_block, encoding="utf-8")
        logger.info("Ethics decision log written → %s", log_output_path)

    return plan, md_block


def _parse_dual_output(raw: str) -> tuple[str, str]:
    """Extract <<<JSON ... >>> and <<<MD ... >>> blocks."""
    try:
        json_start = raw.index("<<<JSON") + len("<<<JSON")
        json_end   = raw.index(">>>", json_start)
        json_block = raw[json_start:json_end].strip()

        md_start   = raw.index("<<<MD",  json_end) + len("<<<MD")
        md_end     = raw.index(">>>", md_start)
        md_block   = raw[md_start:md_end].strip()
    except ValueError as exc:
        raise ValueError(
            "Ethics Agent response did not contain expected <<<JSON>>>  <<<MD>>> "
            f"delimiters.\nRaw output:\n{raw}"
        ) from exc

    return json_block, md_block


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.shared.contracts import load_contract

    contract = load_contract(sys.argv[1] if len(sys.argv) > 1 else "dataset_contract.json")
    plan, log = run_ethics_agent(
        contract=contract,
        plan_output_path="mitigation_plan.json",
        log_output_path="ethics_decision_log.md",
    )
    print(plan.model_dump_json(indent=2))