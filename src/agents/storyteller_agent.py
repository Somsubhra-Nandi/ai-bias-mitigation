"""
src/agents/storyteller_agent.py
────────────────────────────────
Phase 4 — The Brutal Scorecard (upgraded Storyteller Agent).

CRITICAL LOGIC OVERRIDE:
  If human_impact["is_degenerate"] is True  → DEPLOYMENT REJECTED audit report.
  If DIR < 0.60 (four-fifths rule catastrophic failure) → same rejection path.
  Otherwise → standard compliance scorecard.

All metrics are dynamically read from JSON artifacts. Nothing is hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from src.shared.contracts import DatasetContract, MitigationPlan, TrainingResult

logger = logging.getLogger(__name__)

DIR_REJECTION_THRESHOLD = 0.60

_SYSTEM_PROMPT_STANDARD = """
You are the FairGuard Storyteller — a compliance narrative writer.
Produce a fairness_scorecard.md readable in under 3 minutes.

Structure EXACTLY as:
# 🛡️ FairGuard Fairness Scorecard
## Executive Summary
## The Bias We Found
## What We Did About It
## Lives Impacted: Before & After
| Group | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
## Human Cost Analysis
(FP → unnecessary procedures + financial cost. FN → patients at risk.)
## Accuracy Trade-off
## Residual Risk & Monitoring
## Compliance Checklist
- [ ] EOD within ±0.05
- [ ] DIR > 0.80 (four-fifths rule)
- [ ] Predicted YES ratio < 90%
- [ ] Precision > 0.60
- [ ] Human reviewer approved
- [ ] Canary rollout active
- [ ] Vertex Model Monitoring configured

Use ✅/⚠️/❌. Use ONLY the stats provided — never invent numbers.
"""

_SYSTEM_PROMPT_REJECTION = """
You are the FairGuard Compliance Auditor. You write audit documents, not PR reports.
A model has FAILED safety checks. Generate a Critical Audit Report.

Structure EXACTLY as:
# 🚨 FairGuard Critical Audit Report — DEPLOYMENT REJECTED

## Verdict
**DEPLOYMENT REJECTED** — [one sentence: primary failure reason]

## Technical Failure Analysis
### What Went Wrong
### Why This Is Dangerous

## Human Impact Assessment
| Impact Category | Count / Cost | Severity |
|-----------------|--------------|----------|
| Unnecessary biopsies | ... | 🔴 HIGH |
| Missed diagnoses | ... | 🔴 HIGH |
| Financial waste | ₹... / $... | 🟡 MED |
| Regulatory exposure | EU AI Act / EEOC | 🔴 HIGH |

## Agent Debate Summary
(3-4 sentences. State whose argument prevailed and why.)

## Root Cause
## Required Remediation Before Re-deployment
(Numbered list of concrete technical steps.)

## Compliance Checklist
(❌ for failed items, ✅/⚠️ for others.)

Do NOT soften the verdict. Use ONLY the stats provided.
"""


def run_storyteller_agent(
    contract:              DatasetContract,
    plan:                  MitigationPlan,
    result:                TrainingResult,
    gate_passed:           bool,
    endpoint_name:         str,
    human_impact_path:     str = "local_artifacts/human_impact.json",
    debate_path:           str = "local_artifacts/agent_debate.json",
    output_path:           Optional[str] = None,
    model:                 str = "claude-sonnet-4-20250514",
) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load human_impact.json
    hi_path = Path(human_impact_path)
    if hi_path.exists():
        human_impact = json.loads(hi_path.read_text())
        logger.info(
            "Storyteller — human_impact: is_degenerate=%s  FP=%d  FN=%d",
            human_impact["is_degenerate"],
            human_impact["false_positives"],
            human_impact["false_negatives"],
        )
    else:
        logger.warning("human_impact.json not found at %s", hi_path)
        human_impact = None

    # Load debate transcript
    db_path = Path(debate_path)
    debate_summary = None
    if db_path.exists():
        transcript = json.loads(db_path.read_text())
        debate_summary = "\n".join(
            f"[Round {e.get('round','?')}] {e['speaker'].replace('_',' ')}: {e['message']}"
            for e in transcript
        )
        logger.info("Storyteller — loaded debate (%d exchanges)", len(transcript))

    # CRITICAL LOGIC: choose report mode
    is_degenerate = human_impact.get("is_degenerate", False) if human_impact else False
    mitigated_dir = (
        human_impact.get("mitigated_dir", result.mitigated_bias.disparate_impact_ratio)
        if human_impact else result.mitigated_bias.disparate_impact_ratio
    )
    dir_failure   = mitigated_dir < DIR_REJECTION_THRESHOLD
    rejection_mode = is_degenerate or dir_failure

    if rejection_mode:
        system_prompt = _SYSTEM_PROMPT_REJECTION
        mode_label = (
            "DEGENERATE MODEL (majority-class collapse)"
            if is_degenerate else
            f"DISPARATE IMPACT FAILURE (DIR={mitigated_dir:.4f} < {DIR_REJECTION_THRESHOLD})"
        )
        logger.warning("Storyteller → REJECTION mode: %s", mode_label)
    else:
        system_prompt = _SYSTEM_PROMPT_STANDARD
        mode_label = "STANDARD"
        logger.info("Storyteller → STANDARD scorecard mode")

    payload = {
        "report_mode":      "REJECTION_AUDIT" if rejection_mode else "COMPLIANCE_SCORECARD",
        "rejection_reason": mode_label if rejection_mode else None,
        "dataset": {
            "name":               contract.dataset_name,
            "version_tag":        contract.version_tag,
            "protected_attr":     plan.protected_attribute,
            "privileged_group":   plan.privileged_group,
            "unprivileged_group": plan.unprivileged_group,
            "mitigation_method":  str(plan.method),
        },
        "fairness_metrics": {
            "baseline": {
                "accuracy_pct": round(result.baseline_accuracy * 100, 2),
                "eod": result.baseline_bias.equal_opportunity_diff,
                "aod": result.baseline_bias.average_odds_diff,
                "dir": result.baseline_bias.disparate_impact_ratio,
                "spd": result.baseline_bias.statistical_parity_diff,
            },
            "mitigated": {
                "accuracy_pct": round(result.mitigated_accuracy * 100, 2),
                "eod": result.mitigated_bias.equal_opportunity_diff,
                "aod": result.mitigated_bias.average_odds_diff,
                "dir": result.mitigated_bias.disparate_impact_ratio,
                "spd": result.mitigated_bias.statistical_parity_diff,
            },
            "accuracy_drop_pct": round(
                (result.baseline_accuracy - result.mitigated_accuracy) * 100, 3
            ),
            "eod_min": plan.fairness_thresholds[0].min_value if plan.fairness_thresholds else -0.05,
            "eod_max": plan.fairness_thresholds[0].max_value if plan.fairness_thresholds else  0.05,
            "gate_passed": gate_passed,
        },
        "human_impact":      human_impact,
        "debate_transcript": debate_summary,
        "endpoint_name":     endpoint_name,
        "run_id":            result.run_id,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
    }

    user_message = (
        f"Generate the {'Critical Audit Report' if rejection_mode else 'Fairness Scorecard'}. "
        f"Use ONLY the numbers in this payload:\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    logger.info("Storyteller calling GitHub Models (gpt-4o-mini) [%s mode]…", mode_label)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=2500,
    )
    markdown = response.choices[0].message.content.strip()

    banner = (
        f"<!-- FairGuard Auto-Generated Report -->\n"
        f"<!-- Mode: {'REJECTION AUDIT' if rejection_mode else 'COMPLIANCE SCORECARD'} -->\n"
        f"<!-- Run: {result.run_id} | Dataset: {contract.version_tag} -->\n"
        f"<!-- Timestamp: {datetime.utcnow().isoformat()}Z -->\n"
        f"<!-- is_degenerate: {is_degenerate} | dir: {mitigated_dir:.4f} -->\n\n"
    )
    full_doc = banner + markdown

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(full_doc)
        logger.info("Scorecard → %s [%s]", output_path, "REJECTION" if rejection_mode else "STANDARD")

    return full_doc


if __name__ == "__main__":
    from src.shared.contracts import load_contract, load_mitigation_plan
    contract = load_contract("local_artifacts/dataset_contract.json")
    plan     = load_mitigation_plan("local_artifacts/mitigation_plan.json")
    result   = TrainingResult.model_validate_json(
        Path("local_artifacts/training_result.json").read_text()
    )
    md = run_storyteller_agent(
        contract=contract, plan=plan, result=result,
        gate_passed=True, endpoint_name="local-dev-endpoint",
        human_impact_path="local_artifacts/human_impact.json",
        debate_path="local_artifacts/agent_debate.json",
        output_path="local_artifacts/fairness_scorecard.md",
    )
    print(md)