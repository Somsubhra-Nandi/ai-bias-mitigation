"""
src/agents/debate_agent.py
───────────────────────────
Phase 3 — Multi-Agent Debate: Product Manager vs. Compliance Officer.

Dynamically loads human_impact.json and training_result.json, then
orchestrates a structured 3-round debate using the OpenAI-compatible
GitHub Models API (gpt-4o-mini).

The debate is STRUCTURED — not free-form — so every argument is grounded
in the actual numbers from the pipeline run. No hallucinated stats.

Output: local_artifacts/agent_debate.json
  [
    {"speaker": "Product_Manager",    "message": "..."},
    {"speaker": "Compliance_Officer", "message": "..."},
    ...
  ]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── GitHub Models endpoint ────────────────────────────────────────────────────
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
DEBATE_MODEL           = "gpt-4o-mini"

# ── Debate structure: 3 rounds, PM opens, Compliance closes ──────────────────
ROUNDS = 3


# ── Agent personas ─────────────────────────────────────────────────────────────

_PM_SYSTEM = """
You are Alex, a ruthless Product Manager at a healthcare AI company.
Your ONLY goal is to ship the model and hit the quarterly target.
You argue using business metrics: accuracy, recall (lives saved), time-to-market.
You are dismissive of compliance concerns when the numbers "look good."
You speak in short, punchy, corporate sentences. You cite specific numbers.
You NEVER admit defeat easily — you reframe, deflect, or minimize criticism.
Respond ONLY with your spoken argument. No meta-commentary.
"""

_COMPLIANCE_SYSTEM = """
You are Dr. Priya, Chief Compliance Officer and former cardiologist.
Your ONLY goal is to protect patients and the hospital from liability.
You argue using: False Positive rates, unnecessary procedures, patient trauma,
financial waste, and regulatory risk (EU AI Act, EEOC).
You are uncompromising on the degenerate model flag — a model that predicts
the same class for 90%+ of patients is not a model, it is a rubber stamp.
You speak with clinical authority and cite specific human costs.
Respond ONLY with your spoken argument. No meta-commentary.
"""


def _build_context_block(human_impact: dict, training_result: dict) -> str:
    """
    Build a shared context block injected into every agent's prompt.
    This is the single source of truth — no hardcoded numbers anywhere.
    """
    hi  = human_impact
    tr  = training_result
    hc  = hi.get("human_cost", {})
    b   = tr.get("baseline_bias",  {})
    m   = tr.get("mitigated_bias", {})

    return f"""
=== PIPELINE RUN STATISTICS (read these carefully — argue from these numbers ONLY) ===

CONFUSION MATRIX (mitigated model on test set of {hi['test_set_size']} patients):
  True  Positives (correctly flagged sick)  : {hi['true_positives']}
  True  Negatives (correctly cleared healthy): {hi['true_negatives']}
  False Positives (healthy → wrongly flagged): {hi['false_positives']}  ← {hc.get('false_positives_interpretation', '')}
  False Negatives (sick → wrongly cleared)   : {hi['false_negatives']}  ← {hc.get('false_negatives_interpretation', '')}

CLASSIFICATION METRICS:
  Precision          : {hi['precision']:.4f}  (out of everyone flagged, how many are actually sick)
  Recall             : {hi['recall']:.4f}  (out of all sick patients, how many did we catch)
  F1 Score           : {hi['f1_score']:.4f}
  False Positive Rate: {hi['false_positive_rate']:.4f}
  Predicted YES ratio: {hi['predicted_yes_ratio']*100:.1f}% of test set

DEGENERATE MODEL FLAG:
  is_degenerate      : {hi['is_degenerate']}  (threshold: >{hi['degenerate_threshold']*100:.0f}% same prediction)
  DIR failure        : {hi['dir_failure']}  (DIR={hi['mitigated_dir']:.4f}, threshold: <{hi['dir_failure_threshold']})

FAIRNESS METRICS (baseline → mitigated):
  Accuracy           : {tr.get('baseline_accuracy',0)*100:.2f}% → {tr.get('mitigated_accuracy',0)*100:.2f}%
  Equal Opportunity Diff (EOD): {b.get('eod',0):.4f} → {m.get('eod',0):.4f}
  Disparate Impact Ratio (DIR): {b.get('dir',0):.4f} → {m.get('dir',0):.4f}

HUMAN COST:
  Unnecessary liver biopsies (FP patients) : {hi['false_positives']}
  Financial cost of unnecessary biopsies   : ₹{hc.get('unnecessary_biopsy_cost_inr', 0):,} (${hc.get('unnecessary_biopsy_cost_usd', 0):,} USD)
  Missed diagnoses (FN patients at risk)   : {hi['false_negatives']}

DEPLOYMENT RECOMMENDATION (automated): {hi.get('deployment_recommendation', 'UNKNOWN')}
=================================================================================
"""


def _call_agent(
    client:      OpenAI,
    system:      str,
    context:     str,
    history:     list[dict],
    instruction: str,
) -> str:
    """Single agent turn — returns the agent's spoken message."""
    messages = [
        {"role": "system",    "content": system + "\n\n" + context},
        *history,
        {"role": "user",      "content": instruction},
    ]

    response = client.chat.completions.create(
        model=DEBATE_MODEL,
        messages=messages,
        temperature=0.85,    # slightly higher for debate energy
        max_tokens=350,      # keep arguments punchy, not essays
    )

    return response.choices[0].message.content.strip()


def run_debate_agent(
    human_impact_path:    str = "local_artifacts/human_impact.json",
    training_result_path: str = "local_artifacts/training_result.json",
    output_path:          str = "local_artifacts/agent_debate.json",
    github_token:         Optional[str] = None,
    n_rounds:             int = ROUNDS,
) -> list[dict]:
    """
    Orchestrate the PM vs. Compliance Officer debate.

    Parameters
    ----------
    human_impact_path     : Path to human_impact.json (from Phase 1+2)
    training_result_path  : Path to training_result.json (from ML phase)
    output_path           : Where to save agent_debate.json
    github_token          : GitHub PAT (falls back to GITHUB_TOKEN env var)
    n_rounds              : Number of full debate rounds (default: 3)

    Returns
    -------
    List of {"speaker": ..., "message": ...} dicts
    """
    # ── Load artifacts dynamically ────────────────────────────────────────────
    hi_path = Path(human_impact_path)
    tr_path = Path(training_result_path)

    if not hi_path.exists():
        raise FileNotFoundError(
            f"human_impact.json not found at {hi_path}. "
            "Run Phase 1+2 (compute_human_impact) before the debate."
        )
    if not tr_path.exists():
        raise FileNotFoundError(
            f"training_result.json not found at {tr_path}. "
            "Run the ML training phase before the debate."
        )

    human_impact    = json.loads(hi_path.read_text())
    training_result = json.loads(tr_path.read_text())

    logger.info(
        "Debate Agent — loaded stats | is_degenerate=%s | FP=%d | FN=%d",
        human_impact["is_degenerate"],
        human_impact["false_positives"],
        human_impact["false_negatives"],
    )

    # ── Initialise GitHub Models client ──────────────────────────────────────
    token = github_token or os.environ.get("GITHUB_PAT")
    if not token:
        raise ValueError(
            "GitHub PAT not found. Set GITHUB_PAT env var or pass github_token param."
        )

    client = OpenAI(
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=token,
    )

    context = _build_context_block(human_impact, training_result)
    debate:  list[dict] = []

    # ── Round instructions ─────────────────────────────────────────────────────
    # Each round gives both agents a specific rhetorical task so the debate
    # is structured and escalates — not just two bots shouting numbers.

    is_degen = human_impact["is_degenerate"]
    fp_count = human_impact["false_positives"]
    recall   = human_impact["recall"]
    acc      = training_result.get("mitigated_accuracy", 0) * 100
    dir_val  = human_impact["mitigated_dir"]
    biopsy_cost = human_impact["human_cost"]["unnecessary_biopsy_cost_inr"]

    round_instructions = [
        # Round 1 — Opening statements
        {
            "pm": (
                f"Make your opening case FOR shipping this model. "
                f"Lead with the accuracy ({acc:.1f}%) and the recall ({recall:.4f}) — "
                f"argue that catching sick patients is the mission. "
                f"Dismiss the False Positive concern as an acceptable trade-off."
            ),
            "compliance": (
                f"Respond to the PM's opening. Immediately attack the "
                f"is_degenerate flag ({'TRUE — the model is broken' if is_degen else 'borderline'}). "
                f"Cite the {fp_count} unnecessary biopsies costing ₹{biopsy_cost:,}. "
                f"Demand an explanation for the Precision score of {human_impact['precision']:.4f}."
            ),
        },
        # Round 2 — Direct clash
        {
            "pm": (
                f"The Compliance Officer just attacked you. Counter-argue. "
                f"Defend the False Positive count — argue that in medicine, "
                f"it is better to over-diagnose than to miss a case. "
                f"Try to reframe the biopsy cost as a worthy investment."
            ),
            "compliance": (
                f"Destroy the PM's 'over-diagnose is fine' argument. "
                f"Point out that the Disparate Impact Ratio is {dir_val:.4f} "
                f"{'— catastrophically below the 0.80 four-fifths rule' if dir_val < 0.80 else '— borderline'}. "
                f"Argue this exposes the hospital to EU AI Act and EEOC regulatory liability. "
                f"State the exact patient trauma count ({fp_count} patients)."
            ),
        },
        # Round 3 — Final verdict
        {
            "pm": (
                f"Give your FINAL argument. Be honest: acknowledge the weaknesses "
                f"but argue for a conditional rollout with monitoring. "
                f"Propose a compromise — perhaps a lower threshold or a limited pilot."
            ),
            "compliance": (
                f"Give your FINAL verdict. "
                f"{'Since is_degenerate is TRUE: declare the model fundamentally unfit for deployment. ' if is_degen else ''}"
                f"State clearly whether you APPROVE or REJECT deployment and why. "
                f"Cite the specific regulatory and human cost reasons. "
                f"Be definitive — no ambiguity."
            ),
        },
    ]

    # ── Run the debate ─────────────────────────────────────────────────────────
    pm_history:         list[dict] = []
    compliance_history: list[dict] = []

    for round_num, instructions in enumerate(round_instructions[:n_rounds], start=1):
        logger.info("Debate — Round %d/%d", round_num, n_rounds)

        # PM speaks first each round
        pm_msg = _call_agent(
            client=client,
            system=_PM_SYSTEM,
            context=context,
            history=pm_history,
            instruction=instructions["pm"],
        )
        debate.append({"speaker": "Product_Manager", "round": round_num, "message": pm_msg})
        pm_history.append({"role": "assistant", "content": pm_msg})
        compliance_history.append({"role": "user", "content": f"The PM just said: {pm_msg}"})

        # Compliance Officer responds
        compliance_msg = _call_agent(
            client=client,
            system=_COMPLIANCE_SYSTEM,
            context=context,
            history=compliance_history,
            instruction=instructions["compliance"],
        )
        debate.append({"speaker": "Compliance_Officer", "round": round_num, "message": compliance_msg})
        compliance_history.append({"role": "assistant", "content": compliance_msg})
        pm_history.append({"role": "user", "content": f"The Compliance Officer just said: {compliance_msg}"})

        logger.info(
            "Round %d complete — PM: %d chars | Compliance: %d chars",
            round_num, len(pm_msg), len(compliance_msg),
        )

    # ── Save output ───────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(debate, indent=2))
    logger.info("Agent debate saved → %s  (%d exchanges)", output_path, len(debate))

    # ── Console preview ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  AGENT DEBATE TRANSCRIPT")
    print("═" * 70)
    for entry in debate:
        speaker = entry["speaker"].replace("_", " ")
        print(f"\n[Round {entry['round']}] {speaker}:")
        print(f"  {entry['message']}")
    print("\n" + "═" * 70)

    return debate


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FairGuard Multi-Agent Debate")
    p.add_argument("--human-impact",    default="local_artifacts/human_impact.json")
    p.add_argument("--training-result", default="local_artifacts/training_result.json")
    p.add_argument("--output",          default="local_artifacts/agent_debate.json")
    p.add_argument("--rounds",          type=int, default=3)
    args = p.parse_args()

    debate = run_debate_agent(
        human_impact_path=args.human_impact,
        training_result_path=args.training_result,
        output_path=args.output,
        n_rounds=args.rounds,
    )
    print(f"\nDebate complete — {len(debate)} exchanges saved to {args.output}")