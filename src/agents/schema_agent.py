"""
agents/schema_agent.py
───────────────────────
Phase 1.3 — The Schema Mapper Agent.

Receives a statistical profile of the raw dataset (cardinalities, dtypes,
missing-value percentages, sample values) and asks Claude to produce a
strict DatasetContract JSON.  The agent is deliberately narrow: it may
only output JSON that passes Pydantic validation; it cannot free-think
or return prose.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

from src.shared.contracts import DatasetContract, save_model

logger = logging.getLogger(__name__)

# ── System prompt (hard-coded guard-rail) ────────────────────────────────────
_SYSTEM_PROMPT = """
You are the FairGuard Schema Mapper — a strict, machine-readable JSON emitter.

Your ONLY job is to analyse a statistical profile of a tabular dataset and
output a single valid JSON object that conforms EXACTLY to the DatasetContract
schema described below.

Rules you MUST obey:
1. Output ONLY raw JSON — no markdown fences, no prose, no comments.
2. Identify the most likely binary classification TARGET variable.
3. Mark every column that encodes demographic information (gender, race,
   age group, religion, nationality, etc.) as is_protected: true and list
   its known categories in protected_groups.
4. Never invent column names that are not in the profile.
5. If you are uncertain about a field, choose the most conservative option.

DatasetContract schema (field names and types):
{
  "schema_version":       "1.0",                  // string literal
  "dataset_name":         "<name>",               // string
  "dataset_hash":         "<sha256>",             // string
  "version_tag":          "<tag>",                // string, e.g. "ilpd_v1"
  "row_count":            <int>,
  "column_count":         <int>,
  "target_variable":      "<col>",
  "positive_label":       <any>,                  // the positive class value
  "protected_attributes": ["<col>", ...],         // at least one
  "features": [
    {
      "name":             "<col>",
      "dtype":            "numeric"|"categorical"|"binary"|"text"|"datetime",
      "cardinality":      <int|null>,
      "missing_pct":      <float 0-1>,
      "is_protected":     <bool>,
      "protected_groups": [<str>, ...]|null,
      "notes":            "<str>"|null
    },
    ...
  ],
  "notes": "<str>"|null
}

Do NOT include target_variable inside the features list.
"""


def run_schema_agent(
    statistical_profile: Dict[str, Any],
    dataset_name: str,
    dataset_hash: str,
    version_tag:  str,
    output_path:  Optional[str] = None,
    model:        str = "claude-sonnet-4-20250514",
) -> DatasetContract:
    """
    Call Claude to generate a DatasetContract from a statistical profile.

    Parameters
    ----------
    statistical_profile :
        Output of the deterministic profiler (cardinalities, missing pcts,
        sample values, dtypes).
    dataset_name, dataset_hash, version_tag :
        Metadata injected into the contract.
    output_path :
        If provided, the contract is saved as JSON to this path.
    model :
        Anthropic model string (always use claude-sonnet-4-20250514).

    Returns
    -------
    DatasetContract — fully validated Pydantic model.
    """
    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise ValueError("GITHUB_PAT environment variable is missing.")

    # 1. Define the user message FIRST
    user_message = (
        f"Dataset name   : {dataset_name}\n"
        f"SHA-256 hash   : {dataset_hash}\n"
        f"Version tag    : {version_tag}\n\n"
        f"Statistical profile:\n{json.dumps(statistical_profile, indent=2)}\n\n"
        "Generate the DatasetContract JSON now."
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
    logger.info("Schema Agent — calling Claude via GitHub Models...")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"].strip()
    logger.debug("Schema Agent raw response:\n%s", raw_text)

    # Strip accidental markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        contract_dict = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Schema Agent returned non-JSON output:\n%s", raw_text)
        raise ValueError(f"Schema Agent did not return valid JSON: {exc}") from exc

    # Inject authoritative metadata (agent cannot override these)
    # Inject authoritative metadata (agent cannot override these)
    contract_dict["dataset_hash"] = dataset_hash
    contract_dict["version_tag"]  = version_tag
    contract_dict["dataset_name"] = dataset_name
    contract_dict["agent_model"]  = model

    # FORCE-FIX: Remove target variable from features if the LLM hallucinated it
    target_var = contract_dict.get("target_variable")
    if target_var and "features" in contract_dict:
        contract_dict["features"] = [
            f for f in contract_dict["features"] 
            if f.get("name") != target_var
        ]

    try:
        contract = DatasetContract.model_validate(contract_dict)
    except Exception as exc:
        logger.error("DatasetContract validation failed: %s\nRaw dict: %s", exc, contract_dict)
        raise

    if output_path:
        save_model(contract, output_path)
        logger.info("DatasetContract written → %s", output_path)

    return contract


# ── CLI entry-point (for local testing) ─────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib
    profile_path = sys.argv[1] if len(sys.argv) > 1 else "profile.json"
    profile = json.loads(pathlib.Path(profile_path).read_text())
    contract = run_schema_agent(
        statistical_profile=profile,
        dataset_name="ilpd",
        dataset_hash="abc123",
        version_tag="ilpd_v1",
        output_path="dataset_contract.json",
    )
    print(contract.model_dump_json(indent=2))