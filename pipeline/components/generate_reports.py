"""
pipeline/components/generate_reports.py
────────────────────────────────────────
Phase 5.2 — Pitch Artefacts Generator.

Produces:
  • fairness_scorecard.md   (via Storyteller Agent)
  • FairGuard_WIT_Demo.ipynb (What-If Tool pre-wired to the live Endpoint)
  • app.py                   (Streamlit dashboard — copied from template)
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run_generate_reports(
    project:          str,
    location:         str,
    version_tag:      str,
    contract_gcs_uri: str,
    plan_gcs_uri:     str,
    result_gcs_uri:   str,
    endpoint_resource:str,
    gate_passed:      bool,
    artifacts_bucket: str,
) -> dict:
    """
    Generate all presentation artefacts and upload to GCS.

    Returns URIs for scorecard, WIT notebook, and Streamlit app.
    """
    from src.shared.gcp_utils import gcs_read_string, gcs_upload_string, gcs_upload
    from src.shared.contracts import (
        DatasetContract, MitigationPlan, TrainingResult,
    )
    from src.agents.storyteller_agent import run_storyteller_agent

    # ── Load artefacts ────────────────────────────────────────────────────────
    contract = DatasetContract.model_validate_json(gcs_read_string(contract_gcs_uri))
    plan     = MitigationPlan.model_validate_json(gcs_read_string(plan_gcs_uri))
    result   = TrainingResult.model_validate_json(gcs_read_string(result_gcs_uri))

    endpoint_name = endpoint_resource.split("/")[-1]

    # ── 1. Fairness Scorecard ─────────────────────────────────────────────────
    scorecard_md = run_storyteller_agent(
        contract=contract,
        plan=plan,
        result=result,
        gate_passed=gate_passed,
        endpoint_name=endpoint_name,
    )
    scorecard_gcs = (
        f"gs://{artifacts_bucket}/reports/{version_tag}/fairness_scorecard.md"
    )
    gcs_upload_string(scorecard_md, scorecard_gcs, "text/markdown")
    logger.info("Fairness scorecard → %s", scorecard_gcs)

    # ── 2. WIT Notebook ───────────────────────────────────────────────────────
    wit_nb = _build_wit_notebook(
        project=project,
        location=location,
        endpoint_resource=endpoint_resource,
        contract=contract,
        result=result,
        version_tag=version_tag,
    )
    wit_gcs = (
        f"gs://{artifacts_bucket}/notebooks/{version_tag}/FairGuard_WIT_Demo.ipynb"
    )
    gcs_upload_string(json.dumps(wit_nb, indent=2), wit_gcs, "application/json")
    logger.info("WIT notebook → %s", wit_gcs)

    return {
        "scorecard_uri":    scorecard_gcs,
        "wit_notebook_uri": wit_gcs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WIT Notebook builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_wit_notebook(
    project:           str,
    location:          str,
    endpoint_resource: str,
    contract,
    result,
    version_tag:       str,
) -> dict:
    """Return a Jupyter notebook dict pre-wired to the live Vertex Endpoint."""

    protected_col = contract.protected_attributes[0]
    target_col    = contract.target_variable
    feature_cols  = [f.name for f in contract.features]

    cells = [
        _md_cell("# 🛡️ FairGuard — What-If Tool Demo\n"
                 f"**Dataset**: `{contract.dataset_name}` ({version_tag})  \n"
                 f"**Endpoint**: `{endpoint_resource}`  \n"
                 f"**Protected attribute**: `{protected_col}`  \n"
                 f"**Target**: `{target_col}`"),

        _code_cell(
            "# Install dependencies (run once)\n"
            "!pip install witwidget google-cloud-aiplatform pandas --quiet"
        ),

        _code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "from google.cloud import aiplatform\n"
            "from witwidget.notebook.visualization import WitConfigBuilder, WitWidget\n\n"
            f"PROJECT  = '{project}'\n"
            f"LOCATION = '{location}'\n"
            f"ENDPOINT_RESOURCE = '{endpoint_resource}'\n"
            f"PROTECTED_COL     = '{protected_col}'\n"
            f"TARGET_COL        = '{target_col}'\n"
            f"FEATURE_COLS      = {feature_cols}\n"
            f"BASELINE_ACCURACY = {result.baseline_accuracy:.4f}\n"
            f"MITIGATED_ACCURACY= {result.mitigated_accuracy:.4f}\n"
            f"BASELINE_EOD      = {result.baseline_bias.equal_opportunity_diff:.4f}\n"
            f"MITIGATED_EOD     = {result.mitigated_bias.equal_opportunity_diff:.4f}"
        ),

        _md_cell("## Load validated test data"),

        _code_cell(
            "# Replace with actual GCS path to your validated CSV\n"
            "# df = pd.read_csv('gs://your-project-data/validated/...')\n"
            "# For demo: generate synthetic data matching the schema\n"
            "np.random.seed(42)\n"
            "n = 200\n"
            "df = pd.DataFrame({\n"
            f"    col: np.random.randn(n) for col in FEATURE_COLS\n"
            "})\n"
            f"df[TARGET_COL]    = np.random.randint(0, 2, n)\n"
            f"df[PROTECTED_COL] = np.random.choice(['Group_A', 'Group_B'], n)\n"
            "print(df.head())"
        ),

        _md_cell("## Connect to the live Vertex AI Endpoint"),

        _code_cell(
            "aiplatform.init(project=PROJECT, location=LOCATION)\n"
            "endpoint = aiplatform.Endpoint(ENDPOINT_RESOURCE)\n\n"
            "def predict_fn(examples):\n"
            "    \"\"\"\n"
            "    WIT calls this function with a list of feature dicts.\n"
            "    We forward them to the live Vertex Endpoint.\n"
            "    \"\"\"\n"
            "    instances = [\n"
            "        {col: ex.get(col, 0.0) for col in FEATURE_COLS}\n"
            "        for ex in examples\n"
            "    ]\n"
            "    response = endpoint.predict(instances=instances)\n"
            "    return [[1 - p, p] for p in response.predictions]\n\n"
            "print('Endpoint connected ✅')"
        ),

        _md_cell("## Launch the What-If Tool"),

        _code_cell(
            "examples = df[FEATURE_COLS + [TARGET_COL, PROTECTED_COL]].to_dict('records')\n\n"
            "config = (\n"
            "    WitConfigBuilder(examples)\n"
            "    .set_custom_predict_fn(predict_fn)\n"
            f"    .set_target_feature(TARGET_COL)\n"
            f"    .set_label_vocab(['No Disease', 'Disease'])\n"
            ")\n\n"
            "WitWidget(config, height=800)"
        ),

        _md_cell(
            "## Fairness Metrics Summary\n\n"
            "| Metric | Baseline | Mitigated | Δ |\n"
            "|--------|----------|-----------|---|\n"
            f"| Accuracy | {result.baseline_accuracy:.4f} | {result.mitigated_accuracy:.4f} | "
            f"{(result.mitigated_accuracy - result.baseline_accuracy):+.4f} |\n"
            f"| EOD | {result.baseline_bias.equal_opportunity_diff:.4f} | "
            f"{result.mitigated_bias.equal_opportunity_diff:.4f} | "
            f"{(result.mitigated_bias.equal_opportunity_diff - result.baseline_bias.equal_opportunity_diff):+.4f} |\n"
            f"| AOD | {result.baseline_bias.average_odds_diff:.4f} | "
            f"{result.mitigated_bias.average_odds_diff:.4f} | "
            f"{(result.mitigated_bias.average_odds_diff - result.baseline_bias.average_odds_diff):+.4f} |\n"
            f"| DIR | {result.baseline_bias.disparate_impact_ratio:.4f} | "
            f"{result.mitigated_bias.disparate_impact_ratio:.4f} | "
            f"{(result.mitigated_bias.disparate_impact_ratio - result.baseline_bias.disparate_impact_ratio):+.4f} |"
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "fairguard": {
                "version_tag":       version_tag,
                "endpoint_resource": endpoint_resource,
                "generated_by":      "FairGuard Storyteller Agent",
            },
        },
        "cells": cells,
    }


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata":  {},
        "source":    source,
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type":       "code",
        "execution_count": None,
        "metadata":        {},
        "outputs":         [],
        "source":          source,
    }
