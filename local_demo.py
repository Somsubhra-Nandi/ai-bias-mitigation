"""
local_demo.py
─────────────
Runs the FairGuard pipeline locally without touching Google Cloud.
Saves all artifacts to a local `local_artifacts/` directory.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from src.agents.schema_agent import run_schema_agent
from src.agents.ethics_agent import run_ethics_agent
from src.agents.storyteller_agent import run_storyteller_agent
from src.shared.contracts import TrainingResult, BiasMetrics, save_model
from src.ml.task import load_data, train_baseline, predict_with_scaler, predict_proba_with_scaler
from src.ml.metrics import compute_metrics, accuracy
from src.ml.mitigators import KamiranCaldersReweighing, ThresholdOptimizer
from optuna_search import run_optuna_search

from src.ml.human_impact import compute_human_impact
from src.agents.debate_agent import run_debate_agent

# Load GitHub PAT from .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _build_statistical_profile(df: pd.DataFrame) -> dict:
    """Quick local profiler with strict dtype mapping."""
    columns = []
    for col in df.columns:
        series = df[col]
        
        # Map pandas dtypes to our strict enums before giving them to the LLM
        if pd.api.types.is_numeric_dtype(series):
            mapped_dtype = "numeric"
        elif series.nunique() == 2:
            mapped_dtype = "binary"
        else:
            mapped_dtype = "categorical"

        col_info = {
            "name": col, 
            "dtype": mapped_dtype,  # <--- Now feeding strict enums!
            "cardinality": int(series.nunique()),
            "missing_pct": round(float(series.isna().mean()), 4),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info["stats"] = {"min": float(desc["min"]), "max": float(desc["max"]), "mean": float(desc["mean"])}
        else:
            col_info["top_categories"] = list(series.value_counts().head(5).index.astype(str))
        columns.append(col_info)
        
    return {"row_count": len(df), "column_count": len(df.columns), "columns": columns}

def main():
    logger.info("🚀 Starting Local FairGuard Run...")
    
    # 1. Setup local directories
    out_dir = Path("local_artifacts")
    out_dir.mkdir(exist_ok=True)
    raw_csv_path = Path("data/indian_liver_patient.csv")
    
    if not raw_csv_path.exists():
        logger.error(f"Cannot find {raw_csv_path}! Please put the CSV there.")
        return

    # 2. Schema Agent
    logger.info("=== PHASE 1: Schema Agent ===")
    df = pd.read_csv(raw_csv_path)
    profile = _build_statistical_profile(df)
    contract = run_schema_agent(
        statistical_profile=profile,
        dataset_name="Indian Liver Patient Dataset",
        dataset_hash="local_hash_123",
        version_tag="ilpd_local",
        output_path=str(out_dir / "dataset_contract.json")
    )
    
    # 3. Ethics Agent
    logger.info("=== PHASE 2: Ethics Agent ===")
    plan, _ = run_ethics_agent(
        contract=contract,
        policy_path="ethics_policy.json",
        plan_output_path=str(out_dir / "mitigation_plan.json"),
        log_output_path=str(out_dir / "ethics_decision_log.md")
    )
    
    # 4. Local ML Training
    logger.info("=== PHASE 3: ML Training & Mitigation ===")
    X_df, y, sensitive, le, priv_value = load_data(raw_csv_path, contract)
    
    # Fill any missing values (NaNs) with the column median
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(float)
    
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, sensitive, test_size=0.2, random_state=42, stratify=y)
    
   # --- OPTUNA SEARCH with cache invalidation ---
    logger.info("=== OPTUNA SEARCH ===")
    optimal_path = "local_artifacts/optimal_hyperparameters.json"

    import hashlib
    csv_hash = hashlib.sha256(raw_csv_path.read_bytes()).hexdigest()

    _needs_search = True
    if Path(optimal_path).exists():
        try:
            _cached = json.loads(Path(optimal_path).read_text())
            if _cached.get("dataset_hash") == csv_hash:
                logger.info(
                    "Optuna cache is valid (dataset unchanged). Skipping search. "
                    "Delete %s to force a new search.", optimal_path
                )
                _needs_search = False
            else:
                logger.warning(
                    "Dataset hash changed — cached hyperparameters are stale. "
                    "Re-running Optuna search."
                )
        except Exception as e:
            logger.warning("Could not read cached hyperparameters (%s). Re-running.", e)

    if _needs_search:
        logger.info("Running Optuna search (~60 s for 50 trials) ...")
        run_optuna_search(
            X_train=X_tr, y_train=y_tr, sensitive_features=s_tr,
            priv_value=priv_value, n_trials=50, output_path=optimal_path,
            dataset_hash=csv_hash,
        )
    # ---------------------------------------------

    # Baseline (Now automatically uses the Optuna Winner!)
    base_clf = train_baseline(X_tr, y_tr, hyperparams_path=Path(optimal_path))
    y_pred_base = predict_with_scaler(base_clf, X_te)
    base_acc = accuracy(y_te, y_pred_base)
    base_metrics = compute_metrics(y_te, y_pred_base, s_te, priv_value=priv_value)
    
    # Mitigate
    logger.info(f"Applying Agent Strategy: {plan.method.value}")
    if plan.method.value == "reweighing":
        reweigher = KamiranCaldersReweighing(sensitive_col=plan.protected_attribute, priv_value=priv_value)
        sample_weights = reweigher.fit(X_tr, y_tr, s_tr).transform(s_tr, y_tr)
        mit_clf = train_baseline(X_tr, y_tr, sample_weight=sample_weights, hyperparams_path=Path(optimal_path))
        y_pred_mit = predict_with_scaler(mit_clf, X_te)
    else:
        y_prob = predict_proba_with_scaler(base_clf, X_te)[:, 1]
        optimizer = ThresholdOptimizer(sensitive_col=plan.protected_attribute, priv_value=priv_value, max_accuracy_drop=plan.max_accuracy_drop_pct / 100)
        y_pred_mit = optimizer.fit(y_te, y_prob, s_te, baseline_accuracy=base_acc).predict(y_prob, s_te)

    mit_acc = accuracy(y_te, y_pred_mit)
    mit_metrics = compute_metrics(y_te, y_pred_mit, s_te, priv_value=priv_value)
    
    # ── Phase 3.5: Degenerate Detector + Human Impact ─────────────────────────
    logger.info("=== PHASE 3.5: Degenerate Detector & Human Impact ===")
    human_impact = compute_human_impact(
        y_true=y_te,
        y_pred_mitigated=y_pred_mit,
        mitigated_dir=mit_metrics.dir,
        output_path=str(out_dir / "human_impact.json"),
    )
    
    # Save Results
    result = TrainingResult(
        experiment_id="local_dev", run_id="local_run_001", dataset_hash="local_hash_123",
        baseline_accuracy=base_acc, mitigated_accuracy=mit_acc,
        baseline_bias=BiasMetrics(eod=base_metrics.eod, aod=base_metrics.aod, dir=base_metrics.dir, spd=base_metrics.spd),
        mitigated_bias=BiasMetrics(eod=mit_metrics.eod, aod=mit_metrics.aod, dir=mit_metrics.dir, spd=mit_metrics.spd),
        model_gcs_path="local", train_indices_path="local", test_indices_path="local"
    )
    save_model(result, str(out_dir / "training_result.json"))

    # ── Phase 3.7: Multi-Agent Debate ─────────────────────────────────────────
    logger.info("=== PHASE 3.7: Multi-Agent Debate ===")
    try:
        debate = run_debate_agent(
            human_impact_path=str(out_dir / "human_impact.json"),
            training_result_path=str(out_dir / "training_result.json"),
            output_path=str(out_dir / "agent_debate.json"),
        )
    except Exception as e:
        logger.warning(f"Debate agent failed: {e}")
        
    # 5. Storyteller Agent
    logger.info("=== PHASE 4: Storyteller Agent ===")
    run_storyteller_agent(
        contract=contract, plan=plan, result=result, gate_passed=True,
        endpoint_name="local-dev-endpoint", 
        human_impact_path=str(out_dir / "human_impact.json"), # NEW
        debate_path=str(out_dir / "agent_debate.json"),       # NEW
        output_path=str(out_dir / "fairness_scorecard.md")
    )
    
    logger.info(f"✅ Local pipeline complete! All files saved to {out_dir.absolute()}")

if __name__ == "__main__":
    main()