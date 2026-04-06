"""
ml/task.py
───────────
Entrypoint for the Vertex AI CustomTrainingJob container.

Execution environment: Google's pre-built Scikit-learn container.
All heavy compute runs here; your local machine does zero work.

Phases executed inside this script:
  3.1  Fix random seeds, log dataset hash.
  3.2  Baseline training + bias extraction.
  3.3  Algorithmic mitigation (reweighing or threshold optimizer).
  3.4  Full MLflow / Vertex Experiments audit trail.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.ml.metrics import compute_metrics, accuracy, classification_report_dict, FairnessMetrics
from src.ml.mitigators import KamiranCaldersReweighing, ThresholdOptimizer, get_mitigator
from optuna_search import build_model_from_params
from src.shared.contracts import (
    DatasetContract,
    MitigationPlan,
    MitigationMethod,
    TrainingResult,
    BiasMetrics,
    load_contract,
    load_mitigation_plan,
    save_model,
)
from src.shared.gcp_utils import gcs_download, gcs_upload, sha256_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED              = 42
TEST_SIZE         = 0.20
EXPERIMENT_NAME   = "fairguard-debiasing"
MODEL_ARTIFACT    = "model.joblib"
OPTIMAL_HYPERPARAMS_PATH = Path("local_artifacts/optimal_hyperparameters.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FairGuard Training Job")
    p.add_argument("--project",             required=True)
    p.add_argument("--location",            default="us-central1")
    p.add_argument("--data-gcs-uri",        required=True,
                   help="gs:// URI of the validated CSV")
    p.add_argument("--contract-gcs-uri",    required=True,
                   help="gs:// URI of dataset_contract.json")
    p.add_argument("--plan-gcs-uri",        required=True,
                   help="gs:// URI of mitigation_plan.json")
    p.add_argument("--output-gcs-uri",      required=True,
                   help="gs:// directory for model.joblib + result.json")
    p.add_argument("--run-name",            default=None)
    return p.parse_args()


def setup_mlflow(project: str, location: str, run_name: str) -> str:
    """Configure MLflow → Vertex AI Experiments backend."""
    tracking_uri = f"vertex-ai://{location}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI set to %s", tracking_uri)

    mlflow.start_run(run_name=run_name or f"fairguard-{SEED}")
    return mlflow.active_run().info.run_id


def load_data(
    data_path: str | Path,
    contract:  DatasetContract,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load CSV, encode categoricals, split features / target / sensitive.
    Returns (X_df, y, sensitive_array, label_encoder_for_protected_col).
    """
    df = pd.read_csv(data_path)
    logger.info("Loaded data: %d rows, %d cols", *df.shape)

    target_col    = contract.target_variable
    protected_col = contract.protected_attributes[0]   # primary protected attr

    # Encode protected attribute to numeric
    le = LabelEncoder()
    df[protected_col + "_encoded"] = le.fit_transform(df[protected_col].astype(str))
    priv_encoded = int(le.transform([str(contract.features[
        next(i for i,f in enumerate(contract.features) if f.name == protected_col)
    ].protected_groups[0])])[0])

    # Encode all categorical features
    feature_cols = [f.name for f in contract.features if f.name != protected_col]
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[feature_cols].values.astype(float)
    y = (df[target_col].values == contract.positive_label).astype(int)
    sensitive = df[protected_col + "_encoded"].values.astype(int)

    return df[feature_cols], y, sensitive, le, priv_encoded


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    hyperparams_path: Path | None = None,
) -> Any:
    """
    Train the best model found by Optuna, or fall back to GradientBoosting.

    If `hyperparams_path` is provided and the file exists, the Optuna winner
    is instantiated via build_model_from_params().  Otherwise falls back to
    the original hardcoded GradientBoostingClassifier so the pipeline never
    breaks even when Optuna hasn't been run yet.
    """
    from typing import Any

    hp_path = hyperparams_path or OPTIMAL_HYPERPARAMS_PATH

    if hp_path.exists():
        try:
            params = json.loads(hp_path.read_text())
            clf = build_model_from_params(params)
            logger.info(
                "Loaded Optuna winner: model=%s  cv_acc=%.4f  cv_|EOD|=%.4f",
                params["model_name"],
                params.get("cv_accuracy", float("nan")),
                params.get("cv_abs_eod",  float("nan")),
            )
        except Exception as exc:
            logger.warning(
                "Failed to load Optuna hyperparameters (%s) — falling back to GBM.", exc
            )
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.8, random_state=SEED,
            )
    else:
        logger.info(
            "No optimal_hyperparameters.json found at %s — using default GBM. "
            "Run optuna_search.py first for best results.", hp_path,
        )
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=SEED,
        )

    # StandardScaler is needed for distance-based & linear models
    # It is a no-op for tree-based models (sklearn ignores it without error)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf.fit(X_scaled, y_train, **_sample_weight_kwargs(clf, sample_weight))
    # Attach scaler so predict() can use it later
    clf._fairguard_scaler = scaler
    return clf


def _sample_weight_kwargs(clf, sample_weight):
    """Return sample_weight dict only if the classifier supports it."""
    if sample_weight is None:
        return {}
    try:
        import inspect
        sig = inspect.signature(clf.fit)
        if "sample_weight" in sig.parameters:
            return {"sample_weight": sample_weight}
    except Exception:
        pass
    return {}


def predict_with_scaler(clf, X: np.ndarray) -> np.ndarray:
    """Wrapper that applies the attached scaler before predict()."""
    scaler = getattr(clf, "_fairguard_scaler", None)
    X_in = scaler.transform(X) if scaler is not None else X
    return clf.predict(X_in)


def predict_proba_with_scaler(clf, X: np.ndarray) -> np.ndarray:
    """Wrapper that applies the attached scaler before predict_proba()."""
    scaler = getattr(clf, "_fairguard_scaler", None)
    X_in = scaler.transform(X) if scaler is not None else X
    return clf.predict_proba(X_in)


def main() -> None:
    args = parse_args()

    # ── 3.1 Fix seeds ────────────────────────────────────────────────────────
    np.random.seed(SEED)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── Download artefacts from GCS ──────────────────────────────────────
        data_path     = gcs_download(args.data_gcs_uri,     tmpdir / "data.csv")
        contract_path = gcs_download(args.contract_gcs_uri, tmpdir / "contract.json")
        plan_path     = gcs_download(args.plan_gcs_uri,     tmpdir / "plan.json")

        contract = load_contract(contract_path)
        plan     = load_mitigation_plan(plan_path)

        # Verify dataset hash matches contract
        file_hash = sha256_file(data_path)
        if file_hash != contract.dataset_hash:
            raise ValueError(
                f"Dataset hash mismatch! "
                f"Expected {contract.dataset_hash}, got {file_hash}."
            )

        run_id = setup_mlflow(args.project, args.location, args.run_name)
        logger.info("MLflow run_id: %s", run_id)

        mlflow.log_param("seed",           SEED)
        mlflow.log_param("optuna_winner",   str(OPTIMAL_HYPERPARAMS_PATH.exists()))
        mlflow.log_param("dataset_hash",   contract.dataset_hash)
        mlflow.log_param("version_tag",    contract.version_tag)
        mlflow.log_param("mitigation",     plan.method)
        mlflow.log_param("protected_attr", plan.protected_attribute)

        # ── Load & split data ────────────────────────────────────────────────
        X_df, y, sensitive, le, priv_value = load_data(data_path, contract)
        X = X_df.values.astype(float)

        (X_tr, X_te, y_tr, y_te,
         s_tr, s_te) = train_test_split(
            X, y, sensitive,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y,
        )

        # Save indices for reproducibility
        train_idx_path = tmpdir / "train_indices.npy"
        test_idx_path  = tmpdir / "test_indices.npy"
        np.save(train_idx_path, np.where(np.isin(np.arange(len(y)), np.arange(len(y_tr))))[0])
        np.save(test_idx_path,  np.where(np.isin(np.arange(len(y)), np.arange(len(y_tr), len(y))))[0])

        # ── 3.2 Baseline training ─────────────────────────────────────────────
        logger.info("Training baseline model…")
        baseline_clf = train_baseline(X_tr, y_tr)
        y_pred_base  = predict_with_scaler(baseline_clf, X_te)

        base_acc     = accuracy(y_te, y_pred_base)
        base_metrics = compute_metrics(y_te, y_pred_base, s_te, priv_value=priv_value)

        mlflow.log_metric("baseline_accuracy", base_acc)
        mlflow.log_metric("baseline_eod", base_metrics.eod)
        mlflow.log_metric("baseline_aod", base_metrics.aod)
        mlflow.log_metric("baseline_dir", base_metrics.dir)
        mlflow.log_metric("baseline_spd", base_metrics.spd)

        logger.info("Baseline — acc=%.4f  %s", base_acc, base_metrics)

        # ── 3.3 Algorithmic mitigation ────────────────────────────────────────
        logger.info("Applying mitigation: %s", plan.method)
        mit_acc: float
        mit_metrics: FairnessMetrics

        if plan.method == MitigationMethod.REWEIGHING:
            reweigher = KamiranCaldersReweighing(
                sensitive_col=plan.protected_attribute,
                priv_value=priv_value,
                pos_label=1,
            )
            reweigher.fit(X_tr, y_tr, s_tr)
            sample_weights = reweigher.transform(s_tr, y_tr)
            mit_clf  = train_baseline(X_tr, y_tr, sample_weight=sample_weights)
            y_pred_mit = predict_with_scaler(mit_clf, X_te)
            mit_acc    = accuracy(y_te, y_pred_mit)
            mit_metrics = compute_metrics(y_te, y_pred_mit, s_te, priv_value=priv_value)
            final_clf = mit_clf

        elif plan.method in (
            MitigationMethod.THRESHOLD_OPTIMIZER,
            MitigationMethod.CALIBRATED_EO,
        ):
            # First train a standard model, then post-process thresholds
            base_clf2 = train_baseline(X_tr, y_tr)
            y_prob    = predict_proba_with_scaler(base_clf2, X_te)[:, 1]
            optimizer = ThresholdOptimizer(
                sensitive_col=plan.protected_attribute,
                priv_value=priv_value,
                pos_label=1,
                max_accuracy_drop=plan.max_accuracy_drop_pct / 100,
            )
            optimizer.fit(y_te, y_prob, s_te, baseline_accuracy=base_acc)
            y_pred_mit  = optimizer.predict(y_prob, s_te)
            mit_acc     = accuracy(y_te, y_pred_mit)
            mit_metrics = compute_metrics(y_te, y_pred_mit, s_te, priv_value=priv_value)
            final_clf   = base_clf2  # underlying model; threshold stored separately

        else:
            raise ValueError(f"Unsupported method: {plan.method}")

        mlflow.log_metric("mitigated_accuracy", mit_acc)
        mlflow.log_metric("mitigated_eod", mit_metrics.eod)
        mlflow.log_metric("mitigated_aod", mit_metrics.aod)
        mlflow.log_metric("mitigated_dir", mit_metrics.dir)
        mlflow.log_metric("mitigated_spd", mit_metrics.spd)
        mlflow.log_metric("accuracy_drop_pct",
                          round((base_acc - mit_acc) * 100, 4))

        logger.info("Mitigated — acc=%.4f  %s", mit_acc, mit_metrics)

        # ── 3.4 Persist model ─────────────────────────────────────────────────
        import joblib
        model_path = tmpdir / MODEL_ARTIFACT
        joblib.dump(final_clf, model_path)
        mlflow.sklearn.log_model(final_clf, artifact_path="model")

        model_gcs     = f"{args.output_gcs_uri}/{MODEL_ARTIFACT}"
        train_idx_gcs = f"{args.output_gcs_uri}/train_indices.npy"
        test_idx_gcs  = f"{args.output_gcs_uri}/test_indices.npy"

        gcs_upload(model_path,      model_gcs)
        gcs_upload(train_idx_path,  train_idx_gcs)
        gcs_upload(test_idx_path,   test_idx_gcs)

        # ── Build & store TrainingResult ──────────────────────────────────────
        result = TrainingResult(
            experiment_id=EXPERIMENT_NAME,
            run_id=run_id,
            dataset_hash=contract.dataset_hash,
            seed=SEED,
            baseline_accuracy=base_acc,
            mitigated_accuracy=mit_acc,
            baseline_bias=BiasMetrics(
                eod=base_metrics.eod,
                aod=base_metrics.aod,
                dir=base_metrics.dir,
                spd=base_metrics.spd,
            ),
            mitigated_bias=BiasMetrics(
                eod=mit_metrics.eod,
                aod=mit_metrics.aod,
                dir=mit_metrics.dir,
                spd=mit_metrics.spd,
            ),
            model_gcs_path=model_gcs,
            train_indices_path=train_idx_gcs,
            test_indices_path=test_idx_gcs,
        )
        result_path = tmpdir / "training_result.json"
        save_model(result, result_path)
        gcs_upload(result_path, f"{args.output_gcs_uri}/training_result.json")

        mlflow.end_run()
        logger.info("Training job complete. Run ID: %s", run_id)


if __name__ == "__main__":
    main()