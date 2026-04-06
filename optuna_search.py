"""
optuna_search.py
─────────────────
Multi-Objective AutoML Engine for FairGuard V2.

Searches across 10 classifiers simultaneously optimising for:
  - Objective 1: MAXIMIZE accuracy
  - Objective 2: MINIMIZE |EOD| (absolute Equal Opportunity Difference)

Uses Optuna's NSGA-II (Non-dominated Sorting Genetic Algorithm) sampler,
which is purpose-built for Pareto-front exploration.

Usage (standalone):
    python optuna_search.py \
        --csv  data/indian_liver_patient.csv \
        --trials 100 \
        --out  local_artifacts/optimal_hyperparameters.json

Usage (from local_demo.py):
    from optuna_search import run_optuna_search
    best = run_optuna_search(X_train, y_train, sensitive_train, n_trials=100)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Suppress noisy library warnings during search ────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)   # Optuna stays quiet too

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SEED     = 42
N_FOLDS  = 5
EOD_HARD_THRESHOLD = 0.05      # Pareto selection: prefer trials under this


# ─────────────────────────────────────────────────────────────────────────────
# Model factory — suggest + instantiate one of 10 classifiers
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_model(trial: optuna.Trial) -> Any:
    """
    Ask Optuna to pick a model family and its hyperparameters.
    Every model uses random_state=SEED where supported.
    """
    model_name = trial.suggest_categorical("model", [
        "xgboost", "lightgbm", "catboost",
        "random_forest", "logistic_regression",
        "svc", "knn", "decision_tree",
        "adaboost", "mlp",
    ])

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators    = trial.suggest_int("xgb_n_est",       50, 500),
            max_depth       = trial.suggest_int("xgb_depth",        2,   8),
            learning_rate   = trial.suggest_float("xgb_lr",       1e-3, 0.3, log=True),
            subsample       = trial.suggest_float("xgb_sub",       0.5, 1.0),
            colsample_bytree= trial.suggest_float("xgb_col",       0.5, 1.0),
            reg_alpha       = trial.suggest_float("xgb_alpha",    1e-8, 1.0, log=True),
            reg_lambda      = trial.suggest_float("xgb_lambda",   1e-8, 1.0, log=True),
            use_label_encoder=False,
            eval_metric     ="logloss",
            random_state    = SEED,
            verbosity       = 0,
        )

    # ── LightGBM ─────────────────────────────────────────────────────────────
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators    = trial.suggest_int("lgbm_n_est",      50, 500),
            max_depth       = trial.suggest_int("lgbm_depth",       2,   8),
            learning_rate   = trial.suggest_float("lgbm_lr",      1e-3, 0.3, log=True),
            num_leaves      = trial.suggest_int("lgbm_leaves",     15, 127),
            subsample       = trial.suggest_float("lgbm_sub",      0.5, 1.0),
            colsample_bytree= trial.suggest_float("lgbm_col",      0.5, 1.0),
            min_child_samples= trial.suggest_int("lgbm_mcs",        5,  50),
            random_state    = SEED,
            verbose         = -1,
        )

    # ── CatBoost ─────────────────────────────────────────────────────────────
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations      = trial.suggest_int("cb_iter",        50, 400),
            depth           = trial.suggest_int("cb_depth",        2,   8),
            learning_rate   = trial.suggest_float("cb_lr",       1e-3, 0.3, log=True),
            l2_leaf_reg     = trial.suggest_float("cb_l2",        1.0, 10.0),
            random_seed     = SEED,
            verbose         = False,    # suppress all CatBoost output
        )

    # ── Random Forest ─────────────────────────────────────────────────────────
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators    = trial.suggest_int("rf_n_est",       50, 500),
            max_depth       = trial.suggest_int("rf_depth",        2,  20, log=False),
            min_samples_split= trial.suggest_int("rf_mss",         2,  20),
            min_samples_leaf = trial.suggest_int("rf_msl",         1,  10),
            max_features    = trial.suggest_categorical("rf_feat", ["sqrt", "log2", None]),
            random_state    = SEED,
            n_jobs          = -1,
        )

    # ── Logistic Regression ───────────────────────────────────────────────────
    elif model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C               = trial.suggest_float("lr_C",         1e-4, 100.0, log=True),
            penalty         = trial.suggest_categorical("lr_pen", ["l1", "l2"]),
            solver          = "liblinear",
            max_iter        = 1000,
            random_state    = SEED,
        )

    # ── SVC ───────────────────────────────────────────────────────────────────
    elif model_name == "svc":
        from sklearn.svm import SVC
        return SVC(
            C               = trial.suggest_float("svc_C",        1e-3, 100.0, log=True),
            kernel          = trial.suggest_categorical("svc_ker", ["rbf", "linear", "poly"]),
            gamma           = trial.suggest_categorical("svc_gam", ["scale", "auto"]),
            probability     = True,    # needed for predict_proba in ThresholdOptimizer
            random_state    = SEED,
        )

    # ── KNN ───────────────────────────────────────────────────────────────────
    elif model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors     = trial.suggest_int("knn_k",           3,  31, step=2),
            weights         = trial.suggest_categorical("knn_w",  ["uniform", "distance"]),
            metric          = trial.suggest_categorical("knn_m",  ["euclidean", "manhattan"]),
            n_jobs          = -1,
        )

    # ── Decision Tree ─────────────────────────────────────────────────────────
    elif model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            max_depth       = trial.suggest_int("dt_depth",        2,  20),
            min_samples_split= trial.suggest_int("dt_mss",         2,  20),
            min_samples_leaf = trial.suggest_int("dt_msl",         1,  10),
            criterion       = trial.suggest_categorical("dt_crit", ["gini", "entropy"]),
            random_state    = SEED,
        )

    # ── AdaBoost ──────────────────────────────────────────────────────────────
    elif model_name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(
            n_estimators    = trial.suggest_int("ada_n_est",      50, 300),
            learning_rate   = trial.suggest_float("ada_lr",      1e-3, 2.0, log=True),
            algorithm       = trial.suggest_categorical("ada_alg", ["SAMME", "SAMME.R"]),
            random_state    = SEED,
        )

    # ── MLP Neural Network ────────────────────────────────────────────────────
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        n_layers = trial.suggest_int("mlp_layers", 1, 3)
        layer_size = trial.suggest_int("mlp_units", 32, 256)
        hidden = tuple([layer_size] * n_layers)
        return MLPClassifier(
            hidden_layer_sizes = hidden,
            activation      = trial.suggest_categorical("mlp_act", ["relu", "tanh"]),
            alpha           = trial.suggest_float("mlp_alpha",   1e-5, 1e-1, log=True),
            learning_rate_init= trial.suggest_float("mlp_lr",    1e-4, 1e-1, log=True),
            max_iter        = 500,
            early_stopping  = True,
            random_state    = SEED,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fairness metric (inline — no circular import from src.ml.metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_eod(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    sensitive: np.ndarray,
    priv_value,
    pos_label: int = 1,
) -> float:
    """Equal Opportunity Difference = TPR(unprivileged) − TPR(privileged)."""
    def tpr(mask):
        pos = (y_true == pos_label) & mask
        total = pos.sum()
        return float(((y_pred == pos_label) & pos).sum() / total) if total > 0 else 0.0

    priv_mask   = sensitive == priv_value
    unpriv_mask = ~priv_mask
    return tpr(unpriv_mask) - tpr(priv_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────────────

def _make_objective(
    X:         np.ndarray,
    y:         np.ndarray,
    sensitive: np.ndarray,
    priv_value,
):
    """
    Returns a closure that Optuna calls for each trial.
    Uses StratifiedKFold so every fold preserves class balance.
    Sensitive array is split in parallel with X and y.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scaler = StandardScaler()

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        model = _suggest_model(trial)

        fold_accs: list[float] = []
        fold_eods: list[float] = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            s_val       = sensitive[val_idx]

            # Scale features (helps LR, SVC, MLP; harmless for trees)
            X_tr_sc  = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)

            try:
                model.fit(X_tr_sc, y_tr)
                y_pred = model.predict(X_val_sc)
            except Exception as exc:
                # If a trial config is pathological, prune it gracefully
                raise optuna.exceptions.TrialPruned(f"Model fit failed: {exc}")

            acc = float(np.mean(y_pred == y_val))
            eod = _compute_eod(y_val, y_pred, s_val, priv_value)

            fold_accs.append(acc)
            fold_eods.append(abs(eod))

        mean_acc = float(np.mean(fold_accs))
        mean_abs_eod = float(np.mean(fold_eods))

        return mean_acc, mean_abs_eod   # (maximise, minimise)

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Pareto front selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """
    From the Pareto front, pick the single best trial:
      Priority 1 — highest accuracy among trials with |EOD| ≤ 0.05
      Priority 2 — if no trial meets the threshold, pick lowest |EOD|
    """
    pareto = study.best_trials    # Optuna returns the non-dominated set

    # Split into passing and failing the EOD threshold
    passing = [t for t in pareto if t.values[1] <= EOD_HARD_THRESHOLD]
    failing = [t for t in pareto if t.values[1] >  EOD_HARD_THRESHOLD]

    if passing:
        # Among passing trials, pick the one with highest accuracy (values[0])
        best = max(passing, key=lambda t: t.values[0])
        logger.info(
            "Pareto selection (threshold met): accuracy=%.4f  |EOD|=%.4f  model=%s",
            best.values[0], best.values[1], best.params.get("model"),
        )
    else:
        # Fallback: pick lowest absolute EOD from failing trials
        best = min(failing, key=lambda t: t.values[1])
        logger.warning(
            "No trial met EOD ≤ %.2f threshold! "
            "Selecting lowest EOD: accuracy=%.4f  |EOD|=%.4f  model=%s",
            EOD_HARD_THRESHOLD,
            best.values[0], best.values[1], best.params.get("model"),
        )

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_search(
    X_train:            np.ndarray,
    y_train:            np.ndarray,
    sensitive_features: np.ndarray,
    priv_value:         int | str = 1,
    n_trials:           int = 100,
    output_path:        str = "local_artifacts/optimal_hyperparameters.json",
    show_progress:      bool = True,
) -> dict[str, Any]:
    """
    Run multi-objective Optuna search.

    Parameters
    ----------
    X_train, y_train :
        Training features and labels (numpy arrays).
    sensitive_features :
        Encoded sensitive attribute array (same length as y_train).
    priv_value :
        Encoded value representing the privileged group.
    n_trials :
        Number of Optuna trials. 100 is solid; 50 works for a quick run.
    output_path :
        Where to save optimal_hyperparameters.json.
    show_progress :
        Show tqdm progress bar.

    Returns
    -------
    dict with keys:
        model_name, hyperparameters, cv_accuracy, cv_abs_eod,
        pareto_front_size, all_pareto_trials
    """
    logger.info(
        "🔍 Starting Optuna multi-objective search | trials=%d  folds=%d  models=10",
        n_trials, N_FOLDS,
    )

    # NSGA-II is the gold-standard sampler for multi-objective problems
    sampler = optuna.samplers.NSGAIISampler(seed=SEED)

    study = optuna.create_study(
        directions=["maximize", "minimize"],   # accuracy ↑, |EOD| ↓
        sampler=sampler,
        study_name="fairguard_automl",
    )

    objective = _make_objective(X_train, y_train, sensitive_features, priv_value)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
        n_jobs=1,           # keep deterministic; set to -1 to parallelise
        catch=(Exception,), # don't crash the whole search on one bad trial
    )

    # ── Select winner from Pareto front ──────────────────────────────────────
    best = _select_best_trial(study)

    # ── Build output dict ─────────────────────────────────────────────────────
    pareto_trials = [
        {
            "trial_number": t.number,
            "model":        t.params.get("model"),
            "cv_accuracy":  round(t.values[0], 6),
            "cv_abs_eod":   round(t.values[1], 6),
        }
        for t in study.best_trials
    ]

    result = {
        "model_name":        best.params["model"],
        "hyperparameters":   {k: v for k, v in best.params.items() if k != "model"},
        "cv_accuracy":       round(best.values[0], 6),
        "cv_abs_eod":        round(best.values[1], 6),
        "met_eod_threshold": best.values[1] <= EOD_HARD_THRESHOLD,
        "eod_threshold_used":EOD_HARD_THRESHOLD,
        "n_trials_completed":len(study.trials),
        "pareto_front_size": len(study.best_trials),
        "all_pareto_trials": pareto_trials,
        "seed":              SEED,
        "n_folds":           N_FOLDS,
    }

    # ── Save to disk ──────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    logger.info("✅ Optimal hyperparameters saved → %s", output_path)

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║          🏆 Optuna Search Complete                   ║\n"
        "║                                                      ║\n"
        "║  Best model    : %-34s║\n"
        "║  CV Accuracy   : %-34s║\n"
        "║  CV |EOD|      : %-34s║\n"
        "║  EOD threshold : %-34s║\n"
        "║  Pareto front  : %-34s║\n"
        "╚══════════════════════════════════════════════════════╝",
        best.params["model"],
        f"{best.values[0]:.4f}",
        f"{best.values[1]:.4f}",
        f"{'✅ MET' if best.values[1] <= EOD_HARD_THRESHOLD else '⚠️  NOT MET'} (≤{EOD_HARD_THRESHOLD})",
        f"{len(study.best_trials)} trials",
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model instantiation from saved hyperparameters (used by task.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_from_params(params: dict[str, Any]) -> Any:
    """
    Reconstruct the winning model from optimal_hyperparameters.json.
    Called by task.py / local_demo.py to avoid re-running the search.
    """
    model_name = params["model_name"]
    hp         = params["hyperparameters"]

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators     = hp.get("xgb_n_est", 200),
            max_depth        = hp.get("xgb_depth",   4),
            learning_rate    = hp.get("xgb_lr",    0.08),
            subsample        = hp.get("xgb_sub",    0.8),
            colsample_bytree = hp.get("xgb_col",    0.8),
            reg_alpha        = hp.get("xgb_alpha",  0.0),
            reg_lambda       = hp.get("xgb_lambda", 1.0),
            use_label_encoder= False,
            eval_metric      = "logloss",
            random_state     = SEED,
            verbosity        = 0,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators     = hp.get("lgbm_n_est",  200),
            max_depth        = hp.get("lgbm_depth",    4),
            learning_rate    = hp.get("lgbm_lr",     0.08),
            num_leaves       = hp.get("lgbm_leaves",  31),
            subsample        = hp.get("lgbm_sub",     0.8),
            colsample_bytree = hp.get("lgbm_col",     0.8),
            min_child_samples= hp.get("lgbm_mcs",      20),
            random_state     = SEED,
            verbose          = -1,
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations    = hp.get("cb_iter",  200),
            depth         = hp.get("cb_depth",   4),
            learning_rate = hp.get("cb_lr",    0.08),
            l2_leaf_reg   = hp.get("cb_l2",     3.0),
            random_seed   = SEED,
            verbose       = False,
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators     = hp.get("rf_n_est",  200),
            max_depth        = hp.get("rf_depth",   None),
            min_samples_split= hp.get("rf_mss",       2),
            min_samples_leaf = hp.get("rf_msl",       1),
            max_features     = hp.get("rf_feat", "sqrt"),
            random_state     = SEED,
            n_jobs           = -1,
        )
    elif model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C            = hp.get("lr_C",    1.0),
            penalty      = hp.get("lr_pen", "l2"),
            solver       = "liblinear",
            max_iter     = 1000,
            random_state = SEED,
        )
    elif model_name == "svc":
        from sklearn.svm import SVC
        return SVC(
            C           = hp.get("svc_C",     1.0),
            kernel      = hp.get("svc_ker", "rbf"),
            gamma       = hp.get("svc_gam", "scale"),
            probability = True,
            random_state= SEED,
        )
    elif model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors = hp.get("knn_k",       5),
            weights     = hp.get("knn_w", "uniform"),
            metric      = hp.get("knn_m", "euclidean"),
            n_jobs      = -1,
        )
    elif model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            max_depth        = hp.get("dt_depth",  None),
            min_samples_split= hp.get("dt_mss",       2),
            min_samples_leaf = hp.get("dt_msl",       1),
            criterion        = hp.get("dt_crit", "gini"),
            random_state     = SEED,
        )
    elif model_name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(
            n_estimators = hp.get("ada_n_est",    100),
            learning_rate= hp.get("ada_lr",       1.0),
            algorithm    = hp.get("ada_alg", "SAMME.R"),
            random_state = SEED,
        )
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        n_layers   = hp.get("mlp_layers", 2)
        layer_size = hp.get("mlp_units", 128)
        return MLPClassifier(
            hidden_layer_sizes = tuple([layer_size] * n_layers),
            activation         = hp.get("mlp_act",  "relu"),
            alpha              = hp.get("mlp_alpha", 1e-4),
            learning_rate_init = hp.get("mlp_lr",    1e-3),
            max_iter           = 500,
            early_stopping     = True,
            random_state       = SEED,
        )

    raise ValueError(f"Unknown model_name in hyperparameters: '{model_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from sklearn.preprocessing import LabelEncoder as LE

    parser = argparse.ArgumentParser(description="FairGuard Optuna AutoML Search")
    parser.add_argument("--csv",        required=True,  help="Path to the raw CSV file")
    parser.add_argument("--target",     default="Dataset",   help="Target column name")
    parser.add_argument("--protected",  default="Gender",    help="Sensitive attribute column")
    parser.add_argument("--priv-group", default="Male",      help="Privileged group value")
    parser.add_argument("--trials",     type=int, default=100)
    parser.add_argument("--out",        default="local_artifacts/optimal_hyperparameters.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Quick encoding
    target_col    = args.target
    protected_col = args.protected

    y         = (df[target_col] == 1).astype(int).values
    sensitive  = LE().fit_transform(df[protected_col].astype(str))
    priv_enc   = int(LE().fit_transform([args.priv_group])[0])

    feature_cols = [c for c in df.columns if c not in [target_col, protected_col]]
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = LE().fit_transform(df[col].astype(str))
    X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(float)

    result = run_optuna_search(
        X_train=X,
        y_train=y,
        sensitive_features=sensitive,
        priv_value=priv_enc,
        n_trials=args.trials,
        output_path=args.out,
    )

    print(f"\nBest model   : {result['model_name']}")
    print(f"CV Accuracy  : {result['cv_accuracy']:.4f}")
    print(f"CV |EOD|     : {result['cv_abs_eod']:.4f}")
    print(f"Pareto size  : {result['pareto_front_size']} trials")
    print(f"Saved to     : {args.out}")