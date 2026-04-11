"""
shared/contracts.py
────────────────────
Pydantic v2 models that serve as the single source of truth for every
JSON artifact the pipeline produces and consumes.  Every agent, every
component, and every training script imports from here — never from raw
dicts.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class FeatureType(str, Enum):
    NUMERIC      = "numeric"
    CATEGORICAL  = "categorical"
    BINARY       = "binary"
    TEXT         = "text"
    DATETIME     = "datetime"


class MitigationMethod(str, Enum):
    REWEIGHING           = "reweighing"
    DISPARATE_IMPACT_REM = "disparate_impact_remover"
    CALIBRATED_EO        = "calibrated_eq_odds"
    THRESHOLD_OPTIMIZER  = "threshold_optimizer"


class PipelineStatus(str, Enum):
    RUNNING   = "running"
    PASSED    = "passed"
    FAILED    = "failed"
    SUSPENDED = "suspended"   # HITL pause
    APPROVED  = "approved"
    REJECTED  = "rejected"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Dataset Contract  (schema_agent.py → validate_data.py)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSpec(BaseModel):
    name:               str
    dtype:              FeatureType
    cardinality:        Optional[int]   = None
    missing_pct:        float           = 0.0
    is_protected:       bool            = False
    protected_groups:   Optional[List[str]] = None   # e.g. ["Male","Female"]
    notes:              Optional[str]   = None

    @field_validator("missing_pct")
    @classmethod
    def _missing_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("missing_pct must be in [0, 1]")
        return v


class DatasetContract(BaseModel):
    """Output of schema_agent — the binding spec for the entire run."""
    schema_version:       str           = "1.0"
    dataset_name:         str
    dataset_hash:         str           = Field(..., description="SHA-256 of raw CSV")
    version_tag:          str           = Field(..., description="e.g. ilpd_v1")
    row_count:            int
    column_count:         int
    target_variable:      str
    positive_label:       Any           = 1
    features:             List[FeatureSpec]
    protected_attributes: List[str]     = Field(..., min_length=1)
    created_at:           datetime      = Field(default_factory=datetime.utcnow)
    agent_model:          str           = "claude-sonnet-4-20250514"
    notes:                Optional[str] = None

    @model_validator(mode="after")
    def _target_not_in_features(self) -> "DatasetContract":
        feat_names = {f.name for f in self.features}
        if self.target_variable in feat_names:
            raise ValueError(
                f"Target variable '{self.target_variable}' must not appear in features list."
            )
        return self

    @model_validator(mode="after")
    def _protected_attrs_in_features(self) -> "DatasetContract":
        feat_names = {f.name for f in self.features}
        for attr in self.protected_attributes:
            if attr not in feat_names:
                raise ValueError(
                    f"Protected attribute '{attr}' not found in features."
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Validation Report  (validate_data.py)
# ─────────────────────────────────────────────────────────────────────────────

class LeakageFlag(BaseModel):
    feature:     str
    correlation: float
    threshold:   float = 0.95


class ValidationReport(BaseModel):
    status:           PipelineStatus
    dataset_hash:     str
    version_tag:      str
    null_checks:      Dict[str, float]          = Field(default_factory=dict)
    type_checks:      Dict[str, bool]           = Field(default_factory=dict)
    leakage_flags:    List[LeakageFlag]         = Field(default_factory=list)
    failure_reasons:  List[str]                 = Field(default_factory=list)
    validated_gcs_path: Optional[str]           = None
    timestamp:        datetime                  = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Mitigation Plan  (ethics_agent.py → train_and_mitigate.py)
# ─────────────────────────────────────────────────────────────────────────────

class FairnessThreshold(BaseModel):
    metric:    Literal["eod", "aod", "dir", "spd"] = "eod"
    min_value: float = -0.05
    max_value: float =  0.05

    @model_validator(mode="after")
    def _range_valid(self) -> "FairnessThreshold":
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be < max_value")
        return self


class MitigationPlan(BaseModel):
    plan_version:          str             = "1.0"
    dataset_version_tag:   str
    method:                MitigationMethod
    protected_attribute:   str
    privileged_group:      str
    unprivileged_group:    str
    fairness_thresholds:   List[FairnessThreshold] = Field(default_factory=list)
    max_accuracy_drop_pct: float           = 2.0
    hyperparameters:       Dict[str, Any]  = Field(default_factory=dict)
    policy_version:        str             = "ethics_policy_v1"
    rationale_summary:     Optional[str]   = None
    created_at:            datetime        = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Training Metrics  (task.py → evaluate_and_register.py)
# ─────────────────────────────────────────────────────────────────────────────

class BiasMetrics(BaseModel):
    equal_opportunity_diff: float   = Field(..., alias="eod")
    average_odds_diff:      float   = Field(..., alias="aod")
    disparate_impact_ratio: float   = Field(..., alias="dir")
    statistical_parity_diff:float   = Field(..., alias="spd")

    model_config = {"populate_by_name": True}


class TrainingResult(BaseModel):
    experiment_id:     str
    run_id:            str
    dataset_hash:      str
    seed:              int = 42
    baseline_accuracy: float
    mitigated_accuracy:float
    baseline_bias:     BiasMetrics
    mitigated_bias:    BiasMetrics
    model_gcs_path:    str
    train_indices_path:str
    test_indices_path: str
    timestamp:         datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Evaluation Gate  (evaluate_and_register.py)
# ─────────────────────────────────────────────────────────────────────────────

class GateDecision(BaseModel):
    gate_name:      str
    passed:         bool
    checks:         Dict[str, bool]   = Field(default_factory=dict)
    failure_reasons:List[str]         = Field(default_factory=list)
    model_registry_resource: Optional[str] = None
    timestamp:      datetime          = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Deployment Manifest  (deploy_endpoint.py)
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentManifest(BaseModel):
    endpoint_resource_name: str
    deployed_model_id:      str
    canary_traffic_pct:     int = 10
    baseline_traffic_pct:   int = 90
    hitl_pubsub_message_id: Optional[str] = None
    status:                 PipelineStatus = PipelineStatus.SUSPENDED
    approval_webhook_url:   Optional[str]  = None
    timestamp:              datetime       = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_contract(path: str | Path) -> DatasetContract:
    return DatasetContract.model_validate_json(Path(path).read_text())


def load_mitigation_plan(path: str | Path) -> MitigationPlan:
    return MitigationPlan.model_validate_json(Path(path).read_text())


def save_model(model: BaseModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(model.model_dump_json(indent=2))
