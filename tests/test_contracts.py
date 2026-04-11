"""Tests for src/shared/contracts.py"""
import pytest
from pydantic import ValidationError
from src.shared.contracts import (
    DatasetContract, FeatureSpec, FeatureType,
    MitigationPlan, MitigationMethod, BiasMetrics,
    FairnessThreshold, GateDecision, PipelineStatus,
)


VALID_CONTRACT_DICT = {
    "dataset_name": "test_dataset",
    "dataset_hash": "abc123def456",
    "version_tag": "test_v1",
    "row_count": 583,
    "column_count": 11,
    "target_variable": "Dataset",
    "positive_label": 1,
    "protected_attributes": ["Gender"],
    "features": [
        {
            "name": "Gender",
            "dtype": "binary",
            "cardinality": 2,
            "missing_pct": 0.0,
            "is_protected": True,
            "protected_groups": ["Male", "Female"],
        },
        {
            "name": "Age",
            "dtype": "numeric",
            "cardinality": 50,
            "missing_pct": 0.0,
            "is_protected": False,
        },
    ],
}


class TestDatasetContract:
    def test_valid_contract_passes(self):
        contract = DatasetContract.model_validate(VALID_CONTRACT_DICT)
        assert contract.dataset_name == "test_dataset"
        assert contract.target_variable == "Dataset"
        assert "Gender" in contract.protected_attributes

    def test_target_in_features_raises(self):
        bad = dict(VALID_CONTRACT_DICT)
        bad["features"] = list(VALID_CONTRACT_DICT["features"]) + [
            {"name": "Dataset", "dtype": "binary", "missing_pct": 0.0, "is_protected": False}
        ]
        with pytest.raises(ValidationError):
            DatasetContract.model_validate(bad)

    def test_protected_attr_not_in_features_raises(self):
        bad = dict(VALID_CONTRACT_DICT)
        bad["protected_attributes"] = ["NonExistentColumn"]
        with pytest.raises(ValidationError):
            DatasetContract.model_validate(bad)

    def test_missing_pct_out_of_range_raises(self):
        bad = dict(VALID_CONTRACT_DICT)
        bad["features"] = [
            {"name": "Gender", "dtype": "binary", "missing_pct": 1.5, "is_protected": True}
        ]
        bad["protected_attributes"] = ["Gender"]
        with pytest.raises(ValidationError):
            DatasetContract.model_validate(bad)

    def test_empty_protected_attributes_raises(self):
        bad = dict(VALID_CONTRACT_DICT)
        bad["protected_attributes"] = []
        with pytest.raises(ValidationError):
            DatasetContract.model_validate(bad)

    def test_default_schema_version(self):
        contract = DatasetContract.model_validate(VALID_CONTRACT_DICT)
        assert contract.schema_version == "1.0"

    def test_features_excludes_target(self):
        contract = DatasetContract.model_validate(VALID_CONTRACT_DICT)
        feat_names = [f.name for f in contract.features]
        assert "Dataset" not in feat_names


class TestBiasMetrics:
    def test_alias_construction(self):
        """BiasMetrics uses aliases: eod, aod, dir, spd."""
        m = BiasMetrics(eod=0.03, aod=0.02, dir=0.95, spd=-0.01)
        assert m.equal_opportunity_diff == 0.03
        assert m.disparate_impact_ratio == 0.95
        assert m.average_odds_diff == 0.02
        assert m.statistical_parity_diff == -0.01

    def test_field_names_accessible(self):
        m = BiasMetrics(eod=0.0, aod=0.0, dir=1.0, spd=0.0)
        assert hasattr(m, "equal_opportunity_diff")
        assert hasattr(m, "average_odds_diff")
        assert hasattr(m, "disparate_impact_ratio")
        assert hasattr(m, "statistical_parity_diff")


class TestMitigationPlan:
    def test_valid_plan(self):
        plan = MitigationPlan(
            dataset_version_tag="ilpd_v1",
            method=MitigationMethod.REWEIGHING,
            protected_attribute="Gender",
            privileged_group="Male",
            unprivileged_group="Female",
        )
        assert plan.method == MitigationMethod.REWEIGHING
        assert plan.max_accuracy_drop_pct == 2.0   # default

    def test_method_enum_values(self):
        assert MitigationMethod.REWEIGHING == "reweighing"
        assert MitigationMethod.THRESHOLD_OPTIMIZER == "threshold_optimizer"


class TestFairnessThreshold:
    def test_valid_threshold(self):
        t = FairnessThreshold(metric="eod", min_value=-0.05, max_value=0.05)
        assert t.min_value == -0.05

    def test_invalid_range_raises(self):
        with pytest.raises(ValidationError):
            FairnessThreshold(metric="eod", min_value=0.05, max_value=-0.05)

    def test_equal_min_max_raises(self):
        with pytest.raises(ValidationError):
            FairnessThreshold(metric="eod", min_value=0.05, max_value=0.05)


class TestFeatureSpec:
    def test_valid_feature(self):
        f = FeatureSpec(name="Age", dtype=FeatureType.NUMERIC, missing_pct=0.0)
        assert f.name == "Age"
        assert f.is_protected is False

    def test_missing_pct_boundary_zero(self):
        f = FeatureSpec(name="Age", dtype=FeatureType.NUMERIC, missing_pct=0.0)
        assert f.missing_pct == 0.0

    def test_missing_pct_boundary_one(self):
        f = FeatureSpec(name="Age", dtype=FeatureType.NUMERIC, missing_pct=1.0)
        assert f.missing_pct == 1.0

    def test_missing_pct_above_one_raises(self):
        with pytest.raises(ValidationError):
            FeatureSpec(name="Age", dtype=FeatureType.NUMERIC, missing_pct=1.1)

    def test_missing_pct_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FeatureSpec(name="Age", dtype=FeatureType.NUMERIC, missing_pct=-0.1)
