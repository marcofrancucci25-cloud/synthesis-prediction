from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_multiclass_external_benchmark_has_no_training_overlap():
    train = pd.read_csv(ROOT / "data/v12_training_candidates.csv")
    test = pd.read_csv(ROOT / "data/v12_locked_external_benchmark_multiclass.csv")
    assert not set(train.Record_ID) & set(test.Record_ID)
    assert set(test.Outcome_Class) == {0, 1, 2}


def test_promotion_gate_stays_closed_until_real_evidence_is_added():
    report = json.loads((ROOT / "reports/v12_validation_readiness.json").read_text())
    assert not report["promotion_gate_passed"]
    assert not report["readiness_gates"]["benchmark_minimum_10_per_class"]


def test_prospective_plan_contains_controls_and_model_arms():
    plan = pd.read_csv(ROOT / "data/prospective_validation_plan_v10_12.csv")
    assert {"CONTROL_1TO1_DMF", "LITERATURE_REFERENCE", "MODEL_TOP1"}.issubset(
        set(plan.Experiment_Arm)
    )
    assert plan.Required_Replicates.ge(3).all()
