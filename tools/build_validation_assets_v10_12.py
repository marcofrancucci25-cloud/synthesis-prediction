"""Build leakage-safe future-training and multiclass benchmark assets.

All direct laboratory experiments are locked out of future fitting.  This
creates a genuinely external laboratory campaign for future model comparison,
while retaining the positive DOI benchmark as a separate evidence component.
The gate intentionally remains closed because failed/uncertain benchmark
coverage is still too small.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    gold = pd.read_csv(ROOT / "data/v11_gold_synthesis_records.csv")
    lab_mask = gold["Source_Type"].eq("direct_laboratory_experiment")
    future_train = gold[gold["Training_Eligibility"].astype(bool) & ~lab_mask].copy()
    literature_external = gold[~gold["Training_Eligibility"].astype(bool)].copy()
    laboratory_external = gold[lab_mask].copy()
    laboratory_external["Partition_Role"] = "LOCKED_EXTERNAL_LAB_CAMPAIGN"
    laboratory_external["Training_Eligibility"] = False
    external = pd.concat([literature_external, laboratory_external], ignore_index=True)

    train_sources = set(future_train["Source_DOI"].dropna().astype(str)) - {""}
    external_sources = set(external["Source_DOI"].dropna().astype(str)) - {""}
    if train_sources & external_sources:
        raise ValueError("DOI leakage between future training and external benchmark")
    if set(future_train["Record_ID"]) & set(external["Record_ID"]):
        raise ValueError("Record leakage between future training and external benchmark")

    future_train.to_csv(ROOT / "data/v12_training_candidates.csv", index=False)
    external.to_csv(ROOT / "data/v12_locked_external_benchmark_multiclass.csv", index=False)

    distribution = {
        str(k): int(v) for k, v in external["Outcome_Class"].value_counts().sort_index().items()
    }
    gates = {
        "benchmark_contains_all_three_classes": set(distribution) == {"0", "1", "2"},
        "benchmark_minimum_10_per_class": all(distribution.get(str(i), 0) >= 10 for i in range(3)),
        "training_minimum_20_independent_dois": future_train["Source_DOI"].dropna().nunique() >= 20,
        "training_minimum_10_metal_ligand_groups": future_train["Metal_Ligand_Group"].nunique() >= 10,
        "zero_record_overlap": not bool(set(future_train["Record_ID"]) & set(external["Record_ID"])),
        "zero_doi_overlap": not bool(train_sources & external_sources),
    }
    summary = {
        "version": "10.12.0-validation-assets",
        "future_training_records": int(len(future_train)),
        "external_benchmark_records": int(len(external)),
        "external_class_distribution": distribution,
        "external_sources": {
            "locked_peer_reviewed_positive": int(len(literature_external)),
            "locked_direct_laboratory_campaign": int(len(laboratory_external)),
        },
        "readiness_gates": gates,
        "promotion_gate_passed": all(gates.values()),
        "decision": (
            "Do not retrain v11/v12 from these candidates yet; acquire independent failed and "
            "amorphous campaigns with complete DOI/protocol provenance."
        ),
    }
    (ROOT / "reports/v12_validation_readiness.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
