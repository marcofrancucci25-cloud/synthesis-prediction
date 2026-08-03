"""Train the v10.12 production model without documentation-leakage features.

Policy:
- train only rows that passed the deterministic v10.5 quality audit;
- never expose solvent volume or a volume-missing indicator to the model;
- group every validation fold by canonical ligand identity;
- keep laboratory evidence evaluation-only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_leakage_resistant_predictor_v10_7 import (
    CATEGORICAL, CLASSES, NUMERIC_NO_VOLUME, ROOT, SEED, WEIGHTS,
    canonicalize_frame, enrich_external, fit_full, grouped_cv_variant,
    metrics, predict_ensemble,
)


VERSION = "10.12.0-audited-deleaked"


def main():
    audited = pd.read_csv(ROOT / "data/knowledge_database_audited_v10_5.csv")
    data = audited[audited["Quality_Status"].eq("INCLUDE")].copy()
    data = canonicalize_frame(data)
    data["_model_group"] = data["Legante"].str.casefold()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(data, data["Esito_ML"], data["_model_group"]))
    summary, folds, oof = grouped_cv_variant(
        VERSION, data, NUMERIC_NO_VOLUME, splits,
    )

    models, features = fit_full(data, NUMERIC_NO_VOLUME)
    artifact = {
        "version": VERSION,
        "rf_model": models[0],
        "ligand_text_model": models[1],
        "weights": WEIGHTS.tolist(),
        "features": features,
        "classes": CLASSES.tolist(),
        "training_records": int(len(data)),
        "excluded_review_records": int(len(audited) - len(data)),
        "training_policy": (
            "quality_status=INCLUDE only; volume and volume-missingness excluded; "
            "canonical-ligand grouped validation"
        ),
        "volume_feature_removed": True,
        "volume_missing_indicator_removed": True,
        "probability_scope": "relative evidence score; not prospectively validated success probability",
    }

    lab = pd.read_csv(ROOT / "data/knowledge_database_integrated_v10_6.csv")
    lab = lab[lab["Source_Type"].eq("laboratory_experiment")].copy()
    prepared = canonicalize_frame(enrich_external(lab))
    for col in features:
        if col not in prepared:
            prepared[col] = np.nan
    lab_prob = predict_ensemble(models, prepared[features])
    lab_summary = metrics(lab["Esito_ML"].astype(int).to_numpy(), lab_prob)

    report = {
        "version": VERSION,
        "training_records": int(len(data)),
        "excluded_review_records": int(len(audited) - len(data)),
        "class_distribution": {
            str(k): int(v) for k, v in data["Esito_ML"].value_counts().sort_index().items()
        },
        "grouped_cv_by_canonical_ligand": summary,
        "held_out_laboratory_evaluation": lab_summary,
        "promotion_decision": "DEPLOY_WITH_ABSTENTION_AND_NON_PROBABILISTIC_LABEL",
        "limitations": [
            "Minority-class recall remains insufficient for autonomous experimental decisions.",
            "Historical source DOI and row-level diffraction provenance remain incomplete.",
            "Scores are shown only when the applicability and confidence gates pass.",
        ],
    }

    schema = {
        "version": VERSION,
        "features": features,
        "numeric": NUMERIC_NO_VOLUME,
        "categorical": CATEGORICAL,
        "text": "Ligand_Text",
        "classes": {"0": "Failed", "1": "Amorphous/uncertain", "2": "Crystalline MOF"},
        "notes": [
            "Solvent volume is used for physical validity checks only, never prediction.",
            "Rows flagged REVIEW by the deterministic audit are excluded.",
            "Model outputs are relative evidence scores, not prospectively validated probabilities.",
        ],
    }

    joblib.dump(artifact, ROOT / "models/MOF_Audited_Deleaked_v10_12.joblib")
    (ROOT / "models/feature_schema_v10_12.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    (ROOT / "reports/predictor_validation_v10_12.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    folds.to_csv(ROOT / "reports/grouped_cv_fold_metrics_v10_12.csv", index=False)
    oof.to_csv(ROOT / "reports/grouped_cv_predictions_v10_12.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
