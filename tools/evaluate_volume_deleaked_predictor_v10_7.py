"""Evaluate a v10.7 candidate that suppresses only volume-missingness leakage."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tools.train_leakage_resistant_predictor_v10_7 import (
    CATEGORICAL, CLASSES, NUMERIC_WITH_VOLUME, ROOT, SEED, WEIGHTS,
    canonicalize_frame, enrich_external, fit_full, grouped_cv_variant,
    metrics, model_features, predict_ensemble,
)
from src.chem import canonicalize_ligand_for_model


def artifact_predictions(artifact, frame: pd.DataFrame, canonical: bool) -> np.ndarray:
    prepared = enrich_external(frame)
    if canonical:
        prepared = canonicalize_frame(prepared)
    features = artifact["features"]
    for col in features:
        if col not in prepared:
            prepared[col] = np.nan
    x = prepared[features]
    return (
        artifact["weights"][0] * artifact["rf_model"].predict_proba(x)
        + artifact["weights"][1] * artifact["ligand_text_model"].predict_proba(x)
    )


def positive_metrics(probabilities: np.ndarray) -> dict:
    return {
        "n": int(len(probabilities)),
        "crystalline_argmax_recall": float((probabilities.argmax(axis=1) == 2).mean()),
        "p_crystalline_at_least_0_5": float((probabilities[:, 2] >= 0.5).mean()),
        "mean_p_crystalline": float(probabilities[:, 2].mean()),
    }


def main():
    raw = pd.read_csv(ROOT / "data/knowledge_database.csv")
    canonical = canonicalize_frame(raw)
    canonical["_model_group"] = canonical["Legante"].str.casefold()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(splitter.split(
        canonical, canonical["Esito_ML"], canonical["_model_group"]
    ))
    cv_metrics, fold_metrics, cv_predictions = grouped_cv_variant(
        "canonical_volume_without_missing_flag", canonical,
        NUMERIC_WITH_VOLUME, outer_splits, add_missing_indicators=False,
    )
    models, features = fit_full(
        canonical, NUMERIC_WITH_VOLUME, add_missing_indicators=False,
    )
    artifact = {
        "version": "10.7.0-candidate-volume-deleaked",
        "rf_model": models[0], "ligand_text_model": models[1],
        "weights": WEIGHTS.tolist(), "features": features,
        "classes": CLASSES.tolist(), "training_records": int(len(canonical)),
        "training_policy": "legacy_1078_only; laboratory and literature evaluation-only",
        "volume_feature_removed": False,
        "volume_missing_indicator_removed": True,
        "canonical_vocabulary": True,
    }
    joblib.dump(
        artifact,
        ROOT / "models/MOF_ChemAware_Ensemble_v10_7_volume_deleaked_candidate.joblib",
    )

    integrated = pd.read_csv(ROOT / "data/knowledge_database_integrated_v10_6.csv")
    lab = integrated[integrated["Source_Type"].eq("laboratory_experiment")].copy()
    literature = pd.read_csv(ROOT / "data/literature_crystalline_challenge_v10_6.csv")
    literature["Esito_ML"] = 2
    frozen = joblib.load(ROOT / "models/MOF_ChemAware_Ensemble_v8_0.joblib")

    evaluations = {}
    rows = []
    for model_name, current_artifact, use_canonical in [
        ("frozen_v8_raw_input", frozen, False),
        ("frozen_v8_canonical_input", frozen, True),
        ("v10_7_volume_deleaked", artifact, True),
    ]:
        lab_prob = artifact_predictions(current_artifact, lab, use_canonical)
        lit_prob = artifact_predictions(current_artifact, literature, use_canonical)
        evaluations[model_name] = {
            "laboratory_held_out": metrics(lab.Esito_ML.astype(int).to_numpy(), lab_prob),
            "literature_positive": positive_metrics(lit_prob),
        }
        for source, frame, prob in [
            ("laboratory_held_out", lab, lab_prob),
            ("literature_positive", literature, lit_prob),
        ]:
            ids = frame.get("Case_ID", frame.get("ID")).astype(str).to_numpy()
            for i, record_id in enumerate(ids):
                rows.append({
                    "model": model_name, "evaluation_set": source,
                    "record_id": record_id,
                    "true_class": int(frame.iloc[i]["Esito_ML"]),
                    "P_failed": float(prob[i, 0]),
                    "P_amorphous": float(prob[i, 1]),
                    "P_crystalline": float(prob[i, 2]),
                    "predicted_class": int(prob[i].argmax()),
                })

    output = {
        "grouped_cv": cv_metrics,
        "external_evaluation": evaluations,
        "scientific_boundary": (
            "The 17 laboratory records and 9 positive literature protocols were "
            "held out from model fitting. The literature set does not measure specificity."
        ),
    }
    fold_metrics.to_csv(
        ROOT / "reports/grouped_cv_volume_deleaked_fold_metrics_v10_7.csv", index=False
    )
    cv_predictions.to_csv(
        ROOT / "reports/grouped_cv_volume_deleaked_predictions_v10_7.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        ROOT / "reports/external_model_comparison_v10_7.csv", index=False
    )
    (ROOT / "reports/volume_deleaked_validation_v10_7.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
