"""Validate a lab-augmented, volume-deleaked predictor without replica leakage."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tools.train_leakage_resistant_predictor_v10_7 import (
    CLASSES, NUMERIC_WITH_VOLUME, ROOT, SEED, WEIGHTS, canonicalize_frame,
    enrich_external, fit_full, metrics, model_features, predict_ensemble,
)


def prepare_data(repeat: int):
    legacy = canonicalize_frame(pd.read_csv(ROOT / "data/knowledge_database.csv"))
    legacy["_model_group"] = legacy["Legante"].str.casefold()
    integrated = pd.read_csv(ROOT / "data/knowledge_database_integrated_v10_6.csv")
    lab = integrated[integrated["Source_Type"].eq("laboratory_experiment")].copy()
    lab = canonicalize_frame(lab)
    lab["_model_group"] = lab["Legante"].str.casefold()
    augmented = pd.concat([legacy, *([lab] * repeat)], ignore_index=True)
    return legacy, lab, augmented


def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    models, features = fit_full(
        train, NUMERIC_WITH_VOLUME, add_missing_indicators=False,
    )
    return models, features, predict_ensemble(models, test[features])


def legacy_grouped_cv(legacy: pd.DataFrame, lab: pd.DataFrame, repeat: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(
        legacy, legacy["Esito_ML"], legacy["_model_group"]
    ))
    probabilities = np.zeros((len(legacy), 3))
    folds = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train = pd.concat([legacy.iloc[train_idx], *([lab] * repeat)], ignore_index=True)
        _, _, fold_prob = fit_predict(train, legacy.iloc[test_idx])
        probabilities[test_idx] = fold_prob
        folds.append({
            "fold": fold,
            **metrics(legacy.iloc[test_idx].Esito_ML.astype(int).to_numpy(), fold_prob),
        })
        print(f"legacy augmented CV: completed fold {fold}/5", flush=True)
    return metrics(legacy.Esito_ML.astype(int).to_numpy(), probabilities), pd.DataFrame(folds)


def laboratory_condition_cv(legacy: pd.DataFrame, lab: pd.DataFrame, repeat: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    splits = list(splitter.split(
        lab, lab["Esito_ML"], lab["Condition_Group_ID"].astype(str)
    ))
    probabilities = np.zeros((len(lab), 3))
    fold_id = np.zeros(len(lab), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_lab = lab.iloc[train_idx]
        train = pd.concat([legacy, *([train_lab] * repeat)], ignore_index=True)
        _, _, fold_prob = fit_predict(train, lab.iloc[test_idx])
        probabilities[test_idx] = fold_prob
        fold_id[test_idx] = fold
        print(f"laboratory condition CV: completed fold {fold}/5", flush=True)
    predictions = pd.DataFrame({
        "record_id": lab["ID"].astype(str), "fold": fold_id,
        "true_class": lab["Esito_ML"].astype(int),
        "P_failed": probabilities[:, 0], "P_amorphous": probabilities[:, 1],
        "P_crystalline": probabilities[:, 2],
        "predicted_class": probabilities.argmax(axis=1),
        "condition_group": lab["Condition_Group_ID"].astype(str),
    })
    return metrics(lab.Esito_ML.astype(int).to_numpy(), probabilities), predictions


def literature_evaluation(models, features):
    literature = pd.read_csv(ROOT / "data/literature_crystalline_challenge_v10_6.csv")
    prepared = canonicalize_frame(enrich_external(literature))
    probabilities = predict_ensemble(models, prepared[features])
    predictions = pd.DataFrame({
        "record_id": literature["Case_ID"].astype(str),
        "MOF": literature["MOF"].astype(str),
        "true_class": 2, "P_failed": probabilities[:, 0],
        "P_amorphous": probabilities[:, 1], "P_crystalline": probabilities[:, 2],
        "predicted_class": probabilities.argmax(axis=1),
    })
    result = {
        "n": int(len(predictions)),
        "crystalline_argmax_recall": float((predictions.predicted_class == 2).mean()),
        "p_crystalline_at_least_0_5": float((predictions.P_crystalline >= 0.5).mean()),
        "mean_p_crystalline": float(predictions.P_crystalline.mean()),
    }
    return result, predictions


def main(repeat: int):
    legacy, lab, augmented = prepare_data(repeat)
    legacy_metrics, legacy_folds = legacy_grouped_cv(legacy, lab, repeat)
    lab_metrics, lab_predictions = laboratory_condition_cv(legacy, lab, repeat)
    models, features = fit_full(
        augmented, NUMERIC_WITH_VOLUME, add_missing_indicators=False,
    )
    literature_metrics, literature_predictions = literature_evaluation(models, features)
    artifact = {
        "version": f"10.7.0-candidate-lab-r{repeat}",
        "rf_model": models[0], "ligand_text_model": models[1],
        "weights": WEIGHTS.tolist(), "features": features,
        "classes": CLASSES.tolist(), "training_records_unique": int(len(legacy) + len(lab)),
        "legacy_training_records": int(len(legacy)), "laboratory_training_records": int(len(lab)),
        "laboratory_evidence_repeat_factor": repeat,
        "volume_missing_indicator_removed": True, "canonical_vocabulary": True,
    }
    suffix = f"r{repeat}"
    joblib.dump(artifact, ROOT / f"models/MOF_ChemAware_Ensemble_v10_7_lab_{suffix}_candidate.joblib")
    legacy_folds.to_csv(ROOT / f"reports/legacy_grouped_cv_v10_7_lab_{suffix}.csv", index=False)
    lab_predictions.to_csv(ROOT / f"reports/lab_condition_cv_v10_7_{suffix}.csv", index=False)
    literature_predictions.to_csv(ROOT / f"reports/literature_predictions_v10_7_lab_{suffix}.csv", index=False)
    result = {
        "repeat_factor": repeat, "legacy_grouped_cv": legacy_metrics,
        "laboratory_condition_group_cv": lab_metrics,
        "literature_positive_external": literature_metrics,
        "boundaries": [
            "Laboratory CV holds out complete condition groups, including duplicate signatures.",
            "Only one amorphous laboratory record is eligible, so its class recall has high uncertainty.",
            "The literature challenge is positive-only and cannot estimate specificity.",
        ],
    }
    (ROOT / f"reports/lab_augmented_validation_v10_7_{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, choices=range(1, 6))
    args = parser.parse_args()
    main(args.repeat)
