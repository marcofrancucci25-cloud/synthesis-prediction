"""Train and validate a leakage-resistant successor to the frozen v8 model.

The laboratory records and literature challenge are evaluation-only.  They are
never concatenated to the 1,078-row legacy training table.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.chem import (
    build_row, canonicalize_family, canonicalize_ligand_for_model,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 260802
WEIGHTS = np.array([0.7, 0.3])
CLASSES = np.array([0, 1, 2])

NUMERIC_WITH_VOLUME = [
    "Temperatura_C", "Tempo_ore", "mmol_Legante", "mmol_Sale",
    "Rapporto_LM", "Volume solvente", "Hydration_Number", "Oxidation_State",
    "Metal_Atomic_Number", "Metal_Atomic_Weight", "Metal_Group",
    "Metal_Period", "Metal_Electronegativity",
]
NUMERIC_NO_VOLUME = [c for c in NUMERIC_WITH_VOLUME if c != "Volume solvente"]
CATEGORICAL = [
    "Famiglia_Legante", "Metallo", "Sale_Metallico", "Counterion_Class",
    "Metal_Block", "Solvente", "Additivo_Colinker",
]


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Legante"] = out["Legante"].map(canonicalize_ligand_for_model)
    out["Famiglia_Legante"] = [
        canonicalize_family(f, ligand)
        for f, ligand in zip(out["Famiglia_Legante"], out["Legante"])
    ]
    out["Ligand_Text"] = (
        out["Legante"].fillna("").astype(str) + " "
        + out["Famiglia_Legante"].fillna("").astype(str)
    ).str.lower()
    return out


def model_features(numeric: list[str]) -> list[str]:
    return ["Ligand_Text", *numeric, *CATEGORICAL]


def _preprocessor(
    numeric: list[str], include_text: bool, add_missing_indicators: bool = True,
) -> ColumnTransformer:
    transformers = []
    if include_text:
        transformers.append((
            "txt",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                max_features=1200, sublinear_tf=True,
            ),
            "Ligand_Text",
        ))
    numeric_core = numeric
    if "Volume solvente" in numeric and not add_missing_indicators:
        numeric_core = [c for c in numeric if c != "Volume solvente"]
    transformers.append((
        "num",
        Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]),
        numeric_core,
    ))
    if "Volume solvente" in numeric and not add_missing_indicators:
        # Preserve the physical value while preventing the historical absence of
        # that value from becoming a direct class feature.
        transformers.append((
            "volume_without_missing_flag",
            Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=False)),
                ("sc", StandardScaler()),
            ]),
            ["Volume solvente"],
        ))
    transformers.extend([
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]),
            CATEGORICAL,
        ),
    ])
    return ColumnTransformer(transformers)


def make_models(
    numeric: list[str], inner_cv, seed: int, add_missing_indicators: bool = True,
):
    rf = Pipeline([
        ("prep", _preprocessor(
            numeric, include_text=False,
            add_missing_indicators=add_missing_indicators,
        )),
        ("model", RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )),
    ])
    lr = Pipeline([
        ("prep", _preprocessor(
            numeric, include_text=True,
            add_missing_indicators=add_missing_indicators,
        )),
        ("model", LogisticRegression(
            C=2.0, class_weight="balanced", solver="saga", max_iter=3000,
            random_state=seed, n_jobs=1,
        )),
    ])
    return (
        CalibratedClassifierCV(rf, method="sigmoid", cv=inner_cv, n_jobs=1),
        CalibratedClassifierCV(lr, method="sigmoid", cv=inner_cv, n_jobs=1),
    )


def predict_ensemble(models, x: pd.DataFrame) -> np.ndarray:
    return WEIGHTS[0] * models[0].predict_proba(x) + WEIGHTS[1] * models[1].predict_proba(x)


def ece_score(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        include_right = i == bins - 1
        mask = (confidence >= edges[i]) & (
            (confidence <= edges[i + 1]) if include_right else (confidence < edges[i + 1])
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[y.astype(int)]
    per_class = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10": ece_score(y, probabilities),
        "recall_failed": float(per_class[0]),
        "recall_amorphous": float(per_class[1]),
        "recall_crystalline": float(per_class[2]),
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def grouped_cv_variant(
    name: str, data: pd.DataFrame, numeric: list[str], outer_splits,
    add_missing_indicators: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = model_features(numeric)
    y = data["Esito_ML"].astype(int).to_numpy()
    oof = np.zeros((len(data), 3), dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        inner_splits = list(inner.split(
            train[features], train["Esito_ML"].astype(int), train["_model_group"]
        ))
        fitted = make_models(
            numeric, inner_splits, SEED + fold,
            add_missing_indicators=add_missing_indicators,
        )
        fitted[0].fit(train[features], train["Esito_ML"].astype(int))
        fitted[1].fit(train[features], train["Esito_ML"].astype(int))
        fold_prob = predict_ensemble(fitted, test[features])
        oof[test_idx] = fold_prob
        fold_rows.append({
            "variant": name, "fold": fold,
            **metrics(test["Esito_ML"].astype(int).to_numpy(), fold_prob),
        })
        print(f"{name}: completed fold {fold}/5", flush=True)
    prediction_frame = pd.DataFrame({
        "ID": data["ID"], "true_class": y,
        "P_failed": oof[:, 0], "P_amorphous": oof[:, 1], "P_crystalline": oof[:, 2],
        "predicted_class": oof.argmax(axis=1), "variant": name,
    })
    return metrics(y, oof), pd.DataFrame(fold_rows), prediction_frame


def fit_full(
    data: pd.DataFrame, numeric: list[str], add_missing_indicators: bool = True,
):
    features = model_features(numeric)
    y = data["Esito_ML"].astype(int)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    inner_splits = list(inner.split(data[features], y, data["_model_group"]))
    fitted = make_models(
        numeric, inner_splits, SEED,
        add_missing_indicators=add_missing_indicators,
    )
    fitted[0].fit(data[features], y)
    fitted[1].fit(data[features], y)
    return fitted, features


def enrich_external(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in frame.iterrows():
        row = build_row(record.to_dict()).iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def external_predictions(models, features, frame: pd.DataFrame, set_name: str) -> pd.DataFrame:
    prepared = canonicalize_frame(enrich_external(frame))
    for col in features:
        if col not in prepared:
            prepared[col] = np.nan
    probabilities = predict_ensemble(models, prepared[features])
    ids = frame.get("Case_ID", frame.get("ID", pd.Series(range(len(frame))))).astype(str)
    out = pd.DataFrame({
        "evaluation_set": set_name, "record_id": ids,
        "true_class": frame["Esito_ML"].astype(int).to_numpy(),
        "P_failed": probabilities[:, 0], "P_amorphous": probabilities[:, 1],
        "P_crystalline": probabilities[:, 2],
        "predicted_class": probabilities.argmax(axis=1),
    })
    if "MOF" in frame:
        out["MOF"] = frame["MOF"].astype(str).to_numpy()
    return out


def main():
    raw = pd.read_csv(ROOT / "data/knowledge_database.csv")
    canonical = canonicalize_frame(raw)
    raw["_model_group"] = raw["Legante"].map(canonicalize_ligand_for_model).str.casefold()
    canonical["_model_group"] = canonical["Legante"].str.casefold()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(splitter.split(canonical, canonical["Esito_ML"], canonical["_model_group"]))

    variants = [
        ("historical_v8_schema", raw, NUMERIC_WITH_VOLUME),
        ("canonical_with_volume", canonical, NUMERIC_WITH_VOLUME),
        ("canonical_no_volume", canonical, NUMERIC_NO_VOLUME),
    ]
    summary, fold_frames, prediction_frames = {}, [], []
    for name, data, numeric in variants:
        result, folds, predictions = grouped_cv_variant(name, data, numeric, outer_splits)
        summary[name] = result
        fold_frames.append(folds)
        prediction_frames.append(predictions)

    candidate_models, candidate_features = fit_full(canonical, NUMERIC_NO_VOLUME)
    artifact = {
        "version": "10.7.0-candidate", "rf_model": candidate_models[0],
        "ligand_text_model": candidate_models[1], "weights": WEIGHTS.tolist(),
        "features": candidate_features, "classes": CLASSES.tolist(),
        "training_records": int(len(canonical)),
        "training_policy": "legacy_1078_only; laboratory and literature evaluation-only",
        "volume_feature_removed": True, "canonical_vocabulary": True,
    }
    joblib.dump(artifact, ROOT / "models/MOF_ChemAware_Ensemble_v10_7_candidate.joblib")

    schema = {
        "version": "10.7.0-candidate", "features": candidate_features,
        "numeric": NUMERIC_NO_VOLUME, "categorical": CATEGORICAL,
        "text": "Ligand_Text", "classes": {
            "0": "Failed", "1": "Amorphous/uncertain", "2": "Crystalline MOF"
        },
        "notes": [
            "Common linker aliases and public UI families are canonicalized.",
            "Solvent volume is retained for validity checks but excluded from prediction because historical missingness leaked the outcome.",
            "Laboratory and literature records were reserved for evaluation.",
        ],
    }
    (ROOT / "models/feature_schema_v10_7_candidate.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    integrated = pd.read_csv(ROOT / "data/knowledge_database_integrated_v10_6.csv")
    lab = integrated[integrated["Source_Type"].eq("laboratory_experiment")].copy()
    lab_predictions = external_predictions(candidate_models, candidate_features, lab, "laboratory_held_out")
    lab_metrics = metrics(lab_predictions.true_class.to_numpy(), lab_predictions[[
        "P_failed", "P_amorphous", "P_crystalline"
    ]].to_numpy())

    literature = pd.read_csv(ROOT / "data/literature_crystalline_challenge_v10_6.csv")
    literature["Esito_ML"] = 2
    lit_predictions = external_predictions(candidate_models, candidate_features, literature, "literature_positive")
    lit_metrics = {
        "n": int(len(lit_predictions)),
        "crystalline_argmax_recall": float((lit_predictions.predicted_class == 2).mean()),
        "p_crystalline_at_least_0_5": float((lit_predictions.P_crystalline >= 0.5).mean()),
        "mean_p_crystalline": float(lit_predictions.P_crystalline.mean()),
    }
    summary["candidate_external_evaluation"] = {
        "laboratory_held_out": lab_metrics, "literature_positive": lit_metrics,
    }

    pd.concat(fold_frames, ignore_index=True).to_csv(
        ROOT / "reports/grouped_cv_fold_metrics_v10_7_candidate.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        ROOT / "reports/grouped_cv_predictions_v10_7_candidate.csv", index=False
    )
    pd.concat([lab_predictions, lit_predictions], ignore_index=True).to_csv(
        ROOT / "reports/external_predictions_v10_7_candidate.csv", index=False
    )
    (ROOT / "reports/predictor_validation_v10_7_candidate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
