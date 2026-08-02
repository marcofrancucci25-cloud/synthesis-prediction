"""Run the frozen v8 predictor on a positive-only literature challenge set.

This is a diagnostic recall test, not a complete external validation: every
record has a crystalline expected outcome, and the historical training data do
not contain source DOIs, so overlap with training literature cannot be ruled out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine import TRAINING_DB, applicability, predict, prediction_validity


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "literature_crystalline_challenge_v10_6.csv"
OUTPUT = ROOT / "reports" / "literature_crystalline_benchmark_v10_6.csv"
SUMMARY = ROOT / "reports" / "literature_crystalline_benchmark_v10_6.json"

MODEL_FIELDS = [
    "Legante", "Famiglia_Legante", "Metallo", "Sale_Metallico", "Solvente",
    "Additivo_Colinker", "Temperatura_C", "Tempo_ore", "mmol_Legante",
    "mmol_Sale", "Rapporto_LM", "Volume solvente", "Hydration_Number",
    "Counterion_Class", "Oxidation_State",
]
LABELS = {0: "failed", 1: "amorphous", 2: "crystalline"}


def run() -> tuple[pd.DataFrame, dict]:
    challenge = pd.read_csv(INPUT)
    results: list[dict] = []

    for _, record in challenge.iterrows():
        values = {field: record[field] for field in MODEL_FIELDS}
        _, probabilities, predicted = predict(values)
        ad = applicability(values)
        validity = prediction_validity(values)

        missing_volume_values = dict(values)
        missing_volume_values["Volume solvente"] = np.nan
        _, missing_volume_probabilities, missing_volume_predicted = predict(missing_volume_values)

        exact_identity = (
            (TRAINING_DB["Legante"].astype(str).str.casefold() == str(values["Legante"]).casefold())
            & (TRAINING_DB["Metallo"].astype(str) == str(values["Metallo"]))
            & (TRAINING_DB["Sale_Metallico"].astype(str) == str(values["Sale_Metallico"]))
        )
        results.append({
            **record.to_dict(),
            "P_failed": float(probabilities[0]),
            "P_amorphous": float(probabilities[1]),
            "P_crystalline": float(probabilities[2]),
            "Predicted_Class": int(predicted),
            "Predicted_Label": LABELS[int(predicted)],
            "Correct_Crystalline": bool(int(predicted) == 2),
            "Applicability_Label": ad["label"],
            "Applicability_Score": float(ad["score"]),
            "Numerical_Validity_Label": validity["label"],
            "Ligand_Seen_Exact": bool(ad["ligand_seen"]),
            "Salt_Seen_Exact": bool(ad["salt_seen"]),
            "Training_Identity_Rows": int(exact_identity.sum()),
            "P_crystalline_if_volume_missing": float(missing_volume_probabilities[2]),
            "Predicted_if_volume_missing": LABELS[int(missing_volume_predicted)],
            "Volume_Missing_Delta_Pcrystalline": float(missing_volume_probabilities[2] - probabilities[2]),
        })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT, index=False)

    volume_missing_by_class = (
        TRAINING_DB.assign(Volume_Missing=TRAINING_DB["Volume solvente"].isna())
        .groupby("Esito_ML")["Volume_Missing"]
        .mean()
        .to_dict()
    )
    summary = {
        "challenge_records": int(len(output)),
        "crystalline_predictions": int(output["Correct_Crystalline"].sum()),
        "positive_recall": float(output["Correct_Crystalline"].mean()),
        "false_negative_records": output.loc[~output["Correct_Crystalline"], "Case_ID"].tolist(),
        "mean_p_crystalline": float(output["P_crystalline"].mean()),
        "median_p_crystalline": float(output["P_crystalline"].median()),
        "mean_volume_missing_delta_p_crystalline": float(output["Volume_Missing_Delta_Pcrystalline"].mean()),
        "training_volume_missing_fraction": float(TRAINING_DB["Volume solvente"].isna().mean()),
        "training_volume_missing_fraction_by_class": {str(k): float(v) for k, v in volume_missing_by_class.items()},
        "ui_family_categories_seen_by_training": [],
        "limitations": [
            "Positive-only challenge set: specificity and balanced accuracy cannot be estimated.",
            "Training-source DOI provenance is absent, so literature overlap cannot be excluded.",
            "Some multistage protocols are compressed into the single temperature/time representation supported by v8.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary


if __name__ == "__main__":
    frame, metrics = run()
    print(frame[["Case_ID", "MOF", "P_failed", "P_amorphous", "P_crystalline", "Predicted_Label", "Applicability_Label"]].to_string(index=False))
    print(json.dumps(metrics, indent=2))
