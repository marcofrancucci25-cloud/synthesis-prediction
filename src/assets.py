from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from .constants import DATA_DIR, MODEL_DIR, REPORT_DIR

@dataclass(frozen=True)
class Assets:
    model: Any
    applicability: dict
    schema: dict
    database: pd.DataFrame
    external_metrics: dict
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame


def validate_files() -> None:
    required = [
        MODEL_DIR / "MOF_RandomForest_Calibrated_v6_3.joblib",
        MODEL_DIR / "applicability_domain_v7_0.joblib",
        MODEL_DIR / "feature_schema.json",
        DATA_DIR / "knowledge_database.csv",
        REPORT_DIR / "external_test_metrics.json",
        REPORT_DIR / "external_class_metrics.csv",
        REPORT_DIR / "external_confusion_matrix.csv",
    ]
    missing = [str(p.relative_to(p.parents[1])) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("File richiesti mancanti: " + ", ".join(missing))


def load_assets() -> Assets:
    validate_files()
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    database = pd.read_csv(DATA_DIR / "knowledge_database.csv")
    required_cols = set(schema["feature_order"]) | {"Esito_ML", "Legante", "ID"}
    absent = sorted(required_cols - set(database.columns))
    if absent:
        raise ValueError("Colonne mancanti nel database: " + ", ".join(absent))
    return Assets(
        model=joblib.load(MODEL_DIR / "MOF_RandomForest_Calibrated_v6_3.joblib"),
        applicability=joblib.load(MODEL_DIR / "applicability_domain_v7_0.joblib"),
        schema=schema,
        database=database,
        external_metrics=json.loads((REPORT_DIR / "external_test_metrics.json").read_text(encoding="utf-8")),
        class_metrics=pd.read_csv(REPORT_DIR / "external_class_metrics.csv"),
        confusion_matrix=pd.read_csv(REPORT_DIR / "external_confusion_matrix.csv", index_col=0),
    )
