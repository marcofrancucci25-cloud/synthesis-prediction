from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class PredictionResult:
    predicted_class: int
    probabilities: np.ndarray
    distance: float
    domain_label: str
    domain_score: float
    unseen_categories: list[str]

class MOFSynthesisEngine:
    def __init__(self, assets: Any):
        self.model = assets.model
        self.ad = assets.applicability
        self.schema = assets.schema
        self.db = assets.database
        self.features = self.schema["feature_order"]
        self.numeric = self.schema["numeric_features"]
        self.categorical = self.schema["categorical_features"]
        self.category_sets = {c: set(self.db[c].dropna().astype(str)) for c in self.categorical}

    def options(self, column: str) -> list[str]:
        return sorted(self.db[column].dropna().astype(str).unique().tolist())

    def numeric_default(self, column: str) -> float:
        return float(pd.to_numeric(self.db[column], errors="coerce").median())

    def normalize_row(self, row: dict) -> pd.DataFrame:
        return pd.DataFrame([{f: row.get(f, np.nan) for f in self.features}])

    def probabilities(self, row: dict) -> np.ndarray:
        return self.model.predict_proba(self.normalize_row(row))[0]

    def assess_domain(self, row: dict) -> tuple[float, str, float, list[str]]:
        frame = self.normalize_row(row)
        transformed = self.ad["preprocessor"].transform(frame)
        distance = float(self.ad["neighbors"].kneighbors(transformed, n_neighbors=5)[0].mean())
        q75, q95 = float(self.ad["q75"]), float(self.ad["q95"])
        if distance <= q75:
            label, score = "Dentro il dominio", 1.0
        elif distance <= q95:
            label = "Zona intermedia"
            score = max(0.55, 1 - (distance-q75)/(q95-q75+1e-9)*0.45)
        else:
            label = "Fuori dominio"
            score = max(0.15, q95/max(distance, 1e-9)*0.5)
        unseen = [c for c in self.categorical if str(row.get(c, "")) not in self.category_sets[c]]
        if unseen:
            label, score = "Fuori dominio", min(score, 0.35)
        return distance, label, float(score), unseen

    def predict(self, row: dict) -> PredictionResult:
        p = self.probabilities(row)
        distance, label, score, unseen = self.assess_domain(row)
        return PredictionResult(int(np.argmax(p)), p, distance, label, score, unseen)

    def local_sensitivity(self, row: dict) -> pd.DataFrame:
        baseline = self.probabilities(row)[2]
        reference = {
            f: self.numeric_default(f) if f in self.numeric else self.db[f].mode(dropna=True).iloc[0]
            for f in self.features
        }
        rows = []
        for feature in self.features:
            altered = row.copy(); altered[feature] = reference[feature]
            rows.append({
                "Feature": feature,
                "Valore inserito": row[feature],
                "Riferimento database": reference[feature],
                "Variazione P(cristallino)": baseline - self.probabilities(altered)[2],
            })
        return pd.DataFrame(rows).sort_values(
            "Variazione P(cristallino)", key=lambda s: s.abs(), ascending=False
        )

    def knowledge_filter(self, ligand: str, family: str, metal: str) -> pd.DataFrame:
        sub = self.db.copy()
        if ligand != "Tutti": sub = sub[sub["Legante"].astype(str) == ligand]
        if family != "Tutte": sub = sub[sub["Famiglia_Legante"].astype(str) == family]
        if metal != "Tutti": sub = sub[sub["Metallo"].astype(str) == metal]
        return sub

    def optimize(self, base: dict, n: int, vary_solvent: bool, vary_additive: bool) -> pd.DataFrame:
        family_db = self.db[(self.db["Famiglia_Legante"].astype(str) == str(base["Famiglia_Legante"])) &
                            (self.db["Metallo"].astype(str) == str(base["Metallo"]))]
        if len(family_db) < 5:
            family_db = self.db[self.db["Famiglia_Legante"].astype(str) == str(base["Famiglia_Legante"])]
        temps = np.unique(np.clip([base["Temperatura_C"] + x for x in (-20,-10,0,10,20)], 20, 250))
        times = np.unique(np.clip([base["Tempo_ore"] * x for x in (0.5,1,1.5,2)], 0.25, 336))
        ratios = np.unique(np.clip([base["Rapporto_LM"] * x for x in (0.67,1,1.5)], 0.1, 10))
        solvents = family_db["Solvente"].value_counts().head(6).index.tolist() if vary_solvent and len(family_db) else [base["Solvente"]]
        additives = family_db["Additivo_Colinker"].value_counts().head(5).index.tolist() if vary_additive and len(family_db) else [base["Additivo_Colinker"]]
        candidates = []
        for temp, hours, ratio, solvent, additive in itertools.product(temps, times, ratios, solvents, additives):
            row = base.copy()
            row.update({"Temperatura_C": float(temp), "Tempo_ore": float(hours), "Rapporto_LM": float(ratio),
                        "Solvente": str(solvent), "Additivo_Colinker": str(additive)})
            p = self.probabilities(row)
            distance, domain, ad_score, unseen = self.assess_domain(row)
            candidates.append({**row, "P_fallimento": p[0], "P_amorfo": p[1], "P_cristallino": p[2],
                               "AD_score": ad_score, "AD_distance": distance, "Dominio": domain,
                               "Categorie_non_viste": ", ".join(unseen),
                               "Ranking_score": p[2] * (0.65 + 0.35 * ad_score)})
        return (pd.DataFrame(candidates).drop_duplicates(subset=self.features)
                .sort_values(["Ranking_score", "P_cristallino"], ascending=False).head(n).reset_index(drop=True))
