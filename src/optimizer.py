from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import COUNTERIONS, build_row, precursor_formula

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_schema_v8_0.json"
try:
    _NUMERIC_FEATURES = set(json.loads(_SCHEMA_PATH.read_text()).get("numeric", []))
except Exception:
    _NUMERIC_FEATURES = set()


def _coerce_numeric_features(x: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Same defensive coercion used by the predictor (see src/engine.py).

    Candidate rows are built programmatically here and are numeric by
    construction, but this keeps the optimizer's scoring path exception-safe
    against any future candidate-generation path that is not.
    """
    present = [c for c in features if c in _NUMERIC_FEATURES and c in x]
    if present:
        x = x.copy()
        x.loc[:, present] = x.loc[:, present].apply(pd.to_numeric, errors="coerce")
    return x


OBJECTIVES = {
    "Maximum crystallinity": {
        "p_crystalline": 0.57, "positive_support": 0.20, "domain": 0.13,
        "feasibility": 0.10, "change": 0.00, "green": 0.00, "speed": 0.00,
    },
    "Balanced conditions": {
        "p_crystalline": 0.43, "positive_support": 0.18, "domain": 0.16,
        "feasibility": 0.10, "change": 0.07, "green": 0.035, "speed": 0.025,
    },
    "Conservative optimization": {
        "p_crystalline": 0.38, "positive_support": 0.18, "domain": 0.16,
        "feasibility": 0.10, "change": 0.15, "green": 0.015, "speed": 0.015,
    },
    "Green synthesis": {
        "p_crystalline": 0.34, "positive_support": 0.17, "domain": 0.13,
        "feasibility": 0.10, "change": 0.03, "green": 0.19, "speed": 0.04,
    },
    "Fast synthesis": {
        "p_crystalline": 0.36, "positive_support": 0.17, "domain": 0.14,
        "feasibility": 0.10, "change": 0.025, "green": 0.05, "speed": 0.155,
    },
}

SOLVENT_GREEN_PENALTY = {
    "water": 0.00, "h2o": 0.00, "ethanol": 0.08, "etoh": 0.08,
    "methanol": 0.14, "meoh": 0.14, "2-propanol": 0.10, "isopropanol": 0.10,
    "acetone": 0.12, "ethyl acetate": 0.08, "acetonitrile": 0.22, "mecn": 0.22,
    "dmf": 0.55, "def": 0.55, "dma": 0.50, "dmso": 0.28,
    "nmp": 0.60, "thf": 0.35, "toluene": 0.48, "dichloromethane": 0.75,
    "ch2cl2": 0.75, "chloroform": 0.80,
}


def _norm(x: Any) -> str:
    return str(x if x is not None else "").strip().casefold()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _quantile_bounds(db: pd.DataFrame, col: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    values = _numeric(db[col]) if col in db else pd.Series(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return fallback
    lo, hi = float(values.quantile(0.03)), float(values.quantile(0.97))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return fallback
    return lo, hi


def _pool(db: pd.DataFrame, col: str, n: int, default: Sequence[Any]) -> List[Any]:
    if col not in db:
        return list(default)
    values = db[col].dropna().astype(str).str.strip()
    values = values[values.ne("")]
    out = list(values.value_counts().head(n).index)
    return out or list(default)


def _counterion_pool(db: pd.DataFrame, metal: str, n: int = 8) -> List[str]:
    sub = db[db["Metallo"].astype(str).eq(str(metal))] if "Metallo" in db else db.iloc[0:0]
    values = sub.get("Counterion_Class", pd.Series(dtype=str)).dropna().astype(str)
    ranked = list(values[values.ne("")].value_counts().head(n).index)
    for item in COUNTERIONS:
        if item not in ranked:
            ranked.append(item)
    return ranked[:max(n, 6)]


def _oxidation_pool(db: pd.DataFrame, metal: str, current: Any) -> List[int]:
    sub = db[db["Metallo"].astype(str).eq(str(metal))] if "Metallo" in db else db.iloc[0:0]
    vals = pd.to_numeric(sub.get("Oxidation_State", pd.Series(dtype=float)), errors="coerce")
    vals = sorted({int(v) for v in vals.dropna() if 1 <= float(v) <= 8})
    try:
        cur = int(float(current))
        if cur not in vals:
            vals.append(cur)
    except Exception:
        pass
    return sorted(vals) or [2, 3]


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    low = max(float(low), 1e-8)
    high = max(float(high), low * 1.0001)
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def _contains_solvent(text: str, token: str) -> bool:
    s, t = _norm(text), _norm(token)
    return t in s or s in t


def _green_penalty(solvent: str) -> float:
    components = re.split(r"[/+;,:]", _norm(solvent))
    penalties = []
    for comp in components:
        comp = comp.strip()
        if not comp:
            continue
        penalties.append(max((v for k, v in SOLVENT_GREEN_PENALTY.items() if k in comp), default=0.30))
    return float(np.mean(penalties)) if penalties else 0.30


def _change_penalty(base: Dict[str, Any], row: pd.Series, bounds: Dict[str, Tuple[float, float]]) -> float:
    cat_fields = ["Counterion_Class", "Solvente", "Additivo_Colinker"]
    cat = sum(_norm(base.get(f)) != _norm(row.get(f)) for f in cat_fields) / len(cat_fields)
    nums = []
    for f in ["Hydration_Number", "Oxidation_State", "Temperatura_C", "Tempo_ore", "Rapporto_LM", "Volume solvente", "mmol_Sale"]:
        try:
            lo, hi = bounds.get(f, (0.0, 1.0))
            span = max(hi - lo, 1e-8)
            nums.append(min(abs(float(row[f]) - float(base.get(f, row[f]))) / span, 1.0))
        except Exception:
            pass
    return float(0.45 * cat + 0.55 * (np.mean(nums) if nums else 0.0))


def _feasibility(row: pd.Series, constraints: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 1.0
    t, h = float(row["Temperatura_C"]), float(row["Tempo_ore"])
    ratio, vol = float(row["Rapporto_LM"]), float(row["Volume solvente"])
    mmol_m, mmol_l = float(row["mmol_Sale"]), float(row["mmol_Legante"])
    concentration = (mmol_m + mmol_l) / max(vol, 1e-8)
    if ratio <= 0 or mmol_m <= 0 or mmol_l <= 0 or vol <= 0:
        return 0.0, ["Non-positive amount, ratio or volume"]
    if not (0.02 <= concentration <= 5.0):
        score -= 0.45
        reasons.append("Total concentration outside the broad observed/plausible range")
    if t > 220:
        score -= 0.20
        reasons.append("Very high synthesis temperature")
    if h > 168:
        score -= 0.15
        reasons.append("Very long reaction time")
    if ratio > 10:
        score -= 0.20
        reasons.append("Extreme ligand/metal ratio")
    if constraints.get("max_temperature") is not None and t > float(constraints["max_temperature"]):
        return 0.0, ["Temperature exceeds user constraint"]
    if constraints.get("max_time") is not None and h > float(constraints["max_time"]):
        return 0.0, ["Time exceeds user constraint"]
    banned = [_norm(x) for x in constraints.get("banned_solvents", []) if _norm(x)]
    if any(_contains_solvent(row["Solvente"], b) for b in banned):
        return 0.0, ["Solvent excluded by user"]
    allowed = [_norm(x) for x in constraints.get("allowed_solvents", []) if _norm(x)]
    if allowed and not any(_contains_solvent(row["Solvente"], a) for a in allowed):
        return 0.0, ["Solvent not in user allow-list"]
    return max(0.0, float(score)), reasons


def _domain_score(db: pd.DataFrame, base: Dict[str, Any], candidates: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    ligand, metal = _norm(base.get("Legante")), str(base.get("Metallo", ""))
    ligand_seen = ligand in set(db["Legante"].astype(str).map(_norm)) if "Legante" in db else False
    metal_seen = metal in set(db["Metallo"].astype(str)) if "Metallo" in db else False
    salts = set(db.get("Sale_Metallico", pd.Series(dtype=str)).astype(str))
    solvents = set(db.get("Solvente", pd.Series(dtype=str)).astype(str))
    additives = set(db.get("Additivo_Colinker", pd.Series(dtype=str)).fillna("Nessuno").astype(str))
    scores = []
    for _, r in candidates.iterrows():
        s = 0.30 * float(ligand_seen) + 0.20 * float(metal_seen)
        s += 0.15 * float(str(r["Sale_Metallico"]) in salts)
        s += 0.12 * float(str(r["Solvente"]) in solvents)
        s += 0.08 * float(str(r["Additivo_Colinker"]) in additives)
        numeric = 0.0
        for f in ["Temperatura_C", "Tempo_ore", "Rapporto_LM", "Volume solvente", "Hydration_Number"]:
            lo, hi = bounds[f]
            val = float(r[f])
            if lo <= val <= hi:
                numeric += 1.0
            else:
                distance = min(abs(val - lo), abs(val - hi)) / max(hi - lo, 1e-8)
                numeric += max(0.0, 1.0 - distance)
        s += 0.15 * (numeric / 5.0)
        scores.append(min(max(s, 0.0), 1.0))
    return np.asarray(scores, float)


def _nondominated(df: pd.DataFrame, objectives: Sequence[Tuple[str, bool]]) -> pd.Series:
    values = df[[c for c, _ in objectives]].to_numpy(float)
    maximize = np.array([mx for _, mx in objectives], bool)
    adjusted = np.where(maximize, values, -values)
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dominates = np.all(adjusted >= adjusted[i], axis=1) & np.any(adjusted > adjusted[i], axis=1)
        if np.any(dominates):
            keep[i] = False
    return pd.Series(keep, index=df.index)


def _select_positive_templates(positive_db: pd.DataFrame, base: Dict[str, Any]) -> pd.DataFrame:
    """Select successful records by decreasing chemical relevance."""
    p = positive_db.copy()
    if p.empty:
        return p
    ligand, family, metal = _norm(base.get("Legante")), _norm(base.get("Famiglia_Legante")), str(base.get("Metallo", ""))
    p["_template_weight"] = 0.05
    p.loc[p["Metallo"].astype(str).eq(metal), "_template_weight"] += 0.40
    p.loc[p["Legante"].astype(str).map(_norm).eq(ligand), "_template_weight"] += 0.38
    p.loc[p["Famiglia_Legante"].astype(str).map(_norm).eq(family), "_template_weight"] += 0.17
    # The metal is fixed by design. Prefer same-metal templates, but retain family analogues as fallback.
    same_metal = p[p["Metallo"].astype(str).eq(metal)]
    if len(same_metal) >= 8:
        p = same_metal
    return p.sort_values("_template_weight", ascending=False).head(500).reset_index(drop=True)


def _template_candidate(rng: np.random.Generator, base: Dict[str, Any], template: pd.Series,
                        bounds: Dict[str, Tuple[float, float]], constraints: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(base)
    # Keep metal and ligand fixed; borrow the successful joint condition pattern.
    ox_raw = pd.to_numeric(pd.Series([template.get("Oxidation_State")]), errors="coerce").iloc[0]
    if not np.isfinite(ox_raw):
        ox_raw = pd.to_numeric(pd.Series([base.get("Oxidation_State", 2)]), errors="coerce").iloc[0]
    oxidation = int(float(ox_raw)) if np.isfinite(ox_raw) else 2
    counterion_raw = template.get("Counterion_Class")
    counterion = str(counterion_raw) if pd.notna(counterion_raw) and str(counterion_raw).strip() else str(base.get("Counterion_Class", "nitrate"))
    hyd_raw = pd.to_numeric(pd.Series([template.get("Hydration_Number")]), errors="coerce").iloc[0]
    if not np.isfinite(hyd_raw):
        hyd_raw = pd.to_numeric(pd.Series([base.get("Hydration_Number", 0)]), errors="coerce").iloc[0]
    hydration = float(hyd_raw) if np.isfinite(hyd_raw) else 0.0
    if constraints.get("keep_precursor"):
        oxidation = int(float(base.get("Oxidation_State", 2) or 2))
        counterion = str(base.get("Counterion_Class", "nitrate"))
        hydration = float(base.get("Hydration_Number", 0) or 0)
    else:
        hydration = float(np.clip(hydration + rng.normal(0, 0.5), 0, 12))
        hydration = round(hydration * 2) / 2

    solvent = str(template.get("Solvente") or base.get("Solvente", "DMF"))
    additive = str(template.get("Additivo_Colinker") or "Nessuno")
    if constraints.get("keep_solvent"):
        solvent = str(base.get("Solvente", "DMF"))
    if constraints.get("keep_additive"):
        additive = str(base.get("Additivo_Colinker", "Nessuno"))

    def jitter(col: str, default: float, sigma: float, log: bool = False) -> float:
        val = pd.to_numeric(pd.Series([template.get(col)]), errors="coerce").iloc[0]
        val = float(val) if np.isfinite(val) else float(default)
        if log:
            val *= float(np.exp(rng.normal(0, sigma)))
        else:
            val += float(rng.normal(0, sigma))
        lo, hi = bounds[col]
        return float(np.clip(val, lo, hi))

    temperature = jitter("Temperatura_C", base.get("Temperatura_C", 120), 9.0)
    time = jitter("Tempo_ore", base.get("Tempo_ore", 24), 0.28, log=True)
    ratio = jitter("Rapporto_LM", base.get("Rapporto_LM", 1), 0.22, log=True)
    mmol_m = jitter("mmol_Sale", base.get("mmol_Sale", 0.1), 0.30, log=True)
    volume = jitter("Volume solvente", base.get("Volume solvente", 10), 0.25, log=True)
    mmol_l = ratio * mmol_m
    row.update({
        "Oxidation_State": oxidation, "Counterion_Class": counterion,
        "Hydration_Number": hydration,
        "Sale_Metallico": precursor_formula(base.get("Metallo"), oxidation, counterion, hydration),
        "Solvente": solvent, "Additivo_Colinker": additive,
        "Temperatura_C": temperature, "Tempo_ore": time,
        "mmol_Sale": mmol_m, "mmol_Legante": mmol_l,
        "Rapporto_LM": ratio, "Volume solvente": volume,
        "Generation_mode": "successful-template mutation",
        "Template_Positive_ID": template.get("Positive_ID", template.get("ID", "")),
        "Template_Source": template.get("Positive_Library_Source", "successful library"),
    })
    return row


def _exploration_candidate(rng: np.random.Generator, base: Dict[str, Any], db: pd.DataFrame,
                           bounds: Dict[str, Tuple[float, float]], constraints: Dict[str, Any],
                           solvents: List[str], additives: List[str], counterions: List[str], oxidations: List[int]) -> Dict[str, Any]:
    oxidation = int(rng.choice(oxidations))
    counterion = str(rng.choice(counterions))
    hydration = float(base.get("Hydration_Number", 0) or 0) if constraints.get("keep_precursor") else float(np.round(rng.uniform(0, min(12, bounds["Hydration_Number"][1])) * 2) / 2)
    ratio = float(_sample_log_uniform(rng, bounds["Rapporto_LM"][0], bounds["Rapporto_LM"][1], 1)[0])
    mmol_m = float(_sample_log_uniform(rng, max(0.001, bounds["mmol_Sale"][0]), bounds["mmol_Sale"][1], 1)[0])
    volume = float(_sample_log_uniform(rng, max(0.2, bounds["Volume solvente"][0]), bounds["Volume solvente"][1], 1)[0])
    row = dict(base)
    row.update({
        "Oxidation_State": oxidation, "Counterion_Class": counterion,
        "Hydration_Number": hydration,
        "Sale_Metallico": precursor_formula(base.get("Metallo"), oxidation, counterion, hydration),
        "Solvente": str(rng.choice(solvents)), "Additivo_Colinker": str(rng.choice(additives)),
        "Temperatura_C": float(rng.uniform(*bounds["Temperatura_C"])),
        "Tempo_ore": float(_sample_log_uniform(rng, *bounds["Tempo_ore"], 1)[0]),
        "mmol_Sale": mmol_m, "mmol_Legante": ratio * mmol_m,
        "Rapporto_LM": ratio, "Volume solvente": volume,
        "Generation_mode": "broad exploration",
        "Template_Positive_ID": "", "Template_Source": "",
    })
    return row


def _positive_support(positive_db: pd.DataFrame, base: Dict[str, Any], candidates: pd.DataFrame,
                      bounds: Dict[str, Tuple[float, float]], positive_model: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Estimate support from successful syntheses.

    When available, v10.4 uses a fitted conditional nearest-neighbour model built
    from ligand text, metal/family, precursor, solvent, additive and numerical
    conditions. Evidence quality and diversity weights temper the score. This is
    a literature/positive-precedent plausibility score, not a success probability.
    """
    if positive_db is None or positive_db.empty:
        out = pd.DataFrame(index=candidates.index)
        out["Positive_support_score"] = 0.0
        out["Positive_support_count"] = 0
        out["Nearest_positive_similarity"] = 0.0
        out["Nearest_positive_ID"] = ""
        out["Evidence_tier"] = "No positive library"
        return out

    if positive_model is not None and all(k in positive_model for k in ["preprocessor", "nn", "reference"]):
        query = candidates.copy()
        # Ligand and family are fixed by design but may not be materialized in the candidate frame.
        for field in ["Legante", "Famiglia_Legante", "Metallo", "Procedura_Sintetica"]:
            if field not in query:
                query[field] = base.get(field, "Unknown")
            else:
                query[field] = query[field].fillna(base.get(field, "Unknown"))
        Xq = positive_model["preprocessor"].transform(query)
        n_neighbors = min(15, len(positive_model["reference"]))
        distances, indices = positive_model["nn"].kneighbors(Xq, n_neighbors=n_neighbors)
        refs = positive_model["reference"].reset_index(drop=True)
        scale = float(positive_model.get("support_scale", 0.32))
        records = []
        for dist, idxs in zip(distances, indices):
            neigh = refs.iloc[idxs]
            similarities = np.exp(-np.asarray(dist, float) / max(scale, 1e-6))
            evidence = pd.to_numeric(neigh.get("Evidence_Weight", 1.0), errors="coerce").fillna(0.65).to_numpy(float)
            weighted = similarities * evidence
            support = float(np.clip(0.65 * np.max(weighted) + 0.35 * np.average(weighted, weights=np.maximum(evidence, 1e-6)), 0, 1))
            count = int(np.sum(weighted >= 0.65))
            best = int(np.argmax(weighted))
            if support >= 0.74 and count >= 2:
                tier = "Strong positive precedent"
            elif support >= 0.52:
                tier = "Moderate positive precedent"
            else:
                tier = "Limited positive precedent"
            records.append({
                "Positive_support_score": support,
                "Positive_support_count": count,
                "Nearest_positive_similarity": float(similarities[best]),
                "Nearest_positive_ID": str(neigh.iloc[best].get("Positive_ID", "")),
                "Nearest_positive_quality": str(neigh.iloc[best].get("Quality_Tier", "")),
                "Evidence_tier": tier,
            })
        return pd.DataFrame(records, index=candidates.index)

    # Backward-compatible heuristic for deployments missing the fitted artifact.
    refs = _select_positive_templates(positive_db, base)
    if refs.empty:
        refs = positive_db.head(500).copy()
    num_fields = ["Temperatura_C", "Tempo_ore", "Rapporto_LM", "Volume solvente", "Hydration_Number"]
    cat_fields = ["Metallo", "Legante", "Famiglia_Legante", "Counterion_Class", "Solvente", "Additivo_Colinker"]
    cat_weights = np.array([0.20, 0.23, 0.12, 0.09, 0.09, 0.04], float)
    num_weights = np.array([0.075, 0.065, 0.055, 0.045, 0.03], float)
    ref_cat = {f: refs.get(f, pd.Series("", index=refs.index)).fillna("").astype(str).map(_norm).to_numpy() for f in cat_fields}
    ref_num = {f: pd.to_numeric(refs.get(f, pd.Series(np.nan, index=refs.index)), errors="coerce").to_numpy(float) for f in num_fields}
    ref_ids = refs.get("Positive_ID", refs.get("ID", pd.Series("", index=refs.index))).astype(str).to_numpy()
    records = []
    for _, c in candidates.iterrows():
        sim = np.zeros(len(refs), float)
        for w, f in zip(cat_weights, cat_fields):
            sim += w * (ref_cat[f] == _norm(c.get(f))).astype(float)
        for w, f in zip(num_weights, num_fields):
            lo, hi = bounds[f]; scale = max(hi - lo, 1e-8); vals = ref_num[f]
            cv = pd.to_numeric(pd.Series([c.get(f)]), errors="coerce").iloc[0]
            component = np.where(np.isfinite(vals) & np.isfinite(cv), np.exp(-np.abs(vals - float(cv)) / (0.22 * scale + 1e-8)), 0.35)
            sim += w * component
        order = np.argsort(sim)[::-1]; top = sim[order[: min(5, len(order))]]
        maximum = float(top[0]) if len(top) else 0.0; top_mean = float(np.mean(top)) if len(top) else 0.0
        support = float(np.clip(0.70 * maximum + 0.30 * top_mean, 0, 1)); count = int(np.sum(sim >= 0.72))
        tier = "Strong positive precedent" if support >= 0.78 and count >= 3 else ("Moderate positive precedent" if support >= 0.62 else "Limited positive precedent")
        records.append({"Positive_support_score": support,"Positive_support_count": count,"Nearest_positive_similarity": maximum,"Nearest_positive_ID": ref_ids[order[0]] if len(order) else "","Nearest_positive_quality":"","Evidence_tier": tier})
    return pd.DataFrame(records, index=candidates.index)

def joint_optimize(
    base: Dict[str, Any], model_artifact: Dict[str, Any], features: Sequence[str], db: pd.DataFrame,
    positive_db: Optional[pd.DataFrame] = None, positive_model: Optional[Dict[str, Any]] = None,
    objective: str = "Balanced conditions", n_samples: int = 2500, top_n: int = 12,
    constraints: Optional[Dict[str, Any]] = None, random_state: int = 260730,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Hybrid joint optimizer.

    The balanced three-class predictor estimates outcome risk. A separate positive
    synthesis layer supplies coherent successful-condition templates and a support
    score. Only ligand and metal remain fixed.
    """
    constraints = dict(constraints or {})
    rng = np.random.default_rng(random_state)
    n_samples = int(np.clip(n_samples, 300, 10000))
    objective = objective if objective in OBJECTIVES else "Balanced conditions"
    weights = OBJECTIVES[objective]

    evidence_db = positive_db if positive_db is not None and not positive_db.empty else db[pd.to_numeric(db.get("Esito_ML"), errors="coerce").eq(2)].copy()
    bounds_source = evidence_db if len(evidence_db) >= 50 else db
    bounds = {
        "Temperatura_C": _quantile_bounds(bounds_source, "Temperatura_C", (40, 220)),
        "Tempo_ore": _quantile_bounds(bounds_source, "Tempo_ore", (0.5, 168)),
        "Rapporto_LM": _quantile_bounds(bounds_source, "Rapporto_LM", (0.1, 10)),
        "Volume solvente": _quantile_bounds(bounds_source, "Volume solvente", (0.5, 100)),
        "mmol_Sale": _quantile_bounds(bounds_source, "mmol_Sale", (0.005, 10)),
        "Hydration_Number": _quantile_bounds(bounds_source, "Hydration_Number", (0, 12)),
        "Oxidation_State": _quantile_bounds(bounds_source, "Oxidation_State", (1, 6)),
    }
    bounds["Temperatura_C"] = (max(bounds["Temperatura_C"][0], float(constraints.get("min_temperature", bounds["Temperatura_C"][0]))), min(bounds["Temperatura_C"][1], float(constraints.get("max_temperature", bounds["Temperatura_C"][1]))))
    bounds["Tempo_ore"] = (max(bounds["Tempo_ore"][0], float(constraints.get("min_time", bounds["Tempo_ore"][0]))), min(bounds["Tempo_ore"][1], float(constraints.get("max_time", bounds["Tempo_ore"][1]))))
    bounds["Rapporto_LM"] = (max(bounds["Rapporto_LM"][0], float(constraints.get("min_ratio", bounds["Rapporto_LM"][0]))), min(bounds["Rapporto_LM"][1], float(constraints.get("max_ratio", bounds["Rapporto_LM"][1]))))
    if any(lo >= hi for lo, hi in [bounds["Temperatura_C"], bounds["Tempo_ore"], bounds["Rapporto_LM"]]):
        raise ValueError("The selected constraints leave an empty numerical search range.")

    source_for_pools = evidence_db if not evidence_db.empty else db
    metal_success = source_for_pools[source_for_pools["Metallo"].astype(str).eq(str(base.get("Metallo", "")))]
    pool_db = metal_success if len(metal_success) >= 5 else source_for_pools
    solvents = _pool(pool_db, "Solvente", 22, [base.get("Solvente", "DMF")])
    additives = _pool(pool_db, "Additivo_Colinker", 16, [base.get("Additivo_Colinker", "Nessuno")])
    allowed = constraints.get("allowed_solvents") or []
    solvent_warning = None
    if allowed:
        matched_in_pool = [s for s in solvents if any(_contains_solvent(s, a) for a in allowed)]
        if matched_in_pool:
            solvents = matched_in_pool
        else:
            # None of the requested solvents appear in the metal-specific pool.
            # Before falling back to the literal request, check whether they
            # are recorded anywhere in the wider experimental evidence: a
            # solvent that is simply rare for this metal is not the same as
            # one with no experimental precedent at all.
            known_solvents = set(db.get("Solvente", pd.Series(dtype=str)).dropna().astype(str))
            if positive_db is not None and not positive_db.empty:
                known_solvents |= set(positive_db.get("Solvente", pd.Series(dtype=str)).dropna().astype(str))
            matched_known = [s for s in known_solvents if any(_contains_solvent(s, a) for a in allowed)]
            if matched_known:
                solvents = matched_known
                solvent_warning = (
                    "The requested allowed solvent(s) have no precedent for this specific metal; "
                    "using matches found elsewhere in the experimental database instead."
                )
            else:
                solvents = list(allowed)
                solvent_warning = (
                    "None of the requested allowed solvent(s) were found in any experimental record. "
                    "The optimizer is proceeding with them as a literal, unvalidated request: treat the "
                    "resulting proposals as unsupported by precedent, not as validated conditions."
                )
    banned = constraints.get("banned_solvents") or []
    solvents = [s for s in solvents if not any(_contains_solvent(s, b) for b in banned)]

    # "Keep the current solvent/additive" is an explicit, high-priority user
    # request and must always win: it is applied last, after allowed/banned
    # filtering, instead of being silently overridden by a later filter (as
    # allowed_solvents used to do) or left to fail with a generic empty-pool
    # error further downstream (as banned_solvents still correctly did).
    # A genuine contradiction (the kept value is itself excluded) is now
    # reported with a specific, actionable error message.
    if constraints.get("keep_solvent"):
        kept_solvent = base.get("Solvente", "DMF")
        if allowed and not any(_contains_solvent(kept_solvent, a) for a in allowed):
            raise ValueError(
                f"'Keep current solvent' conflicts with the allowed-solvent constraint: "
                f"'{kept_solvent}' is not among the allowed solvents requested."
            )
        if banned and any(_contains_solvent(kept_solvent, b) for b in banned):
            raise ValueError(
                f"'Keep current solvent' conflicts with the banned-solvent constraint: "
                f"'{kept_solvent}' is on the banned-solvent list."
            )
        solvents = [kept_solvent]
    if constraints.get("keep_additive"):
        additives = [base.get("Additivo_Colinker", "Nessuno")]
    if not solvents:
        raise ValueError("No solvent remains after applying the selected constraints.")

    counterions = [base.get("Counterion_Class", "nitrate")] if constraints.get("keep_precursor") else _counterion_pool(pool_db, base.get("Metallo"), 10)
    oxidations = [int(float(base.get("Oxidation_State", 2) or 2))] if constraints.get("keep_precursor") else _oxidation_pool(pool_db, base.get("Metallo"), base.get("Oxidation_State"))
    templates = _select_positive_templates(evidence_db, base)

    template_fraction = 0.72 if len(templates) else 0.0
    n_template = int(n_samples * template_fraction)
    rows: List[Dict[str, Any]] = []
    if n_template:
        weights_template = templates["_template_weight"].to_numpy(float)
        weights_template = weights_template / weights_template.sum()
        chosen = rng.choice(len(templates), size=n_template, replace=True, p=weights_template)
        for idx in chosen:
            rows.append(_template_candidate(rng, base, templates.iloc[int(idx)], bounds, constraints))
    for _ in range(n_samples - len(rows)):
        rows.append(_exploration_candidate(rng, base, pool_db, bounds, constraints, solvents, additives, counterions, oxidations))

    candidates = pd.DataFrame(rows)
    # Re-apply constraints to template-derived categorical choices.
    if banned:
        candidates = candidates[~candidates["Solvente"].map(lambda s: any(_contains_solvent(s, b) for b in banned))]
    feasibility, reasons = [], []
    for _, r in candidates.iterrows():
        s, rs = _feasibility(r, constraints)
        feasibility.append(s)
        reasons.append("; ".join(rs))
    candidates["Feasibility_score"] = feasibility
    candidates["Feasibility_notes"] = reasons
    candidates = candidates[candidates["Feasibility_score"] > 0].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No feasible candidate was generated under the selected constraints.")

    engineered = pd.concat([build_row(r.to_dict()) for _, r in candidates.iterrows()], ignore_index=True)
    for c in features:
        if c not in engineered:
            engineered[c] = np.nan
    x = engineered[list(features)]
    x = _coerce_numeric_features(x, features)
    probs = model_artifact["weights"][0] * model_artifact["rf_model"].predict_proba(x) + model_artifact["weights"][1] * model_artifact["ligand_text_model"].predict_proba(x)
    candidates["P_Failed"], candidates["P_Amorphous"], candidates["P_Crystalline"] = probs[:, 0], probs[:, 1], probs[:, 2]
    candidates["AD_score"] = _domain_score(db, base, candidates, bounds)
    positive_scores = _positive_support(evidence_db, base, candidates, bounds, positive_model=positive_model)
    candidates = pd.concat([candidates, positive_scores], axis=1)
    candidates["Green_penalty"] = candidates["Solvente"].map(_green_penalty)
    candidates["Speed_penalty"] = (np.clip(candidates["Temperatura_C"] / 220, 0, 1) + np.clip(np.log1p(candidates["Tempo_ore"]) / np.log1p(168), 0, 1)) / 2
    candidates["Change_penalty"] = [_change_penalty(base, r, bounds) for _, r in candidates.iterrows()]
    candidates["Optimization_score"] = (
        weights["p_crystalline"] * candidates["P_Crystalline"] +
        weights["positive_support"] * candidates["Positive_support_score"] +
        weights["domain"] * candidates["AD_score"] +
        weights["feasibility"] * candidates["Feasibility_score"] -
        weights["change"] * candidates["Change_penalty"] -
        weights["green"] * candidates["Green_penalty"] -
        weights["speed"] * candidates["Speed_penalty"]
    )

    candidates["Pareto_optimal"] = _nondominated(candidates, [
        ("P_Crystalline", True), ("Positive_support_score", True),
        ("AD_score", True), ("Feasibility_score", True),
        ("Green_penalty", False), ("Speed_penalty", False), ("Change_penalty", False),
    ])
    candidates["Total_concentration_mmol_mL"] = (candidates["mmol_Legante"] + candidates["mmol_Sale"]) / candidates["Volume solvente"]
    candidates["Objective"] = objective
    candidates["Recommendation_note"] = candidates.apply(
        lambda r: f"{r['Evidence_tier']}; generated by {r['Generation_mode']}; balanced predictor P(crystalline)={r['P_Crystalline']:.1%}.", axis=1
    )

    result = (candidates.sort_values(["Pareto_optimal", "Optimization_score", "P_Crystalline", "Positive_support_score", "AD_score"], ascending=[False, False, False, False, False])
              .drop_duplicates(["Sale_Metallico", "Solvente", "Additivo_Colinker", "Temperatura_C", "Tempo_ore", "Rapporto_LM"])
              .head(int(top_n)).reset_index(drop=True))
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    result["Strategy"] = "Alternative"
    if len(result):
        # A single row can legitimately win more than one category (e.g. the
        # best hybrid-score row can also be the least resource-intensive one).
        # Labels are accumulated per row instead of being overwritten, so a
        # later assignment never silently erases an earlier one.
        strategy_labels: Dict[int, List[str]] = {}

        def _add_label(idx: int, label: str) -> None:
            strategy_labels.setdefault(idx, [])
            if label not in strategy_labels[idx]:
                strategy_labels[idx].append(label)

        _add_label(result["P_Crystalline"].idxmax(), "Maximum probability")
        _add_label(result["Optimization_score"].idxmax(), "Best hybrid score")
        _add_label(result["Positive_support_score"].idxmax(), "Strongest successful precedent")
        _add_label((result["Green_penalty"] + result["Speed_penalty"]).idxmin(), "Resource-conscious")
        for idx, labels in strategy_labels.items():
            result.loc[idx, "Strategy"] = " & ".join(labels)

    metadata = {
        "optimizer_version": "10.6.1",
        "warnings": [w for w in [solvent_warning] if w],
        "objective": objective,
        "requested_samples": n_samples,
        "feasible_candidates": int(len(candidates)),
        "returned_candidates": int(len(result)),
        "positive_library_rows": int(len(evidence_db)),
        "template_candidates": int(n_template),
        "exploration_candidates": int(n_samples - n_template),
        "fixed_variables": ["Legante", "Ligand_SMILES", "Famiglia_Legante", "Metallo"],
        "optimized_variables": ["Sale_Metallico", "Counterion_Class", "Hydration_Number", "Oxidation_State", "Solvente", "Additivo_Colinker", "Temperatura_C", "Tempo_ore", "mmol_Legante", "mmol_Sale", "Rapporto_LM", "Volume solvente"],
        "unsupported_not_optimized": ["pH", "solvent fractions", "modulator equivalents", "heating ramp", "cooling rate", "stirring", "addition order", "vessel filling fraction", "synthetic method"],
        "model_scope": "Frozen balanced v8.0 outcome predictor plus a quality- and diversity-weighted conditional successful-synthesis recommendation model. Positive support is not interpreted as an absolute success probability.",
    }
    return result, metadata
