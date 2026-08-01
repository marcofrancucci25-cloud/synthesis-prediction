from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd

from .chem import COUNTERIONS, build_row, precursor_formula


OBJECTIVES = {
    "Maximum crystallinity": {
        "p_crystalline": 0.72, "domain": 0.18, "feasibility": 0.10,
        "change": 0.00, "green": 0.00, "speed": 0.00,
    },
    "Balanced conditions": {
        "p_crystalline": 0.55, "domain": 0.20, "feasibility": 0.10,
        "change": 0.08, "green": 0.04, "speed": 0.03,
    },
    "Conservative optimization": {
        "p_crystalline": 0.48, "domain": 0.20, "feasibility": 0.10,
        "change": 0.18, "green": 0.02, "speed": 0.02,
    },
    "Green synthesis": {
        "p_crystalline": 0.43, "domain": 0.17, "feasibility": 0.10,
        "change": 0.04, "green": 0.22, "speed": 0.04,
    },
    "Fast synthesis": {
        "p_crystalline": 0.47, "domain": 0.18, "feasibility": 0.10,
        "change": 0.03, "green": 0.06, "speed": 0.16,
    },
}

# Approximate qualitative solvent penalties. They are interface preferences, not
# learned model features and are kept separate from P(crystalline).
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
        if cur not in vals: vals.append(cur)
    except Exception:
        pass
    return sorted(vals) or [2, 3]


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
    low=max(float(low),1e-8); high=max(float(high),low*1.0001)
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def _contains_solvent(text: str, token: str) -> bool:
    s=_norm(text); t=_norm(token)
    return t in s or s in t


def _green_penalty(solvent: str) -> float:
    s=_norm(solvent)
    components=re.split(r"[/+;,:]",s)
    penalties=[]
    for comp in components:
        comp=comp.strip()
        if not comp: continue
        match=max((v for k,v in SOLVENT_GREEN_PENALTY.items() if k in comp), default=0.30)
        penalties.append(match)
    return float(np.mean(penalties)) if penalties else 0.30


def _change_penalty(base: Dict[str, Any], row: pd.Series, bounds: Dict[str, Tuple[float,float]]) -> float:
    cat_fields=["Counterion_Class","Solvente","Additivo_Colinker"]
    cat=sum(_norm(base.get(f)) != _norm(row.get(f)) for f in cat_fields)/len(cat_fields)
    nums=[]
    for f in ["Hydration_Number","Oxidation_State","Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente","mmol_Sale"]:
        try:
            lo,hi=bounds.get(f,(0.0,1.0)); span=max(hi-lo,1e-8)
            nums.append(min(abs(float(row[f])-float(base.get(f,row[f])))/span,1.0))
        except Exception:
            pass
    return float(0.45*cat+0.55*(np.mean(nums) if nums else 0.0))


def _feasibility(row: pd.Series, constraints: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons=[]; score=1.0
    t=float(row["Temperatura_C"]); h=float(row["Tempo_ore"])
    ratio=float(row["Rapporto_LM"]); vol=float(row["Volume solvente"])
    mmol_m=float(row["mmol_Sale"]); mmol_l=float(row["mmol_Legante"])
    concentration=(mmol_m+mmol_l)/max(vol,1e-8)
    if ratio <= 0 or mmol_m <= 0 or mmol_l <= 0 or vol <= 0:
        return 0.0,["Non-positive amount, ratio or volume"]
    if not (0.02 <= concentration <= 5.0):
        score-=0.45; reasons.append("Total concentration outside the broad observed/plausible range")
    if t > 220:
        score-=0.20; reasons.append("Very high synthesis temperature")
    if h > 168:
        score-=0.15; reasons.append("Very long reaction time")
    if ratio > 10:
        score-=0.20; reasons.append("Extreme ligand/metal ratio")
    if constraints.get("max_temperature") is not None and t > float(constraints["max_temperature"]):
        return 0.0,["Temperature exceeds user constraint"]
    if constraints.get("max_time") is not None and h > float(constraints["max_time"]):
        return 0.0,["Time exceeds user constraint"]
    banned=[_norm(x) for x in constraints.get("banned_solvents",[]) if _norm(x)]
    if any(_contains_solvent(row["Solvente"],b) for b in banned):
        return 0.0,["Solvent excluded by user"]
    allowed=[_norm(x) for x in constraints.get("allowed_solvents",[]) if _norm(x)]
    if allowed and not any(_contains_solvent(row["Solvente"],a) for a in allowed):
        return 0.0,["Solvent not in user allow-list"]
    return max(0.0,float(score)),reasons


def _domain_score(db: pd.DataFrame, base: Dict[str,Any], candidates: pd.DataFrame, bounds: Dict[str,Tuple[float,float]]) -> np.ndarray:
    ligand=_norm(base.get("Legante")); metal=str(base.get("Metallo",""))
    ligand_seen=ligand in set(db["Legante"].astype(str).map(_norm)) if "Legante" in db else False
    metal_seen=metal in set(db["Metallo"].astype(str)) if "Metallo" in db else False
    salts=set(db.get("Sale_Metallico",pd.Series(dtype=str)).astype(str))
    solvents=set(db.get("Solvente",pd.Series(dtype=str)).astype(str))
    additives=set(db.get("Additivo_Colinker",pd.Series(dtype=str)).fillna("Nessuno").astype(str))
    scores=[]
    for _,r in candidates.iterrows():
        s=0.30*float(ligand_seen)+0.20*float(metal_seen)
        s+=0.15*float(str(r["Sale_Metallico"]) in salts)
        s+=0.12*float(str(r["Solvente"]) in solvents)
        s+=0.08*float(str(r["Additivo_Colinker"]) in additives)
        numeric=0.0
        for f in ["Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente","Hydration_Number"]:
            lo,hi=bounds[f]; val=float(r[f])
            if lo <= val <= hi: numeric += 1.0
            else:
                distance=min(abs(val-lo),abs(val-hi))/max(hi-lo,1e-8)
                numeric += max(0.0,1.0-distance)
        s += 0.15*(numeric/5.0)
        scores.append(min(max(s,0.0),1.0))
    return np.asarray(scores,float)


def _nondominated(df: pd.DataFrame, objectives: Sequence[Tuple[str,bool]]) -> pd.Series:
    values=df[[c for c,_ in objectives]].to_numpy(float)
    maximize=np.array([mx for _,mx in objectives],bool)
    adjusted=np.where(maximize,values,-values)
    keep=np.ones(len(df),dtype=bool)
    for i in range(len(df)):
        if not keep[i]: continue
        dominates=np.all(adjusted >= adjusted[i],axis=1)&np.any(adjusted > adjusted[i],axis=1)
        if np.any(dominates): keep[i]=False
    return pd.Series(keep,index=df.index)


def joint_optimize(
    base: Dict[str,Any], model_artifact: Dict[str,Any], features: Sequence[str], db: pd.DataFrame,
    objective: str="Balanced conditions", n_samples: int=2500, top_n: int=12,
    constraints: Optional[Dict[str,Any]]=None, random_state: int=260730,
) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    """Jointly optimize every model-supported variable except ligand and metal.

    This routine searches the learned v8 feature space. Variables absent from the
    frozen model (e.g. pH, ramp rate, cooling mode) are deliberately not optimized.
    """
    constraints=dict(constraints or {})
    rng=np.random.default_rng(random_state)
    n_samples=int(np.clip(n_samples,300,10000))
    objective=objective if objective in OBJECTIVES else "Balanced conditions"
    weights=OBJECTIVES[objective]

    bounds={
        "Temperatura_C": _quantile_bounds(db,"Temperatura_C",(40,220)),
        "Tempo_ore": _quantile_bounds(db,"Tempo_ore",(0.5,168)),
        "Rapporto_LM": _quantile_bounds(db,"Rapporto_LM",(0.1,10)),
        "Volume solvente": _quantile_bounds(db,"Volume solvente",(0.5,100)),
        "mmol_Sale": _quantile_bounds(db,"mmol_Sale",(0.005,10)),
        "Hydration_Number": _quantile_bounds(db,"Hydration_Number",(0,12)),
        "Oxidation_State": _quantile_bounds(db,"Oxidation_State",(1,6)),
    }
    # User constraints narrow, never expand, the evidence-derived ranges.
    temp_lo=max(bounds["Temperatura_C"][0],float(constraints.get("min_temperature",bounds["Temperatura_C"][0])))
    temp_hi=min(bounds["Temperatura_C"][1],float(constraints.get("max_temperature",bounds["Temperatura_C"][1])))
    time_lo=max(bounds["Tempo_ore"][0],float(constraints.get("min_time",bounds["Tempo_ore"][0])))
    time_hi=min(bounds["Tempo_ore"][1],float(constraints.get("max_time",bounds["Tempo_ore"][1])))
    ratio_lo=max(bounds["Rapporto_LM"][0],float(constraints.get("min_ratio",bounds["Rapporto_LM"][0])))
    ratio_hi=min(bounds["Rapporto_LM"][1],float(constraints.get("max_ratio",bounds["Rapporto_LM"][1])))
    if temp_lo>=temp_hi or time_lo>=time_hi or ratio_lo>=ratio_hi:
        raise ValueError("The selected constraints leave an empty numerical search range.")

    solvents=_pool(db,"Solvente",18,[base.get("Solvente","DMF")])
    additives=_pool(db,"Additivo_Colinker",14,[base.get("Additivo_Colinker","Nessuno")])
    if constraints.get("keep_solvent"):
        solvents=[base.get("Solvente","DMF")]
    if constraints.get("keep_additive"):
        additives=[base.get("Additivo_Colinker","Nessuno")]
    allowed=constraints.get("allowed_solvents") or []
    if allowed:
        solvents=[s for s in solvents if any(_contains_solvent(s,a) for a in allowed)] or list(allowed)
    banned=constraints.get("banned_solvents") or []
    solvents=[s for s in solvents if not any(_contains_solvent(s,b) for b in banned)]
    if not solvents:
        raise ValueError("No solvent remains after applying the selected constraints.")

    counterions=[base.get("Counterion_Class","nitrate")] if constraints.get("keep_precursor") else _counterion_pool(db,base.get("Metallo"),8)
    oxidations=[int(float(base.get("Oxidation_State",2) or 2))] if constraints.get("keep_precursor") else _oxidation_pool(db,base.get("Metallo"),base.get("Oxidation_State"))

    # Stratified random search across mixed categorical/continuous dimensions.
    rows=[]
    for i in range(n_samples):
        oxidation=int(rng.choice(oxidations))
        counterion=str(rng.choice(counterions))
        hydration=float(base.get("Hydration_Number",0) or 0) if constraints.get("keep_precursor") else float(np.round(rng.uniform(max(0,bounds["Hydration_Number"][0]),min(12,bounds["Hydration_Number"][1]))*2)/2)
        ratio=float(_sample_log_uniform(rng,ratio_lo,ratio_hi,1)[0])
        mmol_m=float(_sample_log_uniform(rng,max(0.001,bounds["mmol_Sale"][0]),bounds["mmol_Sale"][1],1)[0])
        mmol_l=ratio*mmol_m
        volume=float(_sample_log_uniform(rng,max(0.2,bounds["Volume solvente"][0]),bounds["Volume solvente"][1],1)[0])
        solvent=str(rng.choice(solvents)); additive=str(rng.choice(additives))
        temperature=float(rng.uniform(temp_lo,temp_hi))
        time=float(_sample_log_uniform(rng,time_lo,time_hi,1)[0])
        salt=precursor_formula(base.get("Metallo"),oxidation,counterion,hydration)
        row=dict(base)
        row.update({
            "Oxidation_State":oxidation,"Counterion_Class":counterion,
            "Hydration_Number":hydration,"Sale_Metallico":salt,
            "Solvente":solvent,"Additivo_Colinker":additive,
            "Temperatura_C":temperature,"Tempo_ore":time,
            "mmol_Sale":mmol_m,"mmol_Legante":mmol_l,
            "Rapporto_LM":ratio,"Volume solvente":volume,
        })
        rows.append(row)

    candidates=pd.DataFrame(rows)
    feasibility=[]; reasons=[]
    for _,r in candidates.iterrows():
        s,rs=_feasibility(r,constraints); feasibility.append(s); reasons.append("; ".join(rs))
    candidates["Feasibility_score"]=feasibility; candidates["Feasibility_notes"]=reasons
    candidates=candidates[candidates["Feasibility_score"]>0].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No feasible candidate was generated under the selected constraints.")

    engineered=pd.concat([build_row(r.to_dict()) for _,r in candidates.iterrows()],ignore_index=True)
    for c in features:
        if c not in engineered: engineered[c]=np.nan
    x=engineered[list(features)]
    probs=model_artifact["weights"][0]*model_artifact["rf_model"].predict_proba(x)+model_artifact["weights"][1]*model_artifact["ligand_text_model"].predict_proba(x)
    candidates["P_Failed"]=probs[:,0]; candidates["P_Amorphous"]=probs[:,1]; candidates["P_Crystalline"]=probs[:,2]
    candidates["AD_score"]=_domain_score(db,base,candidates,bounds)
    candidates["Green_penalty"]=candidates["Solvente"].map(_green_penalty)
    candidates["Speed_penalty"]=(np.clip(candidates["Temperatura_C"]/220,0,1)+np.clip(np.log1p(candidates["Tempo_ore"])/np.log1p(168),0,1))/2
    candidates["Change_penalty"]=[_change_penalty(base,r,bounds) for _,r in candidates.iterrows()]
    candidates["Optimization_score"]=(
        weights["p_crystalline"]*candidates["P_Crystalline"]+
        weights["domain"]*candidates["AD_score"]+
        weights["feasibility"]*candidates["Feasibility_score"]-
        weights["change"]*candidates["Change_penalty"]-
        weights["green"]*candidates["Green_penalty"]-
        weights["speed"]*candidates["Speed_penalty"]
    )

    # Pareto analysis preserves alternatives rather than returning one artificial optimum.
    candidates["Pareto_optimal"]=_nondominated(candidates,[
        ("P_Crystalline",True),("AD_score",True),("Feasibility_score",True),
        ("Green_penalty",False),("Speed_penalty",False),("Change_penalty",False),
    ])
    candidates["Total_concentration_mmol_mL"]=(candidates["mmol_Legante"]+candidates["mmol_Sale"])/candidates["Volume solvente"]
    candidates["Objective"]=objective

    sort_cols=["Pareto_optimal","Optimization_score","P_Crystalline","AD_score"]
    result=(candidates.sort_values(sort_cols,ascending=[False,False,False,False])
            .drop_duplicates(["Sale_Metallico","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM"])
            .head(int(top_n)).reset_index(drop=True))
    result.insert(0,"Rank",np.arange(1,len(result)+1))
    result["Strategy"]="Alternative"
    if len(result):
        result.loc[result["P_Crystalline"].idxmax(),"Strategy"]="Maximum probability"
        result.loc[result["Optimization_score"].idxmax(),"Strategy"]="Best objective score"
        result.loc[(result["Green_penalty"]+result["Speed_penalty"]).idxmin(),"Strategy"]="Resource-conscious"

    metadata={
        "objective":objective,"requested_samples":n_samples,
        "feasible_candidates":int(len(candidates)),"returned_candidates":int(len(result)),
        "fixed_variables":["Legante","Ligand_SMILES","Famiglia_Legante","Metallo"],
        "optimized_variables":["Sale_Metallico","Counterion_Class","Hydration_Number","Oxidation_State","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","mmol_Legante","mmol_Sale","Rapporto_LM","Volume solvente"],
        "unsupported_not_optimized":["pH","solvent fractions","modulator equivalents","heating ramp","cooling rate","stirring","addition order","vessel filling fraction","synthetic method"],
        "model_scope":"Frozen v8.0 predictive core; joint search is limited to features learned by that model.",
    }
    return result,metadata
