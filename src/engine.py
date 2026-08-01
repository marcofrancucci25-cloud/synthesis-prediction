from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from .chem import build_row
from .optimizer import joint_optimize
ROOT=Path(__file__).resolve().parents[1]
ART=joblib.load(ROOT/'models/MOF_ChemAware_Ensemble_v8_0.joblib')
SCHEMA=json.loads((ROOT/'models/feature_schema_v8_0.json').read_text())
DB=pd.read_csv(ROOT/'data/knowledge_database.csv')
POSITIVE_DB=pd.read_csv(ROOT/'data/successful_synthesis_library_v10_4.csv')
POSITIVE_MODEL_PATH=ROOT/'models/Positive_Condition_Recommendation_v10_4.joblib'
POSITIVE_MODEL=joblib.load(POSITIVE_MODEL_PATH) if POSITIVE_MODEL_PATH.exists() else None
FEATURES=ART['features']

def predict(values):
    x=build_row(values)
    for c in FEATURES:
        if c not in x: x[c]=np.nan
    x=x[FEATURES]
    p=ART['weights'][0]*ART['rf_model'].predict_proba(x)+ART['weights'][1]*ART['ligand_text_model'].predict_proba(x)
    return x, p[0], int(np.argmax(p[0]))


NUMERIC_VALIDITY_COLUMNS = [
    "Temperatura_C", "Tempo_ore", "Rapporto_LM",
    "mmol_Legante", "mmol_Sale", "Volume solvente",
]

def _training_ranges():
    ranges = {}
    for c in NUMERIC_VALIDITY_COLUMNS:
        x = pd.to_numeric(DB[c], errors="coerce").dropna()
        if len(x):
            ranges[c] = {
                "min": float(x.min()), "max": float(x.max()),
                "q01": float(x.quantile(0.01)), "q99": float(x.quantile(0.99)),
                "q05": float(x.quantile(0.05)), "q95": float(x.quantile(0.95)),
            }
    return ranges

TRAINING_RANGES = _training_ranges()

def prediction_validity(values):
    """Assess whether an input lies within the experimentally supported range.

    This gate does not alter model probabilities. It reports when those
    probabilities should not be interpreted as validated estimates.
    """
    issues, details = [], []
    penalties = []
    labels = {
        "Temperatura_C":"Temperature", "Tempo_ore":"Reaction time",
        "Rapporto_LM":"Ligand/metal ratio", "mmol_Legante":"Ligand amount",
        "mmol_Sale":"Metal precursor amount", "Volume solvente":"Solvent volume",
    }
    for c in NUMERIC_VALIDITY_COLUMNS:
        try:
            v = float(values.get(c, np.nan))
        except Exception:
            v = np.nan
        r = TRAINING_RANGES.get(c)
        if not r or not np.isfinite(v):
            issues.append(f"{labels[c]} is missing or non-numeric.")
            penalties.append(1.0)
            continue
        if v < r["min"] or v > r["max"]:
            issues.append(f"{labels[c]} ({v:g}) is outside the observed training range {r['min']:g}–{r['max']:g}.")
            severity = 1.0
        elif v < r["q01"] or v > r["q99"]:
            issues.append(f"{labels[c]} ({v:g}) is in an extreme tail of the training data (central 98%: {r['q01']:g}–{r['q99']:g}).")
            severity = 0.65
        elif v < r["q05"] or v > r["q95"]:
            severity = 0.25
        else:
            severity = 0.0
        penalties.append(severity)
        details.append({"field":c,"value":v,**r,"severity":severity})

    # Internal consistency of user-entered stoichiometry.
    try:
        mmol_l=float(values.get("mmol_Legante")); mmol_m=float(values.get("mmol_Sale")); ratio=float(values.get("Rapporto_LM"))
        calculated=mmol_l/mmol_m if mmol_m>0 else np.nan
        rel=abs(calculated-ratio)/max(abs(calculated),0.05)
        if np.isfinite(rel) and rel>0.20:
            issues.append(f"Entered L:M ratio ({ratio:g}) is inconsistent with the reagent amounts (calculated {calculated:.3g}).")
            penalties.append(min(1.0,0.5+rel/2))
    except Exception:
        pass

    # Concentration plausibility where volume is available (mmol/mL numerically equals mol/L).
    try:
        vol=float(values.get("Volume solvente")); total=float(values.get("mmol_Legante"))+float(values.get("mmol_Sale"))
        conc=total/vol if vol>0 else np.inf
        if not np.isfinite(conc) or conc>1.35 or conc<0.003:
            issues.append(f"Total precursor concentration ({conc:.3g} mol/L) is outside the central 98% of recorded syntheses (~0.003–1.32 mol/L).")
            penalties.append(0.8)
    except Exception:
        pass

    worst=max(penalties) if penalties else 0.0
    mean=float(np.mean(penalties)) if penalties else 0.0
    score=float(np.clip(1.0-(0.65*worst+0.35*mean),0,1))
    if worst>=0.95 or score<0.45:
        label="Outside validated experimental range"; reliable=False
    elif worst>=0.60 or score<0.72:
        label="Extrapolative / use with caution"; reliable=False
    else:
        label="Within validated experimental range"; reliable=True
    return {"score":score,"label":label,"reliable":reliable,"issues":issues,"details":details}

def applicability(values):
    ligand=str(values.get('Legante','')).strip().lower(); metal=str(values.get('Metallo','')); salt=str(values.get('Sale_Metallico',''))
    seen_lig=ligand in set(DB['Legante'].astype(str).str.lower())
    seen_metal=metal in set(DB['Metallo'].astype(str))
    seen_salt=salt in set(DB['Sale_Metallico'].astype(str))
    identity_score=0.50*seen_lig+0.30*seen_metal+0.20*seen_salt
    validity=prediction_validity(values)
    # Identity support and numerical support are both required. A high categorical
    # match cannot mask an extreme experimental condition.
    score=float(0.60*identity_score+0.40*validity['score'])
    if score>=0.78 and validity['reliable']: label='Inside domain'
    elif score>=0.38 and validity['label']!='Outside validated experimental range': label='Intermediate / partial extrapolation'
    else: label='Outside domain'
    return {'score':score,'label':label,'ligand_seen':seen_lig,'metal_seen':seen_metal,'salt_seen':seen_salt,'identity_score':float(identity_score),'validity':validity}

def similar(values,n=15):
    d=DB.copy(); metal=str(values.get('Metallo','')); fam=str(values.get('Famiglia_Legante',''))
    d['_score']=0
    d.loc[d['Metallo'].astype(str)==metal,'_score']+=3
    d.loc[d['Famiglia_Legante'].astype(str)==fam,'_score']+=2
    for c,w in [('Temperatura_C',1),('Tempo_ore',1),('Rapporto_LM',1)]:
        try:
            scale=max(float(d[c].std()),1); val=float(values.get(c,np.nan)); d['_score']+=np.exp(-abs(pd.to_numeric(d[c],errors='coerce')-val)/scale)*w
        except: pass
    return d.sort_values('_score',ascending=False).head(n)

def explain_prediction(values):
    """Local, model-based sensitivity summary for the current synthesis.

    Each condition is varied independently across plausible values observed in the
    experimental database. The output is descriptive rather than causal: it shows
    which editable parameters can most change P(crystalline) near the current input.
    """
    _, base_p, _ = predict(values)
    base_cryst = float(base_p[2])
    specs = {
        'Solvente': list(DB['Solvente'].dropna().astype(str).value_counts().head(10).index),
        'Temperatura_C': sorted(set([max(20.0, float(values.get('Temperatura_C',120))+d) for d in (-40,-20,20,40)] + [80.0,100.0,120.0,150.0])),
        'Tempo_ore': sorted(set([max(0.5, float(values.get('Tempo_ore',24))*m) for m in (0.5,2.0)] + [6.0,12.0,24.0,48.0,72.0])),
        'Rapporto_LM': sorted(set([max(0.1, float(values.get('Rapporto_LM',1))+d) for d in (-1.0,-0.5,0.5,1.0)] + [0.5,1.0,2.0,3.0])),
        'Additivo_Colinker': list(DB['Additivo_Colinker'].fillna('Nessuno').astype(str).value_counts().head(8).index),
        'Volume solvente': sorted(set([max(0.5, float(values.get('Volume solvente',10))*m) for m in (0.5,1.5,2.0)])),
    }
    labels = {
        'Solvente':'Solvent', 'Temperatura_C':'Temperature', 'Tempo_ore':'Reaction time',
        'Rapporto_LM':'Ligand/metal ratio', 'Additivo_Colinker':'Additive / co-linker',
        'Volume solvente':'Solvent volume'
    }
    rows=[]
    for field, candidates in specs.items():
        tested=[]
        current=values.get(field)
        for candidate in candidates:
            if str(candidate)==str(current):
                continue
            v=dict(values); v[field]=candidate
            try:
                _, pp, _=predict(v); tested.append((candidate,float(pp[2])))
            except Exception:
                continue
        if not tested:
            continue
        best_val,best_p=max(tested,key=lambda z:z[1])
        mean_alt=float(np.mean([z[1] for z in tested]))
        improvement=best_p-base_cryst
        support=base_cryst-mean_alt
        direction='Limiting' if improvement>0.025 else ('Favorable' if support>0.025 else 'Neutral')
        influence=max(abs(improvement),abs(support))
        rows.append({
            'Parameter':labels[field], 'Field':field, 'Current':current,
            'Influence':influence, 'Direction':direction,
            'Potential_improvement':max(0.0,improvement),
            'Best_alternative':best_val, 'Best_P_crystalline':best_p,
        })
    return pd.DataFrame(rows).sort_values('Influence',ascending=False).reset_index(drop=True), base_cryst



def optimize_joint(values, objective="Balanced conditions", n_samples=2500, top_n=12, constraints=None):
    return joint_optimize(
        values, model_artifact=ART, features=FEATURES, db=DB, positive_db=POSITIVE_DB,
        positive_model=POSITIVE_MODEL, objective=objective, n_samples=n_samples, top_n=top_n,
        constraints=constraints or {},
    )

# Compatibility wrapper retained for older callers.
def optimize(values, top_n=10):
    results, _ = optimize_joint(values, objective="Balanced conditions", n_samples=1500, top_n=top_n)
    return results
