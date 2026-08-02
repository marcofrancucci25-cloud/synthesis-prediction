from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from .chem import build_row, canonicalize_family, canonicalize_ligand_for_model, parse_salt
from .optimizer import joint_optimize
from .mof_registry import known_mof_matches
ROOT=Path(__file__).resolve().parents[1]
ART=joblib.load(ROOT/'models/MOF_ChemAware_Ensemble_v8_0.joblib')
SCHEMA=json.loads((ROOT/'models/feature_schema_v8_0.json').read_text())
TRAINING_DB=pd.read_csv(ROOT/'data/knowledge_database.csv')
EVIDENCE_DB_PATH=ROOT/'data/knowledge_database_integrated_v10_6.csv'
EVIDENCE_DB=pd.read_csv(EVIDENCE_DB_PATH) if EVIDENCE_DB_PATH.exists() else TRAINING_DB.copy()
# Backwards-compatible public name: predictor/domain calculations remain tied to
# the frozen v8 training data. Similar-record retrieval uses EVIDENCE_DB below.
DB=TRAINING_DB
POSITIVE_DB=pd.read_csv(ROOT/'data/successful_synthesis_library_v10_6.csv')
POSITIVE_MODEL_PATH=ROOT/'models/Positive_Condition_Recommendation_v10_4.joblib'
POSITIVE_MODEL=joblib.load(POSITIVE_MODEL_PATH) if POSITIVE_MODEL_PATH.exists() else None
FEATURES=ART['features']

def _verified_evidence_database():
    records=[]
    lab=EVIDENCE_DB[EVIDENCE_DB.get('Source_Type',pd.Series(index=EVIDENCE_DB.index,dtype=object)).eq('laboratory_experiment')].copy()
    if len(lab):
        lab['Evidence_ID']=lab['ID'].astype(str)
        lab['Evidence_Source']='Laboratory experiment'
        lab['Source_DOI']=np.nan
        lab['Evidence_Statement']=lab.get('Evidence',np.nan)
        records.append(lab)
    literature_path=ROOT/'data/literature_crystalline_challenge_v10_6.csv'
    if literature_path.exists():
        lit=pd.read_csv(literature_path)
        lit['ID']=lit['Case_ID']
        lit['Esito_ML']=2
        lit['Evidence_ID']=lit['Case_ID']
        lit['Evidence_Source']='Peer-reviewed literature'
        lit['Evidence_Statement']=lit['Crystallinity_Evidence']
        lit['PXRD_Confirmed']=True
        records.append(lit)
    if not records:
        return pd.DataFrame()
    evidence=pd.concat(records,ignore_index=True,sort=False)
    evidence['Canonical_Ligand']=evidence['Legante'].map(canonicalize_ligand_for_model)
    evidence['Canonical_Family']=[
        canonicalize_family(f,l) for f,l in zip(evidence['Famiglia_Legante'],evidence['Canonical_Ligand'])
    ]
    return evidence

VERIFIED_EVIDENCE=_verified_evidence_database()

def _text_key(value):
    return ' '.join(str(value or '').strip().casefold().replace(';','/').split())

def _relative_difference(left,right,floor=0.1):
    try:
        a=float(left); b=float(right)
        if not np.isfinite(a) or not np.isfinite(b): return np.inf
        return abs(a-b)/max(abs(b),floor)
    except Exception:
        return np.inf

def verified_precedents(values,n=5):
    """Return independently verified laboratory/literature precedents.

    Evidence is deliberately kept separate from classifier probabilities.  An
    exact or close protocol can therefore correct the interpretation of an
    uncertain model result without pretending to be a calibrated probability.
    """
    if VERIFIED_EVIDENCE.empty:
        return VERIFIED_EVIDENCE.copy()
    query=build_row(values).iloc[0]
    ligand=canonicalize_ligand_for_model(query.get('Legante',''))
    metal=str(query.get('Metallo','')).strip()
    candidates=VERIFIED_EVIDENCE[
        VERIFIED_EVIDENCE['Canonical_Ligand'].astype(str).str.casefold().eq(str(ligand).casefold())
        & VERIFIED_EVIDENCE['Metallo'].astype(str).eq(metal)
    ].copy()
    if candidates.empty:
        return candidates
    query_counter=parse_salt(query.get('Sale_Metallico','')).get('Counterion_Class')
    rows=[]
    for _,r in candidates.iterrows():
        salt_exact=_text_key(r.get('Sale_Metallico'))==_text_key(query.get('Sale_Metallico'))
        counter=str(r.get('Counterion_Class') or parse_salt(r.get('Sale_Metallico','')).get('Counterion_Class'))
        counter_match=_text_key(counter)==_text_key(query_counter)
        solvent_match=_text_key(r.get('Solvente'))==_text_key(query.get('Solvente'))
        additive_match=_text_key(r.get('Additivo_Colinker'))==_text_key(query.get('Additivo_Colinker'))
        temp_delta=abs(float(r.get('Temperatura_C'))-float(query.get('Temperatura_C')))
        time_delta=_relative_difference(r.get('Tempo_ore'),query.get('Tempo_ore'),floor=1.0)
        ratio_delta=_relative_difference(r.get('Rapporto_LM'),query.get('Rapporto_LM'),floor=0.25)
        volume_delta=_relative_difference(r.get('Volume solvente'),query.get('Volume solvente'),floor=1.0)
        exact=(salt_exact and solvent_match and additive_match and temp_delta<=1
               and time_delta<=0.05 and ratio_delta<=0.05 and volume_delta<=0.05)
        close=(counter_match and solvent_match and temp_delta<=20
               and time_delta<=0.50 and ratio_delta<=0.35)
        if exact: level='Exact verified protocol'; rank=0
        elif close: level='Close verified protocol'; rank=1
        else: level='Same ligand–metal system'; rank=2
        distance=(temp_delta/50.0 + min(time_delta,2) + min(ratio_delta,2)
                  + min(volume_delta,2) + (0 if solvent_match else 1)
                  + (0 if counter_match else 0.5) + (0 if additive_match else 0.5))
        rows.append({
            'Evidence_ID':r.get('Evidence_ID'),'Match_Level':level,'_rank':rank,
            '_distance':distance,'Outcome_Class':int(r.get('Esito_ML')),
            'Verified_Outcome':{0:'Failed',1:'Amorphous/uncertain',2:'Crystalline MOF'}.get(int(r.get('Esito_ML'))),
            'Evidence_Source':r.get('Evidence_Source'),'Source_DOI':r.get('Source_DOI'),
            'Evidence_Statement':r.get('Evidence_Statement'),'Legante':r.get('Legante'),
            'Metallo':r.get('Metallo'),'Sale_Metallico':r.get('Sale_Metallico'),
            'Solvente':r.get('Solvente'),'Additivo_Colinker':r.get('Additivo_Colinker'),
            'Temperatura_C':r.get('Temperatura_C'),'Tempo_ore':r.get('Tempo_ore'),
            'Rapporto_LM':r.get('Rapporto_LM'),'Volume solvente':r.get('Volume solvente'),
            'Encoding_Note':r.get('Encoding_Note'),
            'MOF':r.get('MOF'),
        })
    return pd.DataFrame(rows).sort_values(['_rank','_distance']).head(n).drop(columns=['_rank','_distance'])

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
        x = pd.to_numeric(TRAINING_DB[c], errors="coerce").dropna()
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
    ligand=canonicalize_ligand_for_model(values.get('Legante','')).casefold(); metal=str(values.get('Metallo','')); salt=str(values.get('Sale_Metallico',''))
    training_ligands=set(TRAINING_DB['Legante'].map(canonicalize_ligand_for_model).astype(str).str.casefold())
    seen_lig=ligand in training_ligands
    seen_metal=metal in set(TRAINING_DB['Metallo'].astype(str))
    seen_salt=salt in set(TRAINING_DB['Sale_Metallico'].astype(str))
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
    d=EVIDENCE_DB.copy(); metal=str(values.get('Metallo','')); fam=canonicalize_family(values.get('Famiglia_Legante',''),values.get('Legante',''))
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
    canonical_values=build_row(values).iloc[0].to_dict()
    result,metadata=joint_optimize(
        canonical_values, model_artifact=ART, features=FEATURES, db=TRAINING_DB, positive_db=POSITIVE_DB,
        positive_model=POSITIVE_MODEL, objective=objective, n_samples=n_samples, top_n=top_n,
        constraints=constraints or {},
    )
    # Canonical identities are model-internal. Preserve the scientist's original
    # ligand label in the experimental plan.
    if 'Legante' in result:
        result['Legante']=values.get('Legante',canonical_values.get('Legante'))
    return result,metadata

# Compatibility wrapper retained for older callers.
def optimize(values, top_n=10):
    results, _ = optimize_joint(values, objective="Balanced conditions", n_samples=1500, top_n=top_n)
    return results
