from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from .chem import build_row, canonicalize_family, canonicalize_ligand_for_model, infer_family, parse_salt, FAMILIES
from .solubility import describe as _solubility_describe
from .vessel_conditions import vessel_requirement as _vessel_requirement
from .modulator_chemistry import modulator_compatibility as _modulator_compatibility
from .solvent_miscibility import miscibility_check as _miscibility_check
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

NUMERIC_MODEL_FEATURES = [c for c in SCHEMA.get('numeric', []) if c in FEATURES]

def _coerce_numeric_features(x):
    """Defensively coerce declared-numeric columns before they reach the model.

    A stray non-numeric value (for example the literal string "unknown" for
    an oxidation state, reached through any caller that does not pre-convert
    it to None the way the Streamlit form does) must degrade to a missing
    value handled by the model's own imputer, not raise an uncaught
    exception that would crash the page.
    """
    present = [c for c in NUMERIC_MODEL_FEATURES if c in x]
    if present:
        x = x.copy()
        x.loc[:, present] = x.loc[:, present].apply(pd.to_numeric, errors='coerce')
    return x

def predict(values):
    x=build_row(values)
    for c in FEATURES:
        if c not in x: x[c]=np.nan
    x=x[FEATURES]
    x=_coerce_numeric_features(x)
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

def _family_consistency(values):
    """Flag a declared ligand family that disagrees with the ligand's own name.

    ``Famiglia_Legante`` is a user-editable selector in the UI: nothing
    upstream guarantees it still matches the ligand once the field is
    changed. Since the family is a model-facing feature, an unnoticed
    mismatch can shift the predicted outcome substantially without any
    signal to the user. The check only fires when the declared value is one
    of the public UI family labels and the name-based heuristic reached a
    confident (non-"Other/unknown") conclusion, to avoid false positives on
    values coming from other, non-UI callers.
    """
    declared=str(values.get('Famiglia_Legante') or '').strip()
    inferred=infer_family(values.get('Legante',''))
    if declared not in FAMILIES or inferred=='Other/unknown':
        return False, declared, inferred
    return declared!=inferred, declared, inferred

def applicability(values):
    ligand=canonicalize_ligand_for_model(values.get('Legante','')).casefold(); metal=str(values.get('Metallo','')); salt=str(values.get('Sale_Metallico',''))
    training_ligands=set(TRAINING_DB['Legante'].map(canonicalize_ligand_for_model).astype(str).str.casefold())
    seen_lig=ligand in training_ligands
    seen_metal=metal in set(TRAINING_DB['Metallo'].astype(str))
    seen_salt=salt in set(TRAINING_DB['Sale_Metallico'].astype(str))
    identity_score=0.50*seen_lig+0.30*seen_metal+0.20*seen_salt
    validity=prediction_validity(values)
    family_mismatch,declared_family,inferred_family=_family_consistency(values)
    # Identity support and numerical support are both required. A high categorical
    # match cannot mask an extreme experimental condition. A declared family that
    # disagrees with the ligand's own name is treated the same way: it caps how
    # confidently the input can be called "inside domain".
    score=float(0.60*identity_score+0.40*validity['score'])
    if family_mismatch:
        score=float(score*0.85)
    if score>=0.78 and validity['reliable'] and not family_mismatch: label='Inside domain'
    elif score>=0.38 and validity['label']!='Outside validated experimental range': label='Intermediate / partial extrapolation'
    else: label='Outside domain'
    # Solubility is a physical-chemistry check, deliberately kept separate from
    # the applicability-domain score above: AD measures how far the *model*
    # is extrapolating, not whether the proposed chemistry is physically
    # sound. See src/solubility.py for what this estimate can and cannot
    # detect (it is a coarse screen, not a substitute for chemical judgment).
    solubility=_solubility_describe(values.get('Ligand_SMILES'), values.get('Solvente',''))
    # Same kind of self-consistency check as family_mismatch above, applied to
    # a different pair of user-facing fields: an explicitly declared
    # "Room Temperature" / "Precipitation" procedure is inconsistent with a
    # temperature that, per src/vessel_conditions.py, is at or above the
    # solvent's boiling point and would actually require a sealed vessel.
    # Purely informational -- it does not change the AD score, since a
    # sealed-vessel requirement is not evidence the model is extrapolating.
    vessel=_vessel_requirement(values.get('Solvente',''), values.get('Temperatura_C'))
    declared_procedure=str(values.get('Procedura_Sintetica') or '').strip()
    vessel_mismatch=bool(vessel.get('requires_sealed_vessel')) and declared_procedure in ('Room Temperature','Precipitation')
    modulator=_modulator_compatibility(values.get('Famiglia_Legante'), values.get('Additivo_Colinker',''), values.get('Legante',''))
    miscibility=_miscibility_check(values.get('Solvente',''))
    return {'score':score,'label':label,'ligand_seen':seen_lig,'metal_seen':seen_metal,'salt_seen':seen_salt,
            'identity_score':float(identity_score),'validity':validity,'family_mismatch':family_mismatch,
            'declared_family':declared_family,'inferred_family':inferred_family,'solubility':solubility,
            'vessel':vessel,'vessel_mismatch':vessel_mismatch,'modulator':modulator,'miscibility':miscibility}

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

def _supported_ratio_candidates(values, maximum=10):
    """Return central, actually observed L/M ratios for comparable chemistry.

    Sensitivity analysis must not manufacture an extreme ratio by subtracting a
    fixed number from the current value.  Prefer the exact ligand/metal system,
    then the same metal and canonical ligand family, then the same metal.  The
    global training set is used only as a final fallback.  In every case the
    candidates are restricted to the central 90% of the frozen training data.
    """
    ratios=pd.to_numeric(DB.get('Rapporto_LM',pd.Series(dtype=float)),errors='coerce')
    finite=np.isfinite(ratios) & ratios.gt(0)
    limits=TRAINING_RANGES.get('Rapporto_LM',{})
    lower=float(limits.get('q05',ratios[finite].quantile(0.05)))
    upper=float(limits.get('q95',ratios[finite].quantile(0.95)))
    central=finite & ratios.between(lower,upper,inclusive='both')
    metal=str(values.get('Metallo','')).strip()
    ligand=canonicalize_ligand_for_model(values.get('Legante','')).casefold()
    family=canonicalize_family(values.get('Famiglia_Legante',''),values.get('Legante',''))
    db_ligand=DB.get('Legante',pd.Series('',index=DB.index)).map(canonicalize_ligand_for_model).astype(str).str.casefold()
    db_family=pd.Series([
        canonicalize_family(f,l) for f,l in zip(
            DB.get('Famiglia_Legante',pd.Series('',index=DB.index)),
            DB.get('Legante',pd.Series('',index=DB.index)),
        )
    ],index=DB.index)
    same_metal=DB.get('Metallo',pd.Series('',index=DB.index)).astype(str).eq(metal)
    pools=[
        (central & same_metal & db_ligand.eq(ligand),'same ligand–metal system',5),
        (central & same_metal & db_family.eq(family),'same metal and ligand family',8),
        (central & same_metal,'same metal',10),
        (central,'central training data',1),
    ]
    selected_scope='central training data'
    selected=ratios[central]
    for mask,scope,minimum in pools:
        candidate=ratios[mask]
        if candidate.notna().sum()>=minimum:
            selected=candidate
            selected_scope=scope
            break
    rounded=selected.round(4)
    counts=rounded.value_counts()
    top=counts.head(int(maximum))
    candidates=sorted(float(x) for x in top.index)
    support={float(x):int(top.loc[x]) for x in top.index}
    return candidates,support,selected_scope,(lower,upper)

def _ratio_perturbation(values,ratio):
    """Change L/M coherently while preserving total precursor concentration."""
    ratio=float(ratio)
    ligand=float(values.get('mmol_Legante'))
    metal=float(values.get('mmol_Sale'))
    total=ligand+metal
    if not np.isfinite(ratio) or ratio<=0 or not np.isfinite(total) or total<=0:
        raise ValueError('Ratio sensitivity requires positive numeric precursor amounts.')
    new_metal=total/(1.0+ratio)
    new_ligand=total-new_metal
    candidate=dict(values)
    candidate['Rapporto_LM']=ratio
    candidate['mmol_Legante']=new_ligand
    candidate['mmol_Sale']=new_metal
    return candidate,new_ligand,new_metal

def explain_prediction(values):
    """Local, model-based sensitivity summary for the current synthesis.

    Each condition is varied independently across plausible values observed in the
    experimental database. The output is descriptive rather than causal: it shows
    which editable parameters can most change P(crystalline) near the current input.
    """
    _, base_p, _ = predict(values)
    base_cryst = float(base_p[2])
    ratio_candidates,ratio_support,ratio_scope,ratio_limits=_supported_ratio_candidates(values)
    specs = {
        'Solvente': list(DB['Solvente'].dropna().astype(str).value_counts().head(10).index),
        'Temperatura_C': sorted(set([max(20.0, float(values.get('Temperatura_C',120))+d) for d in (-40,-20,20,40)] + [80.0,100.0,120.0,150.0])),
        'Tempo_ore': sorted(set([max(0.5, float(values.get('Tempo_ore',24))*m) for m in (0.5,2.0)] + [6.0,12.0,24.0,48.0,72.0])),
        'Rapporto_LM': ratio_candidates,
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
            try:
                if field=='Rapporto_LM' and np.isclose(float(candidate),float(current),rtol=0,atol=1e-9):
                    continue
            except (TypeError,ValueError):
                pass
            if field!='Rapporto_LM' and str(candidate)==str(current):
                continue
            try:
                detail='Numerically within the validated training range.'
                support_count=np.nan
                support_scope=''
                if field=='Rapporto_LM':
                    v,new_ligand,new_metal=_ratio_perturbation(values,candidate)
                    support_count=int(ratio_support.get(float(candidate),0))
                    support_scope=ratio_scope
                    detail=(
                        f"Amounts rebalanced at constant total precursor amount: "
                        f"ligand {new_ligand:.4g} mmol, metal {new_metal:.4g} mmol; "
                        f"{support_count} observed record(s) in {support_scope}."
                    )
                else:
                    v=dict(values); v[field]=candidate
                validity=prediction_validity(v)
                if not validity.get('reliable',False):
                    continue
                _, pp, _=predict(v)
                tested.append({
                    'value':candidate,'probability':float(pp[2]),'detail':detail,
                    'validity_score':float(validity['score']),
                    'support_count':support_count,'support_scope':support_scope,
                })
            except Exception:
                continue
        if not tested:
            continue
        best=max(tested,key=lambda z:z['probability'])
        best_val,best_p=best['value'],best['probability']
        mean_alt=float(np.mean([z['probability'] for z in tested]))
        improvement=best_p-base_cryst
        support=base_cryst-mean_alt
        direction='Limiting' if improvement>0.025 else ('Favorable' if support>0.025 else 'Neutral')
        influence=max(abs(improvement),abs(support))
        rows.append({
            'Parameter':labels[field], 'Field':field, 'Current':current,
            'Influence':influence, 'Direction':direction,
            'Potential_improvement':max(0.0,improvement),
            'Best_alternative':best_val, 'Best_P_crystalline':best_p,
            'Best_Alternative_Detail':best['detail'],
            'Alternative_Validity_Score':best['validity_score'],
            'Alternative_Support_Count':best['support_count'],
            'Alternative_Support_Scope':best['support_scope'],
            'Candidate_Count':len(tested),
        })
    columns=[
        'Parameter','Field','Current','Influence','Direction','Potential_improvement',
        'Best_alternative','Best_P_crystalline','Best_Alternative_Detail',
        'Alternative_Validity_Score','Alternative_Support_Count',
        'Alternative_Support_Scope','Candidate_Count',
    ]
    frame=pd.DataFrame(rows,columns=columns)
    if frame.empty:
        return frame,base_cryst
    return frame.sort_values('Influence',ascending=False).reset_index(drop=True),base_cryst



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
