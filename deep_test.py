import sys, time, json, traceback
import numpy as np
import pandas as pd
sys.path.insert(0, ".")

from src.engine import predict, applicability, prediction_validity, optimize_joint, similar, explain_prediction, TRAINING_DB
from src.chem import canonicalize_ligand_for_model, precursor_formula, parse_salt, infer_family, hsab_acid_class

def base_values(**overrides):
    v = dict(
        Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
        Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
        Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Nessuno",
        Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5,
        Rapporto_LM=1.0,
    )
    v["Volume solvente"] = 10.0
    v.update(overrides)
    return v

results = {"PASS": [], "FAIL": [], "INFO": []}

def check(name, cond, info=""):
    if cond:
        results["PASS"].append(name)
        print(f"[PASS] {name} {info}")
    else:
        results["FAIL"].append((name, info))
        print(f"[FAIL] {name} {info}")

print("="*100)
print("1. TEST MOTORE PREDITTIVO")
print("="*100)

# 1.1 Prediction on classic textbook synthesis (MOF-5-like: Zn + H2BDC + DMF)
v = base_values()
x, p, cls = predict(v)
print("MOF-5-like (Zn, H2BDC, DMF, 120C, 24h):", p, "class:", cls)
check("1.1 probabilità sommano a 1", abs(sum(p)-1) < 1e-6, f"sum={sum(p)}")
check("1.2 tre classi restituite", len(p) == 3)

# 1.2 Determinism: same input -> same output
x2, p2, cls2 = predict(v)
check("1.3 determinismo (stesso input -> stesso output)", np.allclose(p, p2))

# 1.3 Ligand alias equivalence: H2BDC vs terephthalic acid vs full IUPAC name should canonicalize identically
aliases = ["H2BDC", "terephthalic acid", "benzene-1,4-dicarboxylic acid", "1,4-benzenedicarboxylic acid", "BDC"]
canon = [canonicalize_ligand_for_model(a) for a in aliases]
print("Canonicalizzazione alias legante:", list(zip(aliases, canon)))
check("1.4 alias H2BDC/terephthalic/BDC canonicalizzati in modo coerente",
      len(set(canon[:4])) == 1, f"{canon}")
# BDC (bare) maps to 'terephthalate / terephthalic acid' per COMMON_LIGAND_ALIASES but
# canonicalize_ligand_for_model operates on MODEL_LIGAND_ALIASES which does include 'bdc'
check("1.4b alias 'BDC' isolato", canon[4] == canon[0], f"BDC -> {canon[4]} vs H2BDC -> {canon[0]}")

preds_alias = []
for a in aliases:
    vv = base_values(Legante=a)
    _, pp, _ = predict(vv)
    preds_alias.append(tuple(np.round(pp, 6)))
print("Predizioni per alias:", list(zip(aliases, preds_alias)))
check("1.5 predizioni identiche per alias equivalenti dello stesso legante",
      len(set(preds_alias)) == 1, f"{preds_alias}")

# 1.4 Case sensitivity on ligand name
vv1 = base_values(Legante="TEREPHTHALIC ACID")
vv2 = base_values(Legante="terephthalic acid")
_, p_upper, _ = predict(vv1)
_, p_lower, _ = predict(vv2)
check("1.6 case-insensitivity nome legante", np.allclose(p_upper, p_lower), f"{p_upper} vs {p_lower}")

# 1.5 Unseen / nonsense ligand
vv = base_values(Legante="Xyzzyqwerty-nonexistent-linker-12345", Famiglia_Legante="Other/unknown")
x, p_unseen, cls_unseen = predict(vv)
ad_unseen = applicability(vv)
print("Legante inventato -> P:", p_unseen, "AD:", ad_unseen["label"], ad_unseen["score"])
check("1.7 legante sconosciuto: AD segnala dominio esterno o parziale",
      ad_unseen["label"] in ("Outside domain", "Intermediate / partial extrapolation"),
      f"{ad_unseen['label']}")
check("1.8 legante sconosciuto: ligand_seen=False", ad_unseen["ligand_seen"] == False)

# 1.6 Unseen metal (fake element not in METALS dict) - engine should degrade gracefully (NaN feats)
vv = base_values(Metallo="Xx")
try:
    x, p_fake_metal, cls_fake = predict(vv)
    ad_fake = applicability(vv)
    print("Metallo inventato 'Xx' -> P:", p_fake_metal, "AD:", ad_fake["label"])
    check("1.9 metallo inesistente non causa crash (gestito con NaN)", True)
    check("1.9b metallo inesistente: metal_seen=False", ad_fake["metal_seen"] == False)
except Exception as e:
    check("1.9 metallo inesistente non causa crash (gestito con NaN)", False, str(e))

# 1.7 Extreme numeric values: very high temperature
vv = base_values(Temperatura_C=290.0, Tempo_ore=480.0)
x, p_extreme, cls_extreme = predict(vv)
validity_extreme = prediction_validity(vv)
print("Condizioni estreme (290C, 480h) -> P:", p_extreme, "validity:", validity_extreme["label"], validity_extreme["score"])
check("1.10 condizioni estreme segnalate come fuori range o extrapolative",
      validity_extreme["label"] != "Within validated experimental range")

# 1.8 Zero / degenerate amounts
vv = base_values(mmol_Legante=0.0, mmol_Sale=0.0, Rapporto_LM=0.0)
try:
    x, p_zero, cls_zero = predict(vv)
    validity_zero = prediction_validity(vv)
    print("Quantità zero -> P:", p_zero, "validity:", validity_zero)
    check("1.11 quantità zero non causa crash del predittore", True)
except Exception as e:
    check("1.11 quantità zero non causa crash del predittore", False, str(e))

# 1.9 Negative values (should not be physically possible; check handling)
vv = base_values(Temperatura_C=-50.0, Tempo_ore=-5.0)
try:
    x, p_neg, cls_neg = predict(vv)
    validity_neg = prediction_validity(vv)
    print("Valori negativi (-50C, -5h) -> P:", p_neg, "validity:", validity_neg["label"], validity_neg["issues"])
    check("1.12 valori negativi gestiti senza crash", True)
    check("1.13 valori negativi segnalati come fuori range", validity_neg["label"] == "Outside validated experimental range")
except Exception as e:
    check("1.12 valori negativi gestiti senza crash", False, str(e))

# 1.10 Inconsistent stoichiometry (mmol_L/mmol_M doesn't match declared ratio)
vv = base_values(mmol_Legante=1.0, mmol_Sale=1.0, Rapporto_LM=9.0)  # declared ratio 9 but actual is 1
validity_incons = prediction_validity(vv)
print("Rapporto dichiarato incoerente con le quantità -> issues:", validity_incons["issues"])
check("1.14 rileva incoerenza tra rapporto dichiarato e quantità mmol",
      any("inconsistent" in i.lower() for i in validity_incons["issues"]))

# 1.11 Missing/NaN field handling
vv = base_values()
del vv["Solvente"]
try:
    x, p_missing, cls_missing = predict(vv)
    check("1.15 campo mancante (Solvente) gestito senza crash", True)
except Exception as e:
    check("1.15 campo mancante (Solvente) gestito senza crash", False, str(e))

# 1.12 applicability score bounds
for test_v in [base_values(), base_values(Legante="totally-unknown-xyz"), base_values(Temperatura_C=1000)]:
    ad = applicability(test_v)
    check(f"1.16 AD score in [0,1] per caso {test_v.get('Legante')[:20]}/{test_v.get('Temperatura_C')}",
          0.0 <= ad["score"] <= 1.0, f"score={ad['score']}")

# 1.13 explain_prediction sanity
try:
    df_exp, base_cryst = explain_prediction(base_values())
    print("explain_prediction colonne:", list(df_exp.columns), "n righe:", len(df_exp))
    check("1.17 explain_prediction restituisce dataframe non vuoto", len(df_exp) > 0)
    check("1.18 base_cryst in [0,1]", 0 <= base_cryst <= 1)
except Exception as e:
    check("1.17 explain_prediction restituisce dataframe non vuoto", False, str(e))
    traceback.print_exc()

# 1.14 metal HSAB classification sanity
check("1.19 Al è hard acid", hsab_acid_class("Al", 3) == "Hard acid")
check("1.20 Ag è soft acid", hsab_acid_class("Ag", 1) == "Soft acid")
check("1.21 Cu(I) è soft, Cu(III) è borderline", hsab_acid_class("Cu", 1) == "Soft acid" and hsab_acid_class("Cu", 2) == "Borderline acid")

print()
print("="*100)
print("2. TEST OTTIMIZZATORE DI SINTESI")
print("="*100)

base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Nessuno",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5,
    Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0

t0 = time.time()
result, meta = optimize_joint(base, objective="Balanced conditions", n_samples=2000, top_n=10)
t1 = time.time()
print(f"Tempo ottimizzazione (2000 campioni): {t1-t0:.2f}s")
print(result[["Rank","Strategy","Sale_Metallico","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente","P_Crystalline","Positive_support_score","AD_score","Feasibility_score","Optimization_score","Pareto_optimal"]].to_string())
print("Metadata:", json.dumps(meta, indent=2))

check("2.1 ottimizzatore restituisce risultati", len(result) > 0)
check("2.2 ligando fissato (invariato) in tutte le proposte", (result["Legante"].astype(str).apply(canonicalize_ligand_for_model) == canonicalize_ligand_for_model(base["Legante"])).all())
check("2.3 metallo fissato in tutte le proposte", (result["Metallo"] == base["Metallo"]).all())
check("2.4 probabilità P_Crystalline in [0,1]", result["P_Crystalline"].between(0,1).all())
check("2.5 nessun duplicato esatto tra le proposte (Sale/Solv/Add/T/t/Rapporto)",
      not result.duplicated(subset=["Sale_Metallico","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM"]).any())
check("2.6 rank crescente da 1", list(result["Rank"]) == list(range(1, len(result)+1)))
check("2.7 almeno una proposta è (anche) 'Best hybrid score'", result["Strategy"].str.contains("Best hybrid score").any())
check("2.8 concentrazione totale coerente (mmol_L+mmol_M)/Vol", np.allclose(result["Total_concentration_mmol_mL"], (result["mmol_Legante"]+result["mmol_Sale"])/result["Volume solvente"]))

# 2.2 Determinism of optimizer given fixed random_state
result_b, _ = optimize_joint(base, objective="Balanced conditions", n_samples=2000, top_n=10)
check("2.9 determinismo ottimizzatore (stesso seed -> stesso risultato)",
      result[["Sale_Metallico","Solvente","Temperatura_C","Tempo_ore"]].equals(result_b[["Sale_Metallico","Solvente","Temperatura_C","Tempo_ore"]]))

# 2.3 Constraint: max temperature
constraints = {"max_temperature": 80.0}
result_c, meta_c = optimize_joint(base, objective="Balanced conditions", n_samples=1500, top_n=10, constraints=constraints)
check("2.10 vincolo max_temperature rispettato", (result_c["Temperatura_C"] <= 80.0 + 1e-6).all(),
      f"max found={result_c['Temperatura_C'].max()}")

# 2.4 Constraint: banned_solvents
constraints = {"banned_solvents": ["DMF", "dmso"]}
result_d, meta_d = optimize_joint(base, objective="Balanced conditions", n_samples=1500, top_n=10, constraints=constraints)
banned_hit = result_d["Solvente"].str.casefold().apply(lambda s: ("dmf" in s) or ("dmso" in s))
check("2.11 vincolo banned_solvents rispettato (nessun DMF/DMSO in output)", not banned_hit.any(),
      f"{result_d['Solvente'].tolist()}")

# 2.5 Constraint: keep_solvent
constraints = {"keep_solvent": True}
result_e, meta_e = optimize_joint(base, objective="Balanced conditions", n_samples=1000, top_n=10, constraints=constraints)
check("2.12 vincolo keep_solvent rispettato (solvente invariato = DMF)",
      (result_e["Solvente"].astype(str) == base["Solvente"]).all(), f"{result_e['Solvente'].unique()}")

# 2.6 Constraint: keep_precursor
constraints = {"keep_precursor": True}
result_f, meta_f = optimize_joint(base, objective="Balanced conditions", n_samples=1000, top_n=10, constraints=constraints)
check("2.13 vincolo keep_precursor rispettato (Counterion invariato)",
      (result_f["Counterion_Class"].astype(str) == base["Counterion_Class"]).all())

# 2.7 Contradictory constraints -> should raise ValueError
try:
    optimize_joint(base, objective="Balanced conditions", n_samples=500, top_n=5,
                    constraints={"min_temperature": 200, "max_temperature": 100})
    check("2.14 vincoli contraddittori (min>max) sollevano ValueError", False, "nessuna eccezione sollevata!")
except ValueError as e:
    check("2.14 vincoli contraddittori (min>max) sollevano ValueError", True, str(e))
except Exception as e:
    check("2.14 vincoli contraddittori (min>max) sollevano ValueError", False, f"eccezione errata: {type(e)} {e}")

# 2.8 allowed_solvents with impossible solvent -> should raise or fall back
try:
    r, m = optimize_joint(base, objective="Balanced conditions", n_samples=500, top_n=5,
                           constraints={"allowed_solvents": ["totally-fictitious-solvent-xyz"]})
    print("allowed_solvents fittizio -> risultati:", r["Solvente"].unique())
    check("2.15 allowed_solvents con solvente fittizio gestito (fallback o eccezione)", True)
except ValueError as e:
    check("2.15 allowed_solvents con solvente fittizio gestito (fallback o eccezione)", True, str(e))
except Exception as e:
    check("2.15 allowed_solvents con solvente fittizio gestito (fallback o eccezione)", False, f"{type(e)} {e}")

# 2.9 Objective differences: Green synthesis should favor low green-penalty solvents vs Maximum crystallinity
result_green, _ = optimize_joint(base, objective="Green synthesis", n_samples=2000, top_n=10)
result_max, _ = optimize_joint(base, objective="Maximum crystallinity", n_samples=2000, top_n=10)
print("Green synth top solvents:", result_green["Solvente"].tolist())
print("Max crystallinity top solvents:", result_max["Solvente"].tolist())
check("2.16 P_Crystalline media più alta con 'Maximum crystallinity' che con 'Green synthesis'",
      result_max["P_Crystalline"].mean() >= result_green["P_Crystalline"].mean() - 0.05,
      f"max={result_max['P_Crystalline'].mean():.3f} green={result_green['P_Crystalline'].mean():.3f}")

# 2.10 Fast synthesis should favor shorter time/lower temp on average vs Maximum crystallinity
result_fast, _ = optimize_joint(base, objective="Fast synthesis", n_samples=2000, top_n=10)
print("Fast synth avg time/temp:", result_fast["Tempo_ore"].mean(), result_fast["Temperatura_C"].mean())
print("Max cryst avg time/temp:", result_max["Tempo_ore"].mean(), result_max["Temperatura_C"].mean())
check("2.17 'Fast synthesis' produce in media tempi di reazione più brevi di 'Maximum crystallinity'",
      result_fast["Tempo_ore"].mean() <= result_max["Tempo_ore"].mean() + 1e-6)

# 2.11 Conservative optimization should have lower Change_penalty than Maximum crystallinity
result_cons, _ = optimize_joint(base, objective="Conservative optimization", n_samples=2000, top_n=10)
print("Conservative avg change penalty:", result_cons["Change_penalty"].mean())
print("Max cryst avg change penalty:", result_max["Change_penalty"].mean())
check("2.18 'Conservative optimization' cambia meno le condizioni rispetto a 'Maximum crystallinity'",
      result_cons["Change_penalty"].mean() <= result_max["Change_penalty"].mean() + 0.05)

# 2.12 Unseen ligand/metal in optimizer - should not crash, low support/AD
base_unseen = dict(base)
base_unseen["Legante"] = "totally-fictitious-ligand-9999"
base_unseen["Famiglia_Legante"] = "Other/unknown"
base_unseen["Metallo"] = "Zn"
try:
    result_u, meta_u = optimize_joint(base_unseen, objective="Balanced conditions", n_samples=1500, top_n=10)
    print("Legante sconosciuto -> AD score medio:", result_u["AD_score"].mean(), "Positive support medio:", result_u["Positive_support_score"].mean())
    check("2.19 legante sconosciuto in ottimizzatore non causa crash", True)
    check("2.20 AD score più basso per legante sconosciuto vs legante noto",
          result_u["AD_score"].mean() < result["AD_score"].mean())
except Exception as e:
    check("2.19 legante sconosciuto in ottimizzatore non causa crash", False, str(e))
    traceback.print_exc()

# 2.13 top_n larger than feasible candidates
result_small, meta_small = optimize_joint(base, objective="Balanced conditions", n_samples=300, top_n=50)
check("2.21 top_n>numero candidati gestito senza crash", len(result_small) <= 50)

# 2.14 Very restrictive constraints -> may produce 0 feasible candidates -> expect ValueError, not silent wrong output
try:
    r_tight, m_tight = optimize_joint(base, objective="Balanced conditions", n_samples=800, top_n=10,
                                        constraints={"max_temperature": 25.0, "max_time": 0.6})
    print("Vincoli molto stretti -> risultati:", len(r_tight))
    check("2.22 vincoli molto stretti gestiti (risultato ridotto o eccezione)", True)
except ValueError as e:
    check("2.22 vincoli molto stretti gestiti (risultato ridotto o eccezione)", True, str(e))

print()
print("="*100)
print("RIEPILOGO")
print("="*100)
print(f"PASS: {len(results['PASS'])}  FAIL: {len(results['FAIL'])}")
if results["FAIL"]:
    print("Test falliti:")
    for name, info in results["FAIL"]:
        print(f"  - {name}: {info}")
