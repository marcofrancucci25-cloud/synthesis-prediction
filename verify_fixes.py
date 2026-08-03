import sys, time
import numpy as np
sys.path.insert(0, ".")
from src.engine import predict, applicability, optimize_joint

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Nessuno",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0

print("=== FIX 1: etichette Strategy non più perse ===")
result, meta = optimize_joint(base, objective="Balanced conditions", n_samples=2000, top_n=10)
print(result[["Rank","Strategy","Optimization_score"]].to_string())
check("Fix1.a 'Best hybrid score' compare in almeno una proposta",
      result["Strategy"].str.contains("Best hybrid score").any())
best_row = result.loc[result["Optimization_score"].idxmax()]
check("Fix1.b la riga col punteggio ibrido più alto porta l'etichetta 'Best hybrid score'",
      "Best hybrid score" in best_row["Strategy"], f"-> {best_row['Strategy']}")
check("Fix1.c nessuna etichetta 'Alternative' spuria sulla riga col punteggio massimo",
      best_row["Strategy"] != "Alternative")

print()
print("=== FIX 2: nessun crash con Oxidation_State='unknown' (stringa letterale) ===")
v = dict(base); v["Oxidation_State"] = "unknown"
try:
    _, p, _ = predict(v)
    check("Fix2 predict() con Oxidation_State='unknown' non va in crash", True, f"P={np.round(p,4)}")
except Exception as e:
    check("Fix2 predict() con Oxidation_State='unknown' non va in crash", False, str(e))

print()
print("=== FIX 3: avviso su allowed_solvents senza precedenti ===")
r, m = optimize_joint(base, objective="Balanced conditions", n_samples=800, top_n=5,
                       constraints={"allowed_solvents": ["totally-fictitious-solvent-xyz"]})
print("warnings:", m.get("warnings"))
check("Fix3.a metadata include 'warnings'", "warnings" in m)
check("Fix3.b avviso presente per solvente mai osservato nei dati", len(m.get("warnings", [])) > 0)
check("Fix3.c il fallback resta comunque utilizzabile (nessun crash)", len(r) > 0)

# controllo anche il caso "buono": nessun warning se il solvente richiesto esiste
r2, m2 = optimize_joint(base, objective="Balanced conditions", n_samples=800, top_n=5,
                         constraints={"allowed_solvents": ["DMF"]})
check("Fix3.d nessun warning quando il solvente richiesto è valido", len(m2.get("warnings", [])) == 0, f"{m2.get('warnings')}")

print()
print("=== FIX 4: avviso su famiglia legante incoerente ===")
v_mismatch = dict(base); v_mismatch["Famiglia_Legante"] = "Pyridyl/N-donor"  # H2BDC è un carbossilato
ad_mismatch = applicability(v_mismatch)
ad_ok = applicability(base)
print("AD con famiglia sbagliata:", ad_mismatch["label"], ad_mismatch["score"], "mismatch:", ad_mismatch["family_mismatch"])
print("AD con famiglia corretta:", ad_ok["label"], ad_ok["score"], "mismatch:", ad_ok["family_mismatch"])
check("Fix4.a family_mismatch rilevato quando famiglia e legante sono incoerenti", ad_mismatch["family_mismatch"] is True)
check("Fix4.b nessun mismatch quando famiglia e legante sono coerenti", ad_ok["family_mismatch"] is False)
check("Fix4.c punteggio AD penalizzato in caso di mismatch", ad_mismatch["score"] < ad_ok["score"])
check("Fix4.d 'Inside domain' non assegnato quando c'è un mismatch, anche con punteggio alto",
      ad_mismatch["label"] != "Inside domain")
check("Fix4.e caso base resta 'Inside domain' come prima della modifica",
      ad_ok["label"] == "Inside domain", f"-> {ad_ok['label']} score={ad_ok['score']}")

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    for f in FAIL: print("  FAIL:", f)
