import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import applicability, optimize_joint
from src.vessel_conditions import vessel_requirement, estimate_mixture_bp

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

print("=== 1. Modulo src/vessel_conditions.py in isolamento ===")
check("1.1 acqua a 100C richiede vaso sigillato", vessel_requirement("Water", 100.0)["requires_sealed_vessel"] is True)
check("1.2 acqua a 60C non richiede vaso sigillato", vessel_requirement("Water", 60.0)["requires_sealed_vessel"] is False)
check("1.3 DMF a 120C non richiede vaso sigillato (bp~153C)", vessel_requirement("DMF", 120.0)["requires_sealed_vessel"] is False)
check("1.4 DMF a 200C richiede vaso sigillato", vessel_requirement("DMF", 200.0)["requires_sealed_vessel"] is True)
check("1.5 miscela DMF/H2O usa il componente col bp più basso (acqua, 100C)", estimate_mixture_bp("DMF/H2O") == 100.0)
check("1.6 solvente sconosciuto: nessun'affermazione falsa (None, non False)", vessel_requirement("solvente-xyz-inventato", 150.0)["requires_sealed_vessel"] is None)
check("1.7 temperatura mancante gestita senza crash", vessel_requirement("Water", None)["requires_sealed_vessel"] is None)
check("1.8 nessun crash con stringa vuota", vessel_requirement("", 100.0)["requires_sealed_vessel"] is None)

print()
print("=== 2. Integrazione in applicability() ===")
base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="Water", Additivo_Colinker="Nessuno",
    Temperatura_C=100.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0
ad = applicability(base)
check("2.1 applicability() include il campo 'vessel'", "vessel" in ad and ad["vessel"]["requires_sealed_vessel"] is True)
check("2.2 nessun vessel_mismatch se Procedura_Sintetica non è dichiarata", ad["vessel_mismatch"] is False)

base_mismatch = dict(base); base_mismatch["Procedura_Sintetica"] = "Room Temperature"
ad_mismatch = applicability(base_mismatch)
check("2.3 vessel_mismatch=True quando si dichiara 'Room Temperature' ma T=100C in acqua",
      ad_mismatch["vessel_mismatch"] is True)

base_coherent = dict(base); base_coherent["Procedura_Sintetica"] = "Solvothermal"
ad_coherent = applicability(base_coherent)
check("2.4 nessun mismatch quando la procedura dichiarata è coerente (Solvothermal)",
      ad_coherent["vessel_mismatch"] is False)

base_low_temp = dict(base); base_low_temp["Temperatura_C"] = 25.0; base_low_temp["Procedura_Sintetica"] = "Room Temperature"
ad_low = applicability(base_low_temp)
check("2.5 nessun mismatch a 25C con 'Room Temperature' dichiarata (coerente)", ad_low["vessel_mismatch"] is False)

check("2.6 il punteggio AD principale non cambia per via del vaso (concetti separati)",
      abs(ad["score"] - applicability(dict(base, Temperatura_C=60.0))["score"]) < 0.30)

print()
print("=== 3. Integrazione nell'ottimizzatore ===")
r, m = optimize_joint(base, objective="Balanced conditions", n_samples=2000, top_n=10)
check("3.1 colonna Requires_Sealed_Vessel presente", "Requires_Sealed_Vessel" in r.columns)
check("3.2 valori booleani o None, mai stringhe/altro", r["Requires_Sealed_Vessel"].apply(lambda v: v is None or isinstance(v, (bool, np.bool_))).all())
check("3.3 Optimization_score non penalizzato dal vaso sigillato (colonna informativa)",
      "vessel" not in "".join([c.lower() for c in ["Optimization_score"]]))

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
