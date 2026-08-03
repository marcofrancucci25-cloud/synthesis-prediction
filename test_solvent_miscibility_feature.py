import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import applicability, optimize_joint
from src.solvent_miscibility import miscibility_check

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

print("=== 1. Modulo src/solvent_miscibility.py in isolamento ===")
check("1.1 solvente singolo -> checked=False, nessun crash", miscibility_check("DMF")["checked"] is False)
check("1.2 DMF/H2O -> nessun problema noto", miscibility_check("DMF/H2O")["flag"] is None)
check("1.3 Water/Toluene -> immiscibile", miscibility_check("Water/Toluene")["flag"] == "immiscible")
check("1.4 Water/Hexane -> immiscibile", miscibility_check("Water/Hexane")["flag"] == "immiscible")
check("1.5 Water/DCM -> immiscibile", miscibility_check("Water/DCM")["flag"] == "immiscible")
check("1.6 Water/Ethyl acetate -> parzialmente miscibile (non 'immiscible')",
      miscibility_check("Water/Ethyl acetate")["flag"] == "partially_miscible")
check("1.7 H2O/EtOH -> nessun problema", miscibility_check("H2O/EtOH")["flag"] is None)
check("1.8 MeOH/Toluene (nessuna acqua) -> nessun problema", miscibility_check("MeOH/Toluene")["flag"] is None)
check("1.9 miscela a 3 componenti rileva comunque l'acqua immiscibile",
      miscibility_check("water/dmf/toluene")["flag"] == "immiscible")
check("1.10 case-insensitive", miscibility_check("WATER/TOLUENE")["flag"] == "immiscible")
check("1.11 nessun crash con stringa vuota", miscibility_check("")["checked"] is False)
check("1.12 nessun crash con None", miscibility_check(None)["checked"] is False)

print()
print("=== 2. Integrazione in applicability() ===")
base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="Water/Toluene", Additivo_Colinker="Nessuno",
    Temperatura_C=100.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0
ad = applicability(base)
check("2.1 applicability() include il campo 'miscibility'", "miscibility" in ad and ad["miscibility"]["flag"] == "immiscible")
base_ok = dict(base); base_ok["Solvente"] = "DMF"
ad_ok = applicability(base_ok)
check("2.2 il punteggio AD principale non cambia per via della miscibilità (concetti separati)",
      abs(ad["score"] - ad_ok["score"]) < 0.35)

print()
print("=== 3. Integrazione nell'ottimizzatore ===")
r, m = optimize_joint(base, objective="Balanced conditions", n_samples=1500, top_n=10)
check("3.1 colonna Miscibility_Flag presente", "Miscibility_Flag" in r.columns)
check("3.2 nessun crash, risultati comunque prodotti", len(r) > 0)

base_normal = dict(base); base_normal["Solvente"] = "DMF"
r2, m2 = optimize_joint(base_normal, objective="Balanced conditions", n_samples=1500, top_n=10)
check("3.3 caso normale (DMF): nessun avviso di immiscibilità nei metadata",
      not any("miscib" in w.lower() for w in m2.get("warnings", [])), f"{m2.get('warnings')}")

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
