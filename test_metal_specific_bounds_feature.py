import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import optimize_joint

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

def base_for(metal, salt, ox, hyd=6.0):
    b = dict(
        Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo=metal,
        Sale_Metallico=salt, Counterion_Class="nitrate", Hydration_Number=hyd,
        Oxidation_State=ox, Solvente="DMF", Additivo_Colinker="Nessuno",
        Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
    )
    b["Volume solvente"] = 10.0
    return b

print("=== 1. Range specifici per metallo, invece che globali ===")
r_zn, _ = optimize_joint(base_for("Zn", "Zn(NO3)2.6H2O", 2), objective="Balanced conditions", n_samples=2000, top_n=10)
check("1.1 Zn: lo stato di ossidazione proposto resta sempre 2 (mai 3 o 4)",
      set(r_zn["Oxidation_State"].unique()) <= {2}, f"{sorted(r_zn['Oxidation_State'].unique())}")

r_al, _ = optimize_joint(base_for("Al", "AlCl3.6H2O", 3), objective="Balanced conditions", n_samples=2000, top_n=10)
check("1.2 Al: lo stato di ossidazione proposto resta sempre 3 (mai 2)",
      set(r_al["Oxidation_State"].unique()) <= {3}, f"{sorted(r_al['Oxidation_State'].unique())}")
check("1.3 Al: il numero di idratazione riflette la chimica reale dell'alluminio (alto, non quello globale basso)",
      r_al["Hydration_Number"].median() >= 6, f"mediana={r_al['Hydration_Number'].median()}")

r_cu, _ = optimize_joint(base_for("Cu", "Cu(NO3)2.3H2O", 2, hyd=3.0), objective="Balanced conditions", n_samples=2000, top_n=10)
check("1.4 Cu: il numero di idratazione riflette la chimica reale del rame (basso, diverso da Al)",
      r_cu["Hydration_Number"].median() < r_al["Hydration_Number"].median(),
      f"Cu mediana={r_cu['Hydration_Number'].median()} vs Al mediana={r_al['Hydration_Number'].median()}")

print()
print("=== 2. Fallback sicuro quando i dati del metallo sono insufficienti/mancanti ===")
r_fake, m_fake = optimize_joint(
    dict(base_for("Xx", "Xx(NO3)2", 2)), objective="Balanced conditions", n_samples=1500, top_n=10)
check("2.1 metallo inesistente non causa crash (nessun dato -> fallback al dataset globale)", len(r_fake) > 0)
check("2.2 range di temperatura resta comunque fisicamente ragionevole col fallback",
      r_fake["Temperatura_C"].between(0, 300).all())

print()
print("=== 3. Non-regressione: caso noto (Zn/H2BDC/DMF) produce ancora risultati sensati ===")
r_known, m_known = optimize_joint(base_for("Zn", "Zn(NO3)2.6H2O", 2), objective="Balanced conditions", n_samples=2000, top_n=10)
check("3.1 risultati non vuoti", len(r_known) > 0)
check("3.2 P_Crystalline in [0,1]", r_known["P_Crystalline"].between(0, 1).all())
check("3.3 nessun valore non finito nei range numerici chiave",
      r_known[["Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente"]].apply(np.isfinite).all().all())

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
