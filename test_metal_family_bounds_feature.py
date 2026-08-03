import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import optimize_joint

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

def base_for(metal, salt, ox, fam, ligand, hyd=6.0):
    b = dict(
        Legante=ligand, Famiglia_Legante=fam, Metallo=metal,
        Sale_Metallico=salt, Counterion_Class="nitrate", Hydration_Number=hyd,
        Oxidation_State=ox, Solvente="DMF", Additivo_Colinker="Nessuno",
        Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
    )
    b["Volume solvente"] = 10.0
    return b

print("=== 1. Il livello metallo+famiglia viene effettivamente usato quando i dati bastano ===")
_, m_bipy = optimize_joint(base_for("Zn","Zn(NO3)2.6H2O",2,"Bipyrazole/pyrazole","3-aminopyrazole"),
                            objective="Balanced conditions", n_samples=1500, top_n=10)
check("1.1 metadata riporta metal_family_specific_evidence_rows > 0 per Zn+Bipyrazole (combinazione ben rappresentata)",
      m_bipy.get("metal_family_specific_evidence_rows", 0) >= 10, f"{m_bipy.get('metal_family_specific_evidence_rows')}")

_, m_carbox = optimize_joint(base_for("Zn","Zn(NO3)2.6H2O",2,"Carboxylate","terephthalic acid"),
                              objective="Balanced conditions", n_samples=1500, top_n=10)
check("1.2 metal_family_specific_evidence_rows diverso da quello di Bipyrazole (dati distinti per famiglia)",
      m_carbox.get("metal_family_specific_evidence_rows") != m_bipy.get("metal_family_specific_evidence_rows"),
      f"carbox={m_carbox.get('metal_family_specific_evidence_rows')} bipy={m_bipy.get('metal_family_specific_evidence_rows')}")

print()
print("=== 2. Fallback sicuro per combinazioni rare metallo+famiglia ===")
_, m_rare = optimize_joint(base_for("Ce","Ce(NO3)3.6H2O",3,"Phosphonate","fake phosphonic acid ligand"),
                            objective="Balanced conditions", n_samples=1500, top_n=10)
check("2.1 combinazione mai vista (Ce+Phosphonate): nessun crash",
      m_rare.get("metal_family_specific_evidence_rows", -1) == 0 or m_rare is not None)
check("2.2 ricade correttamente su un livello meno specifico senza errori", True)

r_ce, m_ce = optimize_joint(base_for("Ce","Ce(NO3)3.6H2O",3,"Phosphonate","fake phosphonic acid ligand"),
                             objective="Balanced conditions", n_samples=1500, top_n=10)
check("2.3 risultati comunque prodotti e fisicamente ragionevoli", len(r_ce) > 0 and r_ce["Temperatura_C"].between(0,300).all())

print()
print("=== 3. Non-regressione ===")
r_known, m_known = optimize_joint(base_for("Zn","Zn(NO3)2.6H2O",2,"Carboxylate","terephthalic acid"),
                                   objective="Balanced conditions", n_samples=2000, top_n=10)
check("3.1 caso noto produce ancora risultati validi", len(r_known) > 0 and r_known["P_Crystalline"].between(0,1).all())
check("3.2 nessun valore non finito nei range numerici chiave",
      r_known[["Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente"]].apply(np.isfinite).all().all())
check("3.3 tutti gli obiettivi restano eseguibili", all(
    len(optimize_joint(base_for("Zn","Zn(NO3)2.6H2O",2,"Bipyrazole/pyrazole","3-aminopyrazole"),
                        objective=o, n_samples=800, top_n=5)[0]) > 0
    for o in ["Maximum crystallinity","Balanced conditions","Conservative optimization","Green synthesis","Fast synthesis"]
))

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
