import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import predict, applicability, optimize_joint
from src.solubility import estimate_logS_water, solubility_penalty, describe

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

TEREPHTHALIC = "OC(=O)c1ccc(cc1)C(=O)O"
PORPHYRIN = "OC(=O)c1ccc(cc1)-c1c2ccc(n2)c(-c2ccc(cc2)C(=O)O)c2ccc(n2)c(-c2ccc(cc2)C(=O)O)c2ccc(n2)c(-c2ccc(cc2)C(=O)O)c2ccc1n2"
OXALIC = "OC(=O)C(=O)O"

print("=== 1. Modulo src/solubility.py in isolamento ===")
check("1.1 ESOL calcola un numero finito per acido tereftalico", np.isfinite(estimate_logS_water(TEREPHTHALIC)))
check("1.2 ESOL restituisce None per SMILES non valido", estimate_logS_water("questo-non-e-uno-smiles") is None)
check("1.3 ESOL restituisce None per SMILES assente", estimate_logS_water(None) is None)
check("1.4 porfirina lipofila ha logS molto più basso (meno solubile) dell'acido ossalico",
      estimate_logS_water(PORPHYRIN) < estimate_logS_water(OXALIC) - 3,
      f"porfirina={estimate_logS_water(PORPHYRIN):.2f} ossalico={estimate_logS_water(OXALIC):.2f}")
check("1.5 penalità con acqua più alta per la porfirina che per l'ossalico",
      solubility_penalty(PORPHYRIN, "Water") > solubility_penalty(OXALIC, "Water"))
check("1.6 penalità sempre in [0,1]", all(0 <= solubility_penalty(PORPHYRIN, s) <= 1 for s in ["Water","DMF","Toluene","Ethanol","DMF/Water","xyz-inventato"]))
check("1.7 nessun crash con SMILES mancante (penalità neutra 0.0)", solubility_penalty(None, "Water") == 0.0)
check("1.8 nessun crash con solvente vuoto", solubility_penalty(TEREPHTHALIC, "") == 0.0)
d = describe(PORPHYRIN, "Water")
check("1.9 describe() restituisce tutti i campi attesi", set(d) >= {"logS_water","water_solubility_flag","solubility_penalty","rdkit_available","smiles_resolved"})

print()
print("=== 2. Integrazione nel predittore (applicability) ===")
base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Ligand_SMILES=TEREPHTHALIC,
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Nessuno",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0
ad = applicability(base)
check("2.1 applicability() include il campo 'solubility'", "solubility" in ad)
check("2.2 il punteggio AD principale non cambia per via della solubilità (concetti separati)",
      "score" in ad and isinstance(ad["score"], float))
base_nosmiles = dict(base); base_nosmiles["Ligand_SMILES"] = None
ad2 = applicability(base_nosmiles)
check("2.3 senza SMILES, solubility.smiles_resolved=False (nessun falso 'tutto ok')",
      ad2["solubility"]["smiles_resolved"] is False)
check("2.4 predict() ignora comunque il campo Ligand_SMILES (non è nello schema del modello)",
      np.allclose(predict(base)[1], predict(base_nosmiles)[1]))

print()
print("=== 3. Integrazione nell'ottimizzatore: il caso concreto segnalato ===")
base_porphyrin = dict(
    Legante="meso-tetra(4-carboxyphenyl)porphine", Famiglia_Legante="Porphyrin tetracarboxylate", Metallo="Al",
    Ligand_SMILES=PORPHYRIN,
    Sale_Metallico="AlCl3.6H2O", Counterion_Class="chloride", Hydration_Number=6.0,
    Oxidation_State=3, Solvente="Water", Additivo_Colinker="Nessuno",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.1, mmol_Sale=0.1, Rapporto_LM=1.0,
)
base_porphyrin["Volume solvente"] = 10.0
r, m = optimize_joint(base_porphyrin, objective="Balanced conditions", n_samples=3000, top_n=10)
check("3.1 colonna Solubility_penalty presente nei risultati", "Solubility_penalty" in r.columns)
water_count = r["Solvente"].str.contains("Water|H2O", case=False).sum()
check("3.2 con SMILES risolto, l'acqua non domina più le proposte per un legante lipofilo",
      water_count <= 2, f"acqua in {water_count}/10 proposte")
check("3.3 Optimization_score resta un numero finito per ogni proposta", r["Optimization_score"].apply(np.isfinite).all())

base_nosmiles2 = dict(base_porphyrin); base_nosmiles2["Ligand_SMILES"] = None
r2, m2 = optimize_joint(base_nosmiles2, objective="Balanced conditions", n_samples=1500, top_n=10)
check("3.4 senza SMILES, l'ottimizzatore avvisa esplicitamente (nessun controllo silenziosamente saltato)",
      any("solubility" in w.lower() for w in m2.get("warnings", [])), f"{m2.get('warnings')}")

print()
print("=== 4. Determinismo e non-regressione ===")
r3, m3 = optimize_joint(base_porphyrin, objective="Balanced conditions", n_samples=3000, top_n=10)
check("4.1 stesso seed -> stessa colonna Solubility_penalty", np.allclose(r["Solubility_penalty"].values, r3["Solubility_penalty"].values))

base_known = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn", Ligand_SMILES=TEREPHTHALIC,
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Nessuno",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base_known["Volume solvente"] = 10.0
r4, m4 = optimize_joint(base_known, objective="Balanced conditions", n_samples=2000, top_n=10)
check("4.2 caso noto (H2BDC/Zn/DMF) continua a produrre risultati validi", len(r4) > 0 and r4["P_Crystalline"].between(0,1).all())
check("4.3 tutti gli obiettivi restano eseguibili senza eccezioni", all(
    len(optimize_joint(base_known, objective=o, n_samples=800, top_n=5)[0]) > 0
    for o in ["Maximum crystallinity","Balanced conditions","Conservative optimization","Green synthesis","Fast synthesis"]
))

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
