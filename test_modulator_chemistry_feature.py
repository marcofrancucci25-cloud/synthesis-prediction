import sys
sys.path.insert(0, ".")
import numpy as np
from src.engine import applicability, optimize_joint
from src.modulator_chemistry import modulator_compatibility

PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

print("=== 1. Modulo src/modulator_chemistry.py in isolamento ===")
r = modulator_compatibility("Carboxylate", "Nessuno")
check("1.1 'Nessuno' -> checked=False, nessun crash", r["checked"] is False)
r = modulator_compatibility("Carboxylate", "Acido acetico")
check("1.2 acido acetico su Carboxylate -> acidità comparabile", r["checked"] and r["verdict"] == "comparable_acidity")
r = modulator_compatibility("Bipyrazole/pyrazole", "Acido acetico")
check("1.3 acido acetico su Bipyrazole/pyrazole -> modulatore troppo debole", r["checked"] and r["verdict"] == "modulator_too_weak")
r = modulator_compatibility("Carboxylate", "TFA")
check("1.4 TFA su Carboxylate -> modulatore molto più forte", r["checked"] and r["verdict"] == "modulator_much_stronger")
r = modulator_compatibility("Carboxylate", "Trietilammina (TEA)")
check("1.5 TEA riconosciuta come base, non modulatore competitivo", r["checked"] is False and r["role"] == "base")
r = modulator_compatibility("Carboxylate", "BDC")
check("1.6 BDC riconosciuto come co-linker, non modulatore", r["checked"] is False and r["role"] == "co_linker")
r = modulator_compatibility("Curcumin/β-diketonate", "Acido acetico")
check("1.7 famiglia senza pKa rappresentativo -> checked=False, nessuna finta certezza", r["checked"] is False)
r = modulator_compatibility("Carboxylate", "additivo-mai-visto-xyz")
check("1.8 additivo sconosciuto -> checked=False, nessun crash", r["checked"] is False)
r = modulator_compatibility(None, None)
check("1.9 nessun crash con input completamente vuoti", r["checked"] is False)

print()
print("=== 2. Integrazione in applicability() ===")
base = dict(
    Legante="terephthalic acid", Famiglia_Legante="Carboxylate", Metallo="Zn",
    Sale_Metallico="Zn(NO3)2.6H2O", Counterion_Class="nitrate", Hydration_Number=6.0,
    Oxidation_State=2, Solvente="DMF", Additivo_Colinker="Acido acetico",
    Temperatura_C=120.0, Tempo_ore=24.0, mmol_Legante=0.5, mmol_Sale=0.5, Rapporto_LM=1.0,
)
base["Volume solvente"] = 10.0
ad = applicability(base)
check("2.1 applicability() include il campo 'modulator'", "modulator" in ad and ad["modulator"]["checked"])
base_no_add = dict(base); base_no_add["Additivo_Colinker"] = "Nessuno"
ad_no_add = applicability(base_no_add)
check("2.2 il punteggio AD principale è identico con/senza modulatore (concetti separati)",
      abs(ad["score"] - ad_no_add["score"]) < 1e-9, f"{ad['score']} vs {ad_no_add['score']}")

print()
print("=== 3. Integrazione nell'ottimizzatore ===")
r_opt, m = optimize_joint(base, objective="Balanced conditions", n_samples=2000, top_n=10)
check("3.1 colonna Modulator_Note presente", "Modulator_Note" in r_opt.columns)
check("3.2 nessun valore nullo/crash nella colonna", r_opt["Modulator_Note"].notna().all())
valid_verdicts = {"comparable_acidity", "modulator_too_weak", "modulator_much_stronger", "none", "base", "co_linker", "inert_spacer", "not_applicable", None}
check("3.3 tutti i valori sono verdetti riconosciuti", r_opt["Modulator_Note"].apply(lambda v: v in valid_verdicts).all(),
      f"valori trovati: {r_opt['Modulator_Note'].unique()}")

print()
print("=== 4. Regressione: la famiglia canonicalizzata internamente deve dare lo stesso risultato ===")
# Bug reale trovato durante lo sviluppo: joint_optimize() riceve Famiglia_Legante
# già canonicalizzata al vocabolario interno (es. 'Carbossilati aromatici'),
# non l'etichetta pubblica ('Carboxylate'). Se la tabella dei pKa non gestisce
# entrambe le forme in modo coerente, il verdetto vero e proprio ('comparable_acidity')
# non viene mai calcolato e resta silenziosamente al solo 'role' ('acid_modulator').
r_public = modulator_compatibility("Carboxylate", "Acido acetico", "terephthalic acid")
r_internal = modulator_compatibility("Carbossilati aromatici", "Acido acetico", "terephthalic acid")
check("4.1 etichetta pubblica ed etichetta interna canonicalizzata danno lo stesso verdetto",
      r_public["verdict"] == r_internal["verdict"] == "comparable_acidity",
      f"public={r_public.get('verdict')} internal={r_internal.get('verdict')}")

r_triazole = modulator_compatibility("Imidazolate/azolate", "Acido formico", "1,2,4-triazole derivative")
r_imidazole = modulator_compatibility("Imidazolate/azolate", "Acido formico", "2-methylimidazole")
check("4.2 disambiguazione triazolo vs imidazolo tramite il testo del legante",
      r_triazole.get("ligand_family_pka") != r_imidazole.get("ligand_family_pka"),
      f"triazolo pKa={r_triazole.get('ligand_family_pka')} imidazolo pKa={r_imidazole.get('ligand_family_pka')}")

print()
print(f"RIEPILOGO: {len(PASS)} PASS, {len(FAIL)} FAIL")
for f in FAIL: print("  FAIL:", f)
