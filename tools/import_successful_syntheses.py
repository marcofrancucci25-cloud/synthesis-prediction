"""Validate and merge externally curated crystalline MOF synthesis records.

Usage:
  python tools/import_successful_syntheses.py input.csv output.csv

Only explicitly crystalline/PXRD-confirmed records should be supplied. Missing values
remain missing; the script never invents experimental parameters.
"""
from pathlib import Path
import sys, re
import numpy as np
import pandas as pd

REQUIRED={"Legante","Famiglia_Legante","Metallo","Sale_Metallico","Solvente","Temperatura_C","Tempo_ore"}
OPTIONAL=["Additivo_Colinker","mmol_Legante","mmol_Sale","Rapporto_LM","Volume solvente","Hydration_Number","Counterion_Class","Oxidation_State","Procedura_Sintetica","Source_DOI","PXRD_Confirmed"]

def main(inp,out):
    df=pd.read_csv(inp)
    missing=REQUIRED-set(df.columns)
    if missing: raise SystemExit(f"Missing required columns: {sorted(missing)}")
    for c in OPTIONAL:
        if c not in df: df[c]=np.nan
    for c in ["Temperatura_C","Tempo_ore","mmol_Legante","mmol_Sale","Rapporto_LM","Volume solvente","Hydration_Number","Oxidation_State"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df[df["Temperatura_C"].between(0,300) & df["Tempo_ore"].between(0.01,1000)].copy()
    df["PXRD_Confirmed"]=df["PXRD_Confirmed"].fillna(False).astype(bool)
    df["Condition_Signature"]=df.apply(lambda r:"|".join(str(r.get(c,"" )).strip().casefold() for c in ["Metallo","Legante","Sale_Metallico","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM"]),axis=1)
    df=df.drop_duplicates("Condition_Signature").reset_index(drop=True)
    df.to_csv(out,index=False)
    print(f"Validated {len(df)} unique records -> {out}")
if __name__=="__main__":
    if len(sys.argv)!=3: raise SystemExit("Usage: python tools/import_successful_syntheses.py input.csv output.csv")
    main(sys.argv[1],sys.argv[2])
