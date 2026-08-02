#!/usr/bin/env python3
"""Resolve unique ligand names through PubChem and retain provenance/confidence."""
from __future__ import annotations

import argparse, time
from pathlib import Path
from urllib.parse import quote
import requests
import pandas as pd
from rdkit import Chem

CURATED = {
    "3-amino-4,4'-bipyrazole": "Nc1[nH]ncc1-c1cn[nH]c1",
    "3,3'-diamino-4,4'-bipyrazole": "Nc1[nH]ncc1-c1cn[nH]c1N",
    "3,5-diamino-4,4'-bipyrazole": "Nc1[nH]nc(N)c1-c1cn[nH]c1",
    "3,3'-dimethyl-4,4'-bipyrazole": "Cc1[nH]ncc1-c1cn[nH]c1C",
    "NO2BPz": "O=[N+]([O-])c1[nH]ncc1-c1cn[nH]c1",
    "(NO2)2BPz": "O=[N+]([O-])c1[nH]ncc1-c1cn[nH]c1[N+](=O)[O-]",
    "H2BPZNH2": "Nc1[nH]ncc1-c1cn[nH]c1",
    "3,5-DiMePz-4-COOH": "Cc1[nH]nc(C)c1C(=O)O",
    "3-MePz-4-COOH": "Cc1[nH]ncc1C(=O)O",
    "5-MePz-4-COOH": "Cc1c(C(=O)O)c[nH]n1",
}


def resolve(name: str, session: requests.Session):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + quote(name, safe="") +
           "/property/ConnectivitySMILES,InChIKey,IUPACName/JSON")
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return None, None, f"pubchem_http_{response.status_code}"
        prop = response.json()["PropertyTable"]["Properties"][0]
        smi = prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES")
        mol = Chem.MolFromSmiles(str(smi or ""))
        if mol is None:
            return None, None, "invalid_smiles"
        return Chem.MolToSmiles(mol, canonical=True), prop.get("InChIKey"), "resolved"
    except Exception as exc:
        return None, None, f"error:{type(exc).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/knowledge_database.csv")
    ap.add_argument("--out", default="data/ligand_structure_registry_v10_5.csv")
    args = ap.parse_args()
    names = sorted(pd.read_csv(args.data)["Legante"].dropna().astype(str).unique())
    rows=[]; session=requests.Session()
    for i,name in enumerate(names,1):
        if name in CURATED:
            mol=Chem.MolFromSmiles(CURATED[name]); smi=Chem.MolToSmiles(mol,canonical=True)
            key=Chem.MolToInchiKey(mol); status="curated_name_to_structure"
        else:
            smi,key,status=resolve(name,session)
            if not smi and " (" in name:
                smi,key,status=resolve(name.split(" (")[0],session)
        rows.append({"Legante":name,"Ligand_SMILES":smi,"Ligand_InChIKey":key,
                     "Structure_Status":status,"Structure_Source":("Curated project registry" if status.startswith("curated") else ("PubChem PUG REST" if smi else "unresolved"))})
        print(f"[{i:03d}/{len(names):03d}] {name}: {status}")
        time.sleep(0.08)
    pd.DataFrame(rows).to_csv(args.out,index=False)

if __name__ == "__main__": main()
