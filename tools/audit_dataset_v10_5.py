#!/usr/bin/env python3
"""Create deterministic label-quality and leakage audits for the predictor data."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.structural import add_structure_columns

LABELS={0:"Failed / no useful coordination product",1:"Amorphous, uncertain or insufficiently verified product",2:"Crystalline MOF supported by diffraction evidence"}

def main():
    root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/'data/knowledge_database.csv')
    reg=pd.read_csv(root/'data/ligand_structure_registry_v10_5.csv')
    out=add_structure_columns(df,reg)
    prose=out['Procedura_Sintetica'].fillna('').str.lower()
    contradiction=((out.Esito_ML.eq(2)&prose.str.contains(r'no precip|nessun precip|amorph',regex=True)) |
                   (out.Esito_ML.eq(0)&prose.str.contains(r'cristall|pxrd|xrd',regex=True)))
    out['Label_Contradiction_Flag']=contradiction
    out['Structure_Missing_Flag']=out['Ligand_SMILES'].isna()
    out['Quality_Status']='INCLUDE'
    out.loc[contradiction,'Quality_Status']='REVIEW'
    signature=['Legante','Metallo','Sale_Metallico','Solvente','Additivo_Colinker','Temperatura_C','Tempo_ore','mmol_Legante','mmol_Sale','Rapporto_LM']
    conflicts=out.groupby(signature,dropna=False).Esito_ML.transform('nunique').gt(1)
    out['Condition_Outcome_Conflict_Flag']=conflicts
    out.loc[conflicts,'Quality_Status']='REVIEW'
    out.to_csv(root/'data/knowledge_database_audited_v10_5.csv',index=False)
    summary={
      'records':len(out),'classes':{str(k):int(v) for k,v in out.Esito_ML.value_counts().sort_index().items()},
      'unique_ligands':int(out.Legante.nunique()),'resolved_ligands':int(reg.Ligand_SMILES.notna().sum()),
      'records_with_structure':int(out.Ligand_SMILES.notna().sum()),'label_contradictions':int(contradiction.sum()),
      'condition_outcome_conflicts':int(conflicts.sum()),'review_records':int(out.Quality_Status.eq('REVIEW').sum()),
      'label_policy':{str(k):v for k,v in LABELS.items()},
      'limitations':['PXRD_verified and Source_DOI are absent from the legacy predictor dataset; class 2 therefore remains a legacy label pending source-level recuration.']}
    (root/'reports/dataset_audit_v10_5.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
