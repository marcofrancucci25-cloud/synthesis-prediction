#!/usr/bin/env python3
"""Train and validate a structural ligand model with ligand/scaffold grouping."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.structural import LigandStructureTransformer

NUM=['Temperatura_C','Tempo_ore','mmol_Legante','mmol_Sale','Rapporto_LM','Volume solvente','Hydration_Number','Oxidation_State','Metal_Atomic_Number','Metal_Atomic_Weight','Metal_Group','Metal_Period','Metal_Electronegativity']
CAT=['Famiglia_Legante','Metallo','Sale_Metallico','Counterion_Class','Metal_Block','Solvente','Additivo_Colinker']

def build(seed=260802):
    prep=ColumnTransformer([
      ('structure',LigandStructureTransformer(),['Ligand_SMILES']),
      ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),CAT),
    ])
    return Pipeline([('prep',prep),('model',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=-1,max_features='sqrt'))])

def evaluate(X,y,groups,label):
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=260802)
    rows=[]; pred=np.full(len(y),-1)
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups),1):
        model=build(260802+fold); model.fit(X.iloc[tr],y[tr]); p=model.predict(X.iloc[te]); pred[te]=p
        rows.append({'validation':label,'fold':fold,'n_test':len(te),'macro_f1':f1_score(y[te],p,average='macro'),'balanced_accuracy':balanced_accuracy_score(y[te],p),'mcc':matthews_corrcoef(y[te],p)})
    return rows,pred

def main():
    root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/'data/knowledge_database_audited_v10_5.csv')
    df=df[df.Quality_Status.eq('INCLUDE') & df.Ligand_SMILES.notna()].reset_index(drop=True)
    y=df.Esito_ML.astype(int).to_numpy(); X=df[['Ligand_SMILES']+NUM+CAT]
    rows=[]
    r,_=evaluate(X,y,df.Legante.astype(str).to_numpy(),'unseen_ligand'); rows+=r
    valid_scaffolds=df.Ligand_Scaffold.fillna(df.Ligand_SMILES).astype(str).to_numpy()
    r,_=evaluate(X,y,valid_scaffolds,'unseen_scaffold'); rows+=r
    metrics=pd.DataFrame(rows); metrics.to_csv(root/'reports/structural_grouped_cv_v10_5.csv',index=False)
    summary=metrics.groupby('validation')[['macro_f1','balanced_accuracy','mcc']].agg(['mean','std']).round(4)
    model=build(); model.fit(X,y)
    artifact={'version':'10.5.0','model':model,'features':['Ligand_SMILES']+NUM+CAT,'classes':[0,1,2],
              'training_records':len(df),'training_ligands':int(df.Legante.nunique()),'training_scaffolds':int(df.Ligand_Scaffold.nunique()),
              'validation_summary':json.loads(summary.to_json())}
    joblib.dump(artifact,root/'models/MOF_Structural_Predictor_v10_5.joblib',compress=3)
    (root/'reports/structural_validation_summary_v10_5.json').write_text(json.dumps({
      'training_records':len(df),'training_ligands':int(df.Legante.nunique()),'training_scaffolds':int(df.Ligand_Scaffold.nunique()),
      'metrics':json.loads(summary.to_json()),'warning':'This is grouped cross-validation on the structure-resolved legacy subset, not a replacement for the frozen external test.'},indent=2))
    print(summary.to_string())

if __name__=='__main__': main()
