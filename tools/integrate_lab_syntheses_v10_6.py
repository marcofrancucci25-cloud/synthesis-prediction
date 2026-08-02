"""Integrate normalized laboratory syntheses without changing the frozen v8 training set."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "data/laboratory_syntheses_normalized_v10_6.csv"


def _signature(r):
    fields = ["Metallo", "Famiglia_Legante", "Counterion_Class", "Solvente",
              "Additivo_Colinker", "Temperatura_C", "Tempo_ore", "Rapporto_LM"]
    vals=[]
    for c in fields:
        v=r.get(c, "")
        if isinstance(v, float) and np.isfinite(v): v=round(v, 4)
        vals.append(str(v).strip().casefold())
    return "|".join(vals)


def build_integrated_database(lab: pd.DataFrame):
    legacy = pd.read_csv(ROOT / "data/knowledge_database.csv")
    legacy["Source_Type"] = "legacy_project_dataset"
    legacy["Source_Record_ID"] = legacy["ID"].astype(str)
    legacy["PXRD_Confirmed"] = np.nan
    legacy["Evidence"] = np.nan
    legacy["Ligand_SMILES"] = np.nan
    legacy["Ligand_InChIKey"] = np.nan
    legacy["Future_Training_Status"] = "FROZEN_V8_RECORD"

    eligible = lab[lab.Training_Status.eq("INCLUDE")].copy()
    mapped = pd.DataFrame({
        "ID": eligible.Lab_Record_ID,
        "Legante": eligible.Legante,
        "Famiglia_Legante": "Bipyrazole",
        "Metallo": eligible.Metallo,
        "Sale_Metallico": eligible.Sale_Metallico,
        "Solvente": eligible.Solvente,
        "Additivo_Colinker": eligible.Additivo_Colinker,
        "Temperatura_C": eligible.Temperatura_C,
        "Tempo_ore": eligible.Tempo_ore,
        "mmol_Legante": eligible.mmol_Legante,
        "mmol_Sale": eligible.mmol_Sale,
        "Rapporto_LM": eligible.Rapporto_LM,
        "Procedura_Sintetica": eligible.Procedura_Sintetica,
        "Esito_ML": eligible.Outcome_code,
        "Volume solvente": eligible.Volume_solvente_mL,
        "Data_Quality_Flag": eligible.Data_Quality_Flag,
        "Exact_Duplicate_Group": np.nan,
        "Outcome_Conflict_Flag": 0,
        "Training_v5_2_Status": "LAB_EVIDENCE_ONLY",
        "Training_v5_2_Reason": "Not used by frozen v8 model",
        "Training_v5_2_Warning": np.nan,
        "Split_ID": "LABORATORY_TEMPORAL_EVIDENCE_v10.6",
        "Split_Set": "LABORATORY_EVIDENCE",
        "Grouping_Variable": "Legante",
        "Split_Seed": np.nan,
        "Ligand_Text": eligible.Legante.str.lower() + " bipyrazole",
        "Hydration_Number": eligible.Hydration_Number,
        "Counterion_Class": eligible.Counterion_Class,
        "Oxidation_State": eligible.Oxidation_State,
        "Metal_Atomic_Number": 30,
        "Metal_Atomic_Weight": 65.38,
        "Metal_Group": 12,
        "Metal_Period": 4,
        "Metal_Block": "d",
        "Metal_Electronegativity": 1.65,
        "Source_Type": "laboratory_experiment",
        "Source_Record_ID": eligible.Sample_ID,
        "PXRD_Confirmed": eligible.PXRD_Confirmed,
        "Evidence": eligible.Evidence,
        "Ligand_SMILES": eligible.Ligand_SMILES,
        "Ligand_InChIKey": eligible.Ligand_InChIKey,
        "Future_Training_Status": "ELIGIBLE_AFTER_NEW_GROUPED_VALIDATION",
        "Condition_Group_ID": eligible.Condition_Group_ID,
        "Microwave_Power_W": eligible.Microwave_Power_W,
        "Microwave_Power_Range": eligible.Microwave_Power_Range,
        "Heating_Ramp_min": eligible.Heating_Ramp_min,
        "Hold_Time_min": eligible.Hold_Time_min,
        "Cooling_Time_min": eligible.Cooling_Time_min,
        "Sonication_Power_W": eligible.Sonication_Power_W,
        "Sonication_Duration_min": eligible.Sonication_Duration_min,
        "Post_Heating_C": eligible.Post_Heating_C,
        "Post_Heating_Time_h": eligible.Post_Heating_Time_h,
    })
    integrated = pd.concat([legacy, mapped], ignore_index=True, sort=False)
    if integrated.ID.astype(str).duplicated().any():
        raise ValueError("Duplicate IDs found in integrated evidence database")
    integrated.to_csv(ROOT / "data/knowledge_database_integrated_v10_6.csv", index=False)
    return integrated, mapped


def build_positive_library(lab: pd.DataFrame):
    base = pd.read_csv(ROOT / "data/successful_synthesis_library_v10_4.csv")
    # v10.4 still contains repeated condition signatures imported from different
    # source tables. Consolidate them without discarding source traceability.
    consolidated=[]
    for signature, grp in base.groupby("Condition_Signature", dropna=False, sort=True):
        ranked=grp.sort_values(["Evidence_Weight","Source_Quality_Score","Completeness_Score"],ascending=False)
        r=ranked.iloc[0].copy()
        r["Merged_Positive_IDs"]=";".join(grp.Positive_ID.astype(str))
        r["Merged_Source_Record_IDs"]=";".join(grp.ID.astype(str))
        r["Source_Record_Count"]=int(len(grp))
        consolidated.append(r)
    base=pd.DataFrame(consolidated).reset_index(drop=True)
    crystalline = lab[lab.Training_Status.eq("INCLUDE") & lab.Outcome_code.eq(2) & lab.PXRD_Confirmed.eq(True)].copy()
    crystalline["Famiglia_Legante"] = "Bipyrazole"
    crystalline["Condition_Signature"] = crystalline.apply(_signature, axis=1)
    # Template library stores one row per normalized condition; repeated batches
    # strengthen evidence but are not allowed to multiply template frequency.
    grouped=[]
    for signature, grp in crystalline.groupby("Condition_Signature", sort=True):
        r=grp.sort_values(["Replicate_Count","Lab_Record_ID"],ascending=[False,True]).iloc[0].copy()
        r["Lab_Source_Record_IDs"]=";".join(grp.Lab_Record_ID.astype(str))
        r["Lab_Replicate_Count"]=int(grp.Replicate_Count.sum())
        r["Condition_Signature"]=signature
        grouped.append(r)
    new = pd.DataFrame(grouped)
    existing = set(base.Condition_Signature.astype(str))
    new = new[~new.Condition_Signature.astype(str).isin(existing)].reset_index(drop=True)
    start=695
    appended=pd.DataFrame({
        "Positive_ID":[f"POS-{i:04d}" for i in range(start,start+len(new))],
        "ID":new.Lab_Record_ID,
        "Legante":new.Legante,
        "Famiglia_Legante":new.Famiglia_Legante,
        "Metallo":new.Metallo,
        "Sale_Metallico":new.Sale_Metallico,
        "Solvente":new.Solvente,
        "Additivo_Colinker":new.Additivo_Colinker,
        "Temperatura_C":new.Temperatura_C,
        "Tempo_ore":new.Tempo_ore,
        "mmol_Legante":new.mmol_Legante,
        "mmol_Sale":new.mmol_Sale,
        "Rapporto_LM":new.Rapporto_LM,
        "Volume solvente":new.Volume_solvente_mL,
        "Hydration_Number":new.Hydration_Number,
        "Counterion_Class":new.Counterion_Class,
        "Oxidation_State":new.Oxidation_State,
        "Procedura_Sintetica":new.Procedura_Sintetica,
        "Esito_ML":2,
        "Positive_Library_Source":"laboratory_pxrd_positive",
        "Split_Set":"LABORATORY_EVIDENCE",
        "Critical_Completeness":1.0,
        "Extended_Completeness":1.0,
        "Completeness_Score":1.0,
        "Source_Quality_Score":0.95,
        "Diversity_Weight":1.0,
        "Evidence_Weight":np.minimum(0.99,0.90+0.015*np.maximum(new.Lab_Replicate_Count.astype(float)-1,0)),
        "Quality_Tier":"A",
        "PXRD_Confirmed":True,
        "Condition_Signature":new.Condition_Signature,
        "Source_DOI":np.nan,
        "Extraction_Method":"direct_laboratory_record",
        "Lab_Replicate_Count":new.Lab_Replicate_Count,
        "Lab_Source_Record_IDs":new.Lab_Source_Record_IDs,
    })
    for c in appended.columns:
        if c not in base: base[c]=np.nan
    for c in base.columns:
        if c not in appended: appended[c]=np.nan
    combined=pd.concat([base,appended[base.columns]],ignore_index=True)
    if appended.Condition_Signature.astype(str).duplicated().any() or set(appended.Condition_Signature.astype(str)) & set(base.Condition_Signature.astype(str)):
        raise ValueError("Duplicate condition signatures in positive library")
    combined.to_csv(ROOT / "data/successful_synthesis_library_v10_6.csv",index=False)
    return combined,appended


def main():
    lab=pd.read_csv(LAB_PATH)
    integrated,mapped=build_integrated_database(lab)
    positive,appended=build_positive_library(lab)
    special=lab[lab.Training_Status.eq("EXCLUDE_SPECIAL")]
    special.to_csv(ROOT / "data/laboratory_in_situ_drug_loading_v10_6.csv",index=False)
    review=lab[lab.Training_Status.eq("REVIEW")]
    review.to_csv(ROOT / "data/laboratory_records_needing_review_v10_6.csv",index=False)
    summary={
      "version":"10.6.0","source_experiments_unique":int(len(lab)),
      "integrated_evidence_records":int(len(mapped)),"integrated_database_total":int(len(integrated)),
      "class_distribution_integrated":{str(k):int(v) for k,v in mapped.Esito_ML.value_counts().sort_index().items()},
      "crystalline_source_records":int(((lab.Training_Status.eq('INCLUDE')) & lab.Outcome_code.eq(2)).sum()),
      "positive_condition_templates_added":int(len(appended)),"positive_library_total":int(len(positive)),
      "special_drug_loading_records":int(len(special)),"records_needing_review":int(len(review)),
      "frozen_predictor_retrained":False,
      "scientific_boundary":"Laboratory records support evidence retrieval and positive-template optimization; v8 metrics remain tied to the frozen training set."
    }
    (ROOT / "reports/laboratory_integration_summary_v10_6.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
