"""Build the provenance-first v11 curation dataset without retraining production.

The builder enforces article-level leakage grouping.  Every protocol from a DOI
already present in the locked literature challenge remains in the benchmark;
it is never allowed to enter the training-candidate partition.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.I)

OUTPUT_COLUMNS = [
    "Record_ID", "Partition_Role", "Training_Eligibility", "Leakage_Group",
    "Metal_Ligand_Group", "Source_Type", "Source_Record_ID", "Source_DOI",
    "Source_Data_DOI", "Source_Data_URL", "Source_URL", "Article_Title", "MOF", "Legante", "Ligand_SMILES",
    "Ligand_InChIKey", "Famiglia_Legante", "Metallo", "Sale_Metallico",
    "Counterion_Class", "Hydration_Number", "Oxidation_State", "Solvente",
    "Solvent_1", "Solvent_1_mL", "Solvent_2", "Solvent_2_mL",
    "Solvent_3", "Solvent_3_mL", "Solvent_4", "Solvent_4_mL",
    "Solvent_5", "Solvent_5_mL", "Additivo_Colinker", "mmol_Additive",
    "Temperatura_C", "Tempo_ore", "mmol_Legante", "mmol_Sale",
    "Rapporto_LM", "Volume_solvente_mL",
    "Total_Precursor_Concentration_M", "Heating_Method", "Microwave_Power_W", "Workup",
    "Outcome_Class", "Outcome_Label", "Outcome_Raw_Score", "Outcome_Score_Scale",
    "Outcome_Mapping_Rule", "Curation_Policy", "Yield_percent", "BET_m2_g",
    "PXRD_Confirmed", "SCXRD_Confirmed",
    "Evidence_Type", "Evidence_Statement", "Extraction_Method",
    "Extraction_Note", "Protocol_Completeness", "Precursor_Qualification",
    "Experiment_Generation", "Condition_Signature", "Replicate_Group_Size",
    "Source_Group_Size", "Source_Group_Cap_Factor", "Evidence_Quality_Weight",
    "Training_Weight",
    "Curation_Version",
]


def _clean(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def _doi(value):
    doi = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            doi = doi[len(prefix):].strip()
    if doi and not DOI_PATTERN.fullmatch(doi):
        raise ValueError(f"Invalid DOI syntax: {doi}")
    return doi


def _signature(row):
    fields = [
        "Legante", "Metallo", "Sale_Metallico", "Solvente", "Solvent_1",
        "Solvent_1_mL", "Solvent_2", "Solvent_2_mL", "Solvent_3",
        "Solvent_3_mL", "Solvent_4", "Solvent_4_mL", "Solvent_5",
        "Solvent_5_mL", "Additivo_Colinker", "mmol_Additive",
        "Temperatura_C", "Tempo_ore", "mmol_Legante", "mmol_Sale",
        "Rapporto_LM", "Volume_solvente_mL", "Heating_Method", "Microwave_Power_W",
    ]
    payload = "|".join(_clean(row.get(c)) for c in fields)
    return "COND-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _concentration(row):
    try:
        total = float(row.get("mmol_Legante")) + float(row.get("mmol_Sale"))
        volume = float(row.get("Volume_solvente_mL"))
        return total / volume if volume > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def _metal_ligand_group(row):
    return f"PAIR::{_clean(row.get('Metallo'))}::{_clean(row.get('Legante'))}"


def _finalize(frame):
    frame = frame.copy()
    frame["Source_DOI"] = frame["Source_DOI"].fillna("").map(_doi)
    if "Source_Data_DOI" not in frame:
        frame["Source_Data_DOI"] = ""
    frame["Source_Data_DOI"] = frame["Source_Data_DOI"].fillna("").map(_doi)
    frame["Source_URL"] = frame["Source_DOI"].map(
        lambda doi: f"https://doi.org/{doi}" if doi else ""
    )
    frame["Source_Data_URL"] = frame["Source_Data_DOI"].map(
        lambda doi: f"https://doi.org/{doi}" if doi else ""
    )
    frame["Metal_Ligand_Group"] = frame.apply(_metal_ligand_group, axis=1)
    frame["Condition_Signature"] = frame.apply(_signature, axis=1)
    frame["Replicate_Group_Size"] = frame.groupby("Condition_Signature")[
        "Condition_Signature"
    ].transform("size").astype(int)
    # Retain repeated batches as evidence while capping exact-condition
    # influence at one condition-equivalent.  A single literature article can
    # contribute at most 20 condition-equivalents, preventing a large robotic
    # campaign from overwhelming independent sources during future fitting.
    eligible = frame["Training_Eligibility"].astype(bool)
    literature_eligible = eligible & frame["Source_DOI"].ne("")
    source_sizes = frame.loc[literature_eligible, "Source_DOI"].value_counts()
    frame["Source_Group_Size"] = 1
    frame.loc[literature_eligible, "Source_Group_Size"] = (
        frame.loc[literature_eligible, "Source_DOI"].map(source_sizes).astype(int)
    )
    frame["Source_Group_Cap_Factor"] = 1.0
    frame.loc[literature_eligible, "Source_Group_Cap_Factor"] = (
        20.0 / frame.loc[literature_eligible, "Source_Group_Size"]
    ).clip(upper=1.0)
    if "Evidence_Quality_Weight" not in frame:
        frame["Evidence_Quality_Weight"] = 1.0
    frame["Evidence_Quality_Weight"] = pd.to_numeric(
        frame["Evidence_Quality_Weight"], errors="coerce"
    ).fillna(1.0).clip(lower=0.0, upper=1.0)
    frame["Training_Weight"] = np.where(
        eligible,
        frame["Evidence_Quality_Weight"]
        * frame["Source_Group_Cap_Factor"]
        / frame["Replicate_Group_Size"],
        0.0,
    )
    frame["Total_Precursor_Concentration_M"] = frame.apply(_concentration, axis=1)
    frame["Curation_Version"] = "11.0-foundation-tranche-03"
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[OUTPUT_COLUMNS]
    if frame["Record_ID"].astype(str).duplicated().any():
        raise ValueError("Duplicate v11 Record_ID")
    return frame


def literature_benchmark():
    src = pd.read_csv(DATA / "literature_crystalline_challenge_v10_6.csv")
    out = pd.DataFrame({
        "Record_ID": src.Case_ID,
        "Partition_Role": "LOCKED_EXTERNAL_BENCHMARK",
        "Training_Eligibility": False,
        "Leakage_Group": src.Source_DOI.map(lambda x: f"DOI::{_doi(x).casefold()}"),
        "Source_Type": "peer_reviewed_literature",
        "Source_Record_ID": src.Case_ID,
        "Source_DOI": src.Source_DOI,
        "Article_Title": "",
        "MOF": src.MOF,
        "Legante": src.Legante,
        "Ligand_SMILES": "",
        "Ligand_InChIKey": "",
        "Famiglia_Legante": src.Famiglia_Legante,
        "Metallo": src.Metallo,
        "Sale_Metallico": src.Sale_Metallico,
        "Counterion_Class": src.Counterion_Class,
        "Hydration_Number": src.Hydration_Number,
        "Oxidation_State": src.Oxidation_State,
        "Solvente": src.Solvente,
        "Additivo_Colinker": src.Additivo_Colinker,
        "Temperatura_C": src.Temperatura_C,
        "Tempo_ore": src.Tempo_ore,
        "mmol_Legante": src.mmol_Legante,
        "mmol_Sale": src.mmol_Sale,
        "Rapporto_LM": src.Rapporto_LM,
        "Volume_solvente_mL": src["Volume solvente"],
        "Outcome_Class": src.Expected_Class,
        "Outcome_Label": "Crystalline MOF",
        "PXRD_Confirmed": True,
        "SCXRD_Confirmed": False,
        "Evidence_Type": "PXRD/XRD",
        "Evidence_Statement": src.Crystallinity_Evidence,
        "Extraction_Method": "manual_primary_article",
        "Extraction_Note": src.Encoding_Note,
        "Protocol_Completeness": "CORE_COMPLETE",
        "Precursor_Qualification": "See locked benchmark encoding note",
    })
    return out


def laboratory_candidates():
    src = pd.read_csv(DATA / "laboratory_syntheses_normalized_v10_6.csv")
    src = src[src.Training_Status.eq("INCLUDE")].copy()
    out = pd.DataFrame({
        "Record_ID": src.Lab_Record_ID,
        "Partition_Role": "TRAINING_CANDIDATE",
        "Training_Eligibility": True,
        # Ligand-level grouping prevents near-identical laboratory conditions
        # from appearing on both sides of a future validation split.
        "Leakage_Group": src.apply(
            lambda r: "LIGAND::" + _clean(r.Ligand_InChIKey or r.Legante), axis=1
        ),
        "Source_Type": "direct_laboratory_experiment",
        "Source_Record_ID": src.Sample_ID,
        "Source_DOI": "",
        "Article_Title": "",
        "MOF": "Laboratory product",
        "Legante": src.Legante,
        "Ligand_SMILES": src.Ligand_SMILES,
        "Ligand_InChIKey": src.Ligand_InChIKey,
        "Famiglia_Legante": "Bipyrazole/pyrazole",
        "Metallo": src.Metallo,
        "Sale_Metallico": src.Sale_Metallico,
        "Counterion_Class": src.Counterion_Class,
        "Hydration_Number": src.Hydration_Number,
        "Oxidation_State": src.Oxidation_State,
        "Solvente": src.Solvente,
        "Solvent_1": src.Solvent_1,
        "Solvent_1_mL": src.Solvent_1_mL,
        "Solvent_2": src.Solvent_2,
        "Solvent_2_mL": src.Solvent_2_mL,
        "Additivo_Colinker": src.Additivo_Colinker,
        "Temperatura_C": src.Temperatura_C,
        "Tempo_ore": src.Tempo_ore,
        "mmol_Legante": src.mmol_Legante,
        "mmol_Sale": src.mmol_Sale,
        "Rapporto_LM": src.Rapporto_LM,
        "Volume_solvente_mL": src.Volume_solvente_mL,
        "Heating_Method": src.Procedura_Sintetica,
        "Workup": "See direct laboratory notebook record",
        "Outcome_Class": src.Outcome_code,
        "Outcome_Label": src.Outcome_label,
        "PXRD_Confirmed": src.PXRD_Confirmed,
        "SCXRD_Confirmed": False,
        "Evidence_Type": src.PXRD_Confirmed.map({True: "PXRD", False: "Direct experimental outcome"}),
        "Evidence_Statement": src.Evidence,
        "Extraction_Method": "direct_laboratory_record",
        "Extraction_Note": src.Data_Quality_Flag,
        "Protocol_Completeness": "FULL_PROTOCOL",
        "Precursor_Qualification": "Directly recorded precursor",
    })
    return out


def expansion_tranche(locked_dois):
    files = sorted(DATA.glob("v11_literature_expansion_tranche_*.csv"))
    if not files:
        raise FileNotFoundError("No v11 literature expansion tranches found")
    src = pd.concat((pd.read_csv(path) for path in files), ignore_index=True, sort=False)
    src["Source_DOI"] = src.Source_DOI.map(_doi)
    is_locked = src.Source_DOI.str.casefold().isin(locked_dois)
    out = src.rename(columns={"Record_ID": "Record_ID"}).copy()
    out["Partition_Role"] = np.where(
        is_locked, "LOCKED_EXTERNAL_BENCHMARK_EXPANSION", "TRAINING_CANDIDATE"
    )
    out["Training_Eligibility"] = ~is_locked
    out["Leakage_Group"] = out.Source_DOI.map(lambda x: f"DOI::{x.casefold()}")
    out["Source_Type"] = "peer_reviewed_literature"
    out["Source_Record_ID"] = out.Record_ID
    return out


def readiness_summary(gold):
    train = gold[gold.Training_Eligibility.eq(True)].copy()
    distribution = train.Outcome_Class.value_counts().sort_index().to_dict()
    effective_distribution = train.groupby("Outcome_Class").Training_Weight.sum().to_dict()
    minimum_class = min((int(distribution.get(i, 0)) for i in (0, 1, 2)), default=0)
    doi_groups = train.loc[train.Source_DOI.ne(""), "Leakage_Group"].nunique()
    pair_groups = train.Metal_Ligand_Group.nunique()
    families = train.Famiglia_Legante.nunique()
    gates = {
        "minimum_30_records_per_class": minimum_class >= 30,
        "minimum_20_independent_literature_doi_groups": doi_groups >= 20,
        "minimum_10_metal_ligand_groups": pair_groups >= 10,
        "minimum_5_ligand_families": families >= 5,
        "no_doi_overlap_between_training_and_benchmark": not bool(
            set(train.Source_DOI) - {""} & set(gold.loc[~gold.Training_Eligibility, "Source_DOI"]) - {""}
        ),
    }
    return {
        "version": "11.0-foundation-tranche-03",
        "gold_records_total": int(len(gold)),
        "training_candidates": int(len(train)),
        "locked_external_records": int((~gold.Training_Eligibility).sum()),
        "training_class_distribution": {str(k): int(v) for k, v in distribution.items()},
        "training_effective_class_distribution": {
            str(k): round(float(v), 3) for k, v in effective_distribution.items()
        },
        "training_independent_doi_groups": int(doi_groups),
        "training_metal_ligand_groups": int(pair_groups),
        "training_ligand_families": int(families),
        "readiness_gates": gates,
        "promotion_gate_passed": all(gates.values()),
        "production_model_retrained": False,
        "decision": "Continue curation; class balance and independent-source coverage are insufficient for a defensible v11 retraining.",
    }


def build():
    benchmark = literature_benchmark()
    locked_dois = set(benchmark.Source_DOI.map(_doi).str.casefold())
    expansion = expansion_tranche(locked_dois)
    lab = laboratory_candidates()
    gold = _finalize(pd.concat([benchmark, expansion, lab], ignore_index=True, sort=False))
    # Article-level leakage is forbidden even when precursor conditions differ.
    train_dois = set(gold.loc[gold.Training_Eligibility, "Source_DOI"]) - {""}
    test_dois = set(gold.loc[~gold.Training_Eligibility, "Source_DOI"]) - {""}
    if train_dois & test_dois:
        raise ValueError(f"DOI leakage detected: {sorted(train_dois & test_dois)}")
    gold.to_csv(DATA / "v11_gold_synthesis_records.csv", index=False)
    gold[gold.Training_Eligibility].to_csv(DATA / "v11_training_candidates.csv", index=False)
    gold[~gold.Training_Eligibility].to_csv(DATA / "v11_locked_external_benchmark.csv", index=False)
    summary = readiness_summary(gold)
    (REPORTS / "v11_dataset_readiness.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return gold, summary


def main():
    _, summary = build()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
