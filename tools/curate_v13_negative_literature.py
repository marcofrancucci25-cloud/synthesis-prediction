#!/usr/bin/env python3
"""Build the v10.13 DOI-linked negative/amorphous evidence tranche.

The machine-readable campaign key is preserved verbatim.  This script only
maps outcomes explicitly assigned by the authors; it never treats an empty
field, an unmatched pattern, or absence of discussion as a failed synthesis.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "v13_source_rare_earth_landscape.csv"
OUTPUT = ROOT / "data" / "v13_negative_amorphous_literature.csv"
REVIEW = ROOT / "data" / "v13_literature_records_needing_review.csv"

ARTICLE_DOI = "10.1039/d5sc09992g"
DATA_DOI = "10.5281/zenodo.17902549"
ARTICLE_TITLE = (
    "Mapping the crystallization landscape of rare earth MOFs: a "
    "high-throughput investigation of structure, kinetics, and selectivity"
)

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

LINKERS = {
    "BDC": ("terephthalic acid (1,4-BDC)", "Aromatic dicarboxylate"),
    "Phthalic": ("phthalic acid (1,2-BDC)", "Aromatic dicarboxylate"),
    "Isophthalic": ("isophthalic acid (1,3-BDC)", "Aromatic dicarboxylate"),
    "BTC": ("benzene-1,3,5-tricarboxylic acid (H3BTC)", "Aromatic tricarboxylate"),
    "Trimellitic": ("trimellitic acid", "Aromatic tricarboxylate"),
    "Pyromellitic": ("pyromellitic acid", "Aromatic tetracarboxylate"),
    "Mellitic": ("mellitic acid", "Aromatic hexacarboxylate"),
}


def _base_record(row: pd.Series) -> dict:
    ligand, family = LINKERS[row["Linker"]]
    is_binary = "+" in str(row["Metal"])
    reaction_time = 1.0 if is_binary else 18.0
    acid_m = float(row["AA Conc"])
    return {
        "Record_ID": f"V13-REHT-{int(row['Sample#']):04d}",
        "Source_Record_ID": row["SampleID"],
        "Source_DOI": ARTICLE_DOI,
        "Source_Data_DOI": DATA_DOI,
        "Source_URL": f"https://doi.org/{ARTICLE_DOI}",
        "Source_Data_URL": f"https://doi.org/{DATA_DOI}",
        "Article_Title": ARTICLE_TITLE,
        "Metal_Ligand_Group": f"PAIR::{str(row['Metal']).casefold()}::{ligand.casefold()}",
        "MOF": np.nan,
        "Legante": ligand,
        "Ligand_SMILES": np.nan,
        "Ligand_InChIKey": np.nan,
        "Famiglia_Legante": family,
        "Metallo": row["Metal"],
        "Sale_Metallico": "rare-earth nitrate salt(s); hydration not encoded in campaign key",
        "Counterion_Class": "nitrate",
        "Hydration_Number": np.nan,
        "Oxidation_State": 3,
        "Solvente": "EtOH/H2O (fractions not encoded in campaign key)",
        "Solvent_Fractions": np.nan,
        "Additivo_Colinker": "Nessuno" if acid_m == 0 else "Acido acetico",
        "Acetic_Acid_Concentration_M": acid_m,
        "Temperatura_C": float(row["Temp"]),
        "Tempo_ore": reaction_time,
        "mmol_Legante": np.nan,
        "mmol_Sale": np.nan,
        "Rapporto_LM": np.nan,
        "Volume_solvente_mL": np.nan,
        "Total_Precursor_Concentration_M": np.nan,
        "Synthetic_Method": "Automated solvothermal synthesis",
        "Vessel_Type": "Automated vial; exact geometry in source SI",
        "First_Heterogeneous_Phase_h": row["first_hetero_h"],
        "First_Solid_h": row["first_solid_h"],
        "First_Heterogeneous_or_Solid_h": row["first_hetero_or_solid_h"],
        "Primary_Phase_Raw": row["Primary Phase"],
        "Secondary_Phase_Raw": row["Secondary Phase"],
        "Source_Type": "peer_reviewed_high_throughput_literature",
        "Evidence_Type": "Author-assigned HT-PXRD/CV campaign outcome",
        "Extraction_Method": "direct_machine_readable_campaign_key",
        "Supporting_File": "v13_source_rare_earth_landscape.csv",
        "Independent_Review_Status": "SOURCE_VERIFIED_SINGLE_CURATION_PASS",
        "Training_Eligibility": False,
        "Exclusion_Reason": (
            "Evidence-only: absolute precursor amounts, ligand:metal ratio, and "
            "solvent volume are not encoded in the public campaign key."
        ),
    }


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    expected = {
        "Sample#", "SampleID", "Metal", "Linker", "AA Conc", "Temp",
        "first_hetero_h", "first_solid_h", "first_hetero_or_solid_h",
        "Primary Phase", "Secondary Phase",
    }
    missing = expected.difference(source.columns)
    if missing:
        raise ValueError(f"Campaign key is missing columns: {sorted(missing)}")
    if source["Sample#"].duplicated().any() or source["SampleID"].duplicated().any():
        raise ValueError("Campaign key contains duplicate sample identifiers")

    records = []
    review_records = []
    for _, row in source.iterrows():
        phase = str(row["Primary Phase"]).strip().casefold()
        rec = _base_record(row)
        if phase == "no solid product":
            rec.update({
                "Outcome_Class": 0,
                "Outcome_Label": "Failed / no solid product",
                "Outcome_Raw_Text": "no solid product",
                "Outcome_Mapping_Rule": "Exact author label 'no solid product' -> class 0",
                "PXRD_Confirmed": False,
                "SCXRD_Confirmed": False,
                "Target_Phase_Match": False,
                "Evidence_Statement": "Campaign key explicitly reports no solid product.",
            })
            records.append(rec)
        elif phase == "no crystalline product":
            rec.update({
                "Outcome_Class": 1,
                "Outcome_Label": "Amorphous / no crystalline product",
                "Outcome_Raw_Text": "no crystalline product",
                "Outcome_Mapping_Rule": "Exact author label 'no crystalline product' -> class 1",
                "PXRD_Confirmed": True,
                "SCXRD_Confirmed": False,
                "Target_Phase_Match": False,
                "Evidence_Statement": "HT-PXRD campaign key explicitly reports no crystalline product.",
            })
            records.append(rec)
        elif phase == "unknown phase":
            rec.update({
                "Outcome_Class": np.nan,
                "Outcome_Label": "Unknown crystalline phase — review",
                "Outcome_Raw_Text": row["Primary Phase"],
                "Outcome_Mapping_Rule": "No class assigned: unknown phase is not equivalent to failure or amorphous product",
                "PXRD_Confirmed": True,
                "SCXRD_Confirmed": False,
                "Target_Phase_Match": np.nan,
                "Evidence_Statement": "Author-assigned unknown phase; retained outside negative labels.",
            })
            review_records.append(rec)

    # Direct, room-temperature synthesis of an amorphous ZIF-like product.
    records.append({
        "Record_ID": "V13-AMOF-WU-001",
        "Source_Record_ID": "aZIF-8 direct aqueous synthesis",
        "Source_DOI": "10.1038/s41467-019-13153-x",
        "Source_Data_DOI": np.nan,
        "Source_URL": "https://doi.org/10.1038/s41467-019-13153-x",
        "Source_Data_URL": np.nan,
        "Article_Title": "Packaging and delivering enzymes by amorphous metal-organic frameworks",
        "Metal_Ligand_Group": "PAIR::zn::2-methylimidazole",
        "MOF": "amorphous ZIF (aZIF)",
        "Legante": "2-methylimidazole",
        "Ligand_SMILES": "Cc1ncc[nH]1",
        "Ligand_InChIKey": "LXBGSDVWAMZHDD-UHFFFAOYSA-N",
        "Famiglia_Legante": "Imidazolate",
        "Metallo": "Zn",
        "Sale_Metallico": "zinc acetate",
        "Counterion_Class": "acetate",
        "Hydration_Number": np.nan,
        "Oxidation_State": 2,
        "Solvente": "H2O",
        "Solvent_Fractions": "H2O=1.0",
        "Additivo_Colinker": "glucose oxidase (GOx)",
        "Acetic_Acid_Concentration_M": np.nan,
        "Temperatura_C": 25.0,
        "Tempo_ore": 0.5,
        "mmol_Legante": np.nan,
        "mmol_Sale": np.nan,
        "Rapporto_LM": 4.0,
        "Volume_solvente_mL": np.nan,
        "Total_Precursor_Concentration_M": 0.05,
        "Synthetic_Method": "One-pot aqueous co-precipitation under stirring",
        "Vessel_Type": "Open ambient vessel; exact geometry not reported",
        "First_Heterogeneous_Phase_h": np.nan,
        "First_Solid_h": np.nan,
        "First_Heterogeneous_or_Solid_h": np.nan,
        "Primary_Phase_Raw": "amorphous structure",
        "Secondary_Phase_Raw": np.nan,
        "Outcome_Class": 1,
        "Outcome_Label": "Amorphous MOF",
        "Outcome_Raw_Text": "XRD patterns ... implied the existence of amorphous structures",
        "Outcome_Mapping_Rule": "Explicit XRD-supported amorphous assignment -> class 1",
        "PXRD_Confirmed": True,
        "SCXRD_Confirmed": False,
        "Target_Phase_Match": False,
        "Source_Type": "peer_reviewed_primary_literature",
        "Evidence_Type": "XRD and SAED",
        "Evidence_Statement": (
            "2-MeIM 40 mM, zinc acetate 10 mM and GOx 0.25 mg/mL were stirred "
            "in water at room temperature for 30 min; XRD/SAED supported an amorphous product."
        ),
        "Extraction_Method": "manual_primary_article",
        "Supporting_File": "main article methods and results",
        "Independent_Review_Status": "SOURCE_VERIFIED_SINGLE_CURATION_PASS",
        "Training_Eligibility": False,
        "Exclusion_Reason": "Evidence-only enzyme/MOF composite; reaction volume and salt hydration are not stated.",
    })

    negative = pd.DataFrame(records)
    review = pd.DataFrame(review_records)
    if not negative["Source_DOI"].map(lambda x: bool(DOI_RE.match(str(x)))).all():
        raise ValueError("Invalid or missing DOI in negative/amorphous tranche")
    if not negative["Outcome_Class"].isin([0, 1]).all():
        raise ValueError("Negative/amorphous tranche contains an unsupported class")
    if negative["Record_ID"].duplicated().any():
        raise ValueError("Duplicate curated record IDs")
    return negative, review


def main() -> None:
    negative, review = build()
    negative.to_csv(OUTPUT, index=False)
    review.to_csv(REVIEW, index=False)
    print(
        f"Wrote {len(negative)} negative/amorphous records "
        f"({negative['Source_DOI'].nunique()} article DOIs); "
        f"held {len(review)} unknown-phase records for review."
    )


if __name__ == "__main__":
    main()
