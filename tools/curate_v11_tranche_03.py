"""Build tranche 03 from two primary high-throughput MOF studies.

Sources
-------
1. Supplementary Data 1 for DOI 10.1038/s41467-019-08483-9 contains
   90 Cu-HKUST-1 robotic syntheses and a normalized PXRD crystallinity /
   phase-purity score.  The source score is retained verbatim.  The three-class
   mapping below is a preregistered curation rule, not an author-provided label:
   <=0.30 very-low target crystallinity, 0.35--0.75 partial/poor crystallinity,
   >=0.80 high target crystallinity.
2. Tables 1 and 2 of DOI 10.1021/acscentsci.3c01087 explicitly identify ten
   representative high-crystallinity MOF-321/MOF-322 protocols.

The script never uses BET to assign an outcome class; a zero in the published
BET column may indicate that porosity was not measured and is therefore kept
as a separate raw field only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "v11_literature_expansion_tranche_03.csv"

HKUST_DOI = "10.1038/s41467-019-08483-9"
HKUST_DATA_DOI = "10.24435/materialscloud:2018.0011/v3"
HKUST_TITLE = "Capturing chemical intuition in synthesis of metal-organic frameworks"
HKUST_POLICY = "V11-PXRD-NORMALIZED-0.30-0.80-v1"

AI_LAB_DOI = "10.1021/acscentsci.3c01087"
AI_LAB_TITLE = "ChatGPT Research Group for Optimizing the Crystallinity of MOFs and COFs"


def _hkust_outcome(score: float) -> tuple[int, str]:
    if score <= 0.30:
        return 0, "Very low target crystallinity / failed target synthesis"
    if score < 0.80:
        return 1, "Partially successful / poor target crystallinity"
    return 2, "High-crystallinity target MOF"


def _solvent_summary(row: pd.Series) -> str:
    labels = (("H2O", row.H2O), ("DMF", row.DMF), ("EtOH", row.EeOH),
              ("MeOH", row.MeOH), ("iPrOH", row.iPrOH))
    return " / ".join(f"{name} {float(volume):g} mL" for name, volume in labels if volume > 0)


def hkust_records() -> pd.DataFrame:
    src = pd.read_csv(DATA / "v11_source_hkust1_moesm5.csv", comment="#")
    rows = []
    for row in src.itertuples(index=False):
        outcome_class, outcome_label = _hkust_outcome(float(row.Crystallinity))
        total_volume = float(row.H2O + row.DMF + row.EeOH + row.MeOH + row.iPrOH)
        generation = int(str(row.sample).split("-")[0][1:])
        rows.append({
            "Record_ID": f"V11-HKUST-{row.sample}",
            "MOF": "HKUST-1 / Cu-BTC",
            "Legante": "benzene-1,3,5-tricarboxylic acid (H3BTC)",
            "Ligand_SMILES": "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1",
            "Ligand_InChIKey": "",
            "Famiglia_Legante": "Aromatic tricarboxylate",
            "Metallo": "Cu",
            "Sale_Metallico": "Cu nitrate (hydration not stated in Supplementary Data 1)",
            "Counterion_Class": "nitrate",
            "Hydration_Number": np.nan,
            "Oxidation_State": 2,
            "Solvente": _solvent_summary(pd.Series(row._asdict())),
            "Solvent_1": "H2O", "Solvent_1_mL": row.H2O,
            "Solvent_2": "DMF", "Solvent_2_mL": row.DMF,
            "Solvent_3": "EtOH", "Solvent_3_mL": row.EeOH,
            "Solvent_4": "MeOH", "Solvent_4_mL": row.MeOH,
            "Solvent_5": "iPrOH", "Solvent_5_mL": row.iPrOH,
            "Additivo_Colinker": "Nessuno",
            "mmol_Additive": np.nan,
            "Temperatura_C": row.Temp,
            "Tempo_ore": row.Time / 60.0,
            "mmol_Legante": np.nan,
            "mmol_Sale": np.nan,
            # The published ReaRatio is metal salt / BTC; the app schema uses L/M.
            "Rapporto_LM": 1.0 / row.ReaRatio,
            "Volume_solvente_mL": total_volume,
            "Heating_Method": "Automated microwave synthesis",
            "Microwave_Power_W": row.Power,
            "Workup": "Not encoded in the published Supplementary Data 1 table.",
            "Outcome_Class": outcome_class,
            "Outcome_Label": outcome_label,
            "Outcome_Raw_Score": row.Crystallinity,
            "Outcome_Score_Scale": "Normalized PXRD crystallinity and phase-purity fitness (0-1)",
            "Outcome_Mapping_Rule": (
                "Preregistered curation thresholds: <=0.30 class 0; 0.35-0.75 class 1; "
                ">=0.80 class 2. The continuous source score is retained."
            ),
            "Curation_Policy": HKUST_POLICY,
            "Yield_percent": np.nan,
            "BET_m2_g": row.BET,
            "PXRD_Confirmed": outcome_class == 2,
            "SCXRD_Confirmed": False,
            "Evidence_Type": "PXRD FWHM plus phase-purity fitness",
            "Evidence_Statement": (
                f"Published Supplementary Data 1 record {row.sample}; normalized "
                f"crystallinity/phase-purity score {row.Crystallinity:g}."
            ),
            "Source_DOI": HKUST_DOI,
            "Source_Data_DOI": HKUST_DATA_DOI,
            "Article_Title": HKUST_TITLE,
            "Extraction_Method": "direct_machine_readable_supplementary_csv",
            "Extraction_Note": (
                "All nine experimental variables and the source outcome score were copied "
                "from the authors' CSV; absolute precursor amounts were not invented."
            ),
            "Protocol_Completeness": "COMPLETE_CONDITIONS_RELATIVE_STOICHIOMETRY",
            "Precursor_Qualification": "Copper nitrate identity stated; hydration/absolute amount absent from shared CSV.",
            "Experiment_Generation": generation,
            "Evidence_Quality_Weight": 0.85,
        })
    return pd.DataFrame(rows)


# exp, ligand, family, MOF, Al mmol, NaOH mmol, water mL, time min, temperature C
REPRESENTATIVE_HIGH_CRYSTALLINITY = [
    (84, "H2PZVDC", "Pyrazine vinyl dicarboxylate", "MOF-321", 0.75, 1.75, 4.7, 60, 125),
    (96, "H2PZVDC", "Pyrazine vinyl dicarboxylate", "MOF-321", 0.70, 1.50, 4.0, 60, 105),
    (101, "H2PZVDC", "Pyrazine vinyl dicarboxylate", "MOF-321", 0.46, 1.75, 3.6, 60, 120),
    (114, "H2PZVDC", "Pyrazine vinyl dicarboxylate", "MOF-321", 0.66, 1.75, 4.3, 45, 120),
    (120, "H2PZVDC", "Pyrazine vinyl dicarboxylate", "MOF-321", 0.66, 1.50, 4.0, 55, 135),
    (22, "H2TVDC", "Thiophene vinyl dicarboxylate", "MOF-322", 0.46, 2.00, 3.6, 40, 145),
    (68, "H2TVDC", "Thiophene vinyl dicarboxylate", "MOF-322", 0.21, 1.75, 1.5, 35, 145),
    (86, "H2TVDC", "Thiophene vinyl dicarboxylate", "MOF-322", 0.41, 1.50, 4.3, 40, 150),
    (103, "H2TVDC", "Thiophene vinyl dicarboxylate", "MOF-322", 0.46, 2.00, 3.4, 60, 140),
    (109, "H2TVDC", "Thiophene vinyl dicarboxylate", "MOF-322", 0.99, 2.00, 3.5, 50, 150),
]


def ai_lab_positive_records() -> pd.DataFrame:
    rows = []
    for exp, ligand, family, mof, metal_mmol, naoh_mmol, water, minutes, temperature in REPRESENTATIVE_HIGH_CRYSTALLINITY:
        rows.append({
            "Record_ID": f"V11-{mof.replace('-', '')}-EXP{exp:03d}",
            "MOF": mof,
            "Legante": ligand,
            "Ligand_SMILES": "",
            "Ligand_InChIKey": "",
            "Famiglia_Legante": family,
            "Metallo": "Al",
            "Sale_Metallico": "AlCl3·6H2O",
            "Counterion_Class": "chloride",
            "Hydration_Number": 6,
            "Oxidation_State": 3,
            "Solvente": f"H2O {water:g} mL",
            "Solvent_1": "H2O", "Solvent_1_mL": water,
            "Additivo_Colinker": "NaOH",
            "mmol_Additive": naoh_mmol,
            "Temperatura_C": temperature,
            "Tempo_ore": minutes / 60.0,
            "mmol_Legante": 1.0,
            "mmol_Sale": metal_mmol,
            "Rapporto_LM": 1.0 / metal_mmol,
            "Volume_solvente_mL": water,
            "Heating_Method": "Microwave synthesis",
            "Microwave_Power_W": 300,
            "Workup": "See primary article supporting information, Section S4.",
            "Outcome_Class": 2,
            "Outcome_Label": f"Representative high-crystallinity {mof}",
            "Outcome_Raw_Score": np.nan,
            "Outcome_Score_Scale": "Article-designated representative high crystallinity",
            "Outcome_Mapping_Rule": "Direct article designation; no numerical threshold inferred.",
            "Curation_Policy": "DIRECT-HIGH-CRYSTALLINITY-TABLE-v1",
            "Yield_percent": np.nan,
            "BET_m2_g": np.nan,
            "PXRD_Confirmed": True,
            "SCXRD_Confirmed": False,
            "Evidence_Type": "PXRD; representative high-crystallinity condition",
            "Evidence_Statement": f"Primary article Table {1 if mof == 'MOF-321' else 2}, experiment {exp}.",
            "Source_DOI": AI_LAB_DOI,
            "Source_Data_DOI": "",
            "Article_Title": AI_LAB_TITLE,
            "Extraction_Method": "manual_primary_article_table",
            "Extraction_Note": "Direct transcription from main-article Table 1 or Table 2.",
            "Protocol_Completeness": "FULL_PROTOCOL",
            "Precursor_Qualification": "AlCl3·6H2O stock and NaOH stock explicitly reported in Section S4.",
            "Experiment_Generation": np.nan,
            "Evidence_Quality_Weight": 1.0,
        })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    frame = pd.concat([hkust_records(), ai_lab_positive_records()], ignore_index=True, sort=False)
    if frame.Record_ID.duplicated().any():
        raise ValueError("Duplicate tranche-03 Record_ID")
    return frame


def main() -> None:
    frame = build()
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} verified records to {OUTPUT}")
    print(frame.Outcome_Class.value_counts().sort_index().to_dict())
    print(frame.Source_DOI.value_counts().to_dict())


if __name__ == "__main__":
    main()
