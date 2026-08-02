"""Materialize the verified Al-PMOF high-throughput literature tranche.

The transcription below comes from Tables S3--S5 of the supporting
information for DOI 10.1038/s42004-022-00785-2.  The paper defines its own
PXRD score semantics: 1 = no powder, 2--5 = amorphous/poor crystallinity and
6--10 = increasingly crystalline Al-PMOF.  We preserve both the raw score and
that explicit mapping instead of inventing a threshold from yield.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "v11_literature_expansion_tranche_02.csv"
ARTICLE_DOI = "10.1038/s42004-022-00785-2"
DATA_DOI = "10.5281/zenodo.7186602"
ARTICLE_TITLE = (
    "Using genetic algorithms to systematically improve the synthesis "
    "conditions of Al-PMOF"
)

# sample, power_W, temperature_C, time_min, concentration_code, organic solvent,
# categorical PXRD score, yield_percent (generation 1 did not report yield)
GENERATION_1 = [
    ("G1S1", 300, 200, 60, 3, "DMSO", 1, None),
    ("G1S2", 200, 190, 60, 3, "EtOH", 1, None),
    ("G1S3", 300, 175, 40, 1, "DMSO", 5, None),
    ("G1S4", 250, 200, 20, 2, "DMF", 1, None),
    ("G1S5", 200, 175, 20, 3, "DMSO", 5, None),
    ("G1S6", 300, 200, 60, 1, "EtOH", 4, None),
    ("G1S7", 200, 200, 60, 1, "DMSO", 8, None),
    ("G1S8", 300, 175, 20, 3, "EtOH", 2, None),
    ("G1S9", 200, 175, 60, 2, "DMA", 4, None),
    ("G1S10", 250, 175, 60, 1, "1-propanol", 7, None),
    ("G1S11", 200, 190, 20, 1, "DMSO", 7, None),
    ("G1S12", 300, 200, 30, 3, "EtOH", 2, None),
    ("G1S13", 300, 175, 30, 2, "DMF", 1, None),
    ("G1S14", 300, 190, 60, 2, "DMF", 3, None),
    ("G1S15", 200, 200, 20, 1, "EtOH", 3, None),
    ("G1S16", 200, 200, 20, 3, "DMSO", 3, None),
    ("G1S17", 200, 190, 40, 3, "DMF", 2, None),
    ("G1S18", 200, 190, 40, 1, "DMF", 2, None),
    ("G1S19", 250, 180, 40, 2, "EtOH", 8, None),
    ("G1S20", 200, 190, 40, 2, "DMSO", 8, None),
    ("G1S21", 300, 175, 50, 3, "1-propanol", 3, None),
    ("G1S22", 200, 175, 20, 1, "DMA", 5, None),
    ("G1S23", 200, 175, 60, 1, "DMA", 5, None),
    ("G1S24", 300, 190, 20, 1, "1-propanol", 5, None),
    ("G1S25", 300, 190, 20, 3, "DMA", 2, None),
]

GENERATION_2 = [
    ("G2S1", 200, 190, 45, 2, "1-propanol", 8, 38.0),
    ("G2S2", 220, 200, 50, 2, "DMSO", 10, 15.0),
    ("G2S3", 290, 180, 55, 2, "1-propanol", 8, 32.0),
    ("G2S4", 250, 190, 50, 1, "EtOH", 8, 75.0),
    ("G2S5", 300, 190, 40, 1, "EtOH", 8, 56.0),
    ("G2S6", 200, 185, 30, 2, "1-propanol", 7, 34.0),
    ("G2S7", 200, 195, 40, 2, "EtOH", 10, 10.0),
    ("G2S8", 250, 185, 50, 2, "EtOH", 8, 40.0),
    ("G2S9", 300, 200, 30, 2, "1-propanol", 8, 50.0),
    ("G2S10", 250, 180, 50, 2, "1-propanol", 8, 33.0),
    ("G2S11", 240, 200, 50, 2, "1-propanol", 7, 20.0),
    ("G2S12", 200, 190, 50, 2, "DMSO", 7, 1.0),
    ("G2S13", 200, 180, 40, 2, "DMSO", 9, 3.5),
    ("G2S14", 200, 195, 50, 1, "DMSO", 10, 12.0),
    ("G2S15", 250, 195, 60, 1, "EtOH", 9, 65.0),
    ("G2S16", 200, 185, 40, 2, "DMSO", 10, 11.0),
    ("G2S17", 200, 190, 25, 1, "DMSO", 10, 4.0),
    ("G2S18", 230, 190, 40, 2, "EtOH", 8, 51.0),
    ("G2S19", 200, 180, 45, 2, "DMSO", 10, 3.0),
    ("G2S20", 200, 180, 30, 2, "EtOH", 8, 45.0),
]

CONCENTRATIONS = {
    1: {"volume": 2.0, "ligand": 0.051, "metal": 0.099},
    2: {"volume": 2.0, "ligand": 0.025, "metal": 0.050},
    3: {"volume": 4.0, "ligand": 0.025, "metal": 0.050},
}


def _outcome(score: int) -> tuple[int, str]:
    if score == 1:
        return 0, "Failed / no isolated powder"
    if score <= 5:
        return 1, "Amorphous or poorly crystalline product"
    return 2, "Crystalline Al-PMOF"


def build() -> pd.DataFrame:
    rows = []
    for generation, experiments in ((1, GENERATION_1), (2, GENERATION_2)):
        for sample, power, temperature, minutes, concentration, organic, score, yield_pct in experiments:
            amounts = CONCENTRATIONS[concentration]
            outcome_class, outcome_label = _outcome(score)
            rows.append({
                "Record_ID": f"V11-ALPMOF-{sample}",
                "MOF": "Al-PMOF (Al2(OH)2TCPP)",
                "Legante": "meso-tetra(4-carboxyphenyl)porphine (H2TCPP)",
                "Ligand_SMILES": "",
                "Ligand_InChIKey": "",
                "Famiglia_Legante": "Porphyrin tetracarboxylate",
                "Metallo": "Al",
                "Sale_Metallico": "AlCl3·6H2O",
                "Counterion_Class": "chloride",
                "Hydration_Number": 6,
                "Oxidation_State": 3,
                "Solvente": f"H2O/{organic} (80:20 v/v)",
                "Solvent_1": "H2O",
                "Solvent_1_mL": amounts["volume"] * 0.8,
                "Solvent_2": organic,
                "Solvent_2_mL": amounts["volume"] * 0.2,
                "Additivo_Colinker": "Nessuno",
                "Temperatura_C": temperature,
                "Tempo_ore": minutes / 60.0,
                "mmol_Legante": amounts["ligand"],
                "mmol_Sale": amounts["metal"],
                "Rapporto_LM": amounts["ligand"] / amounts["metal"],
                "Volume_solvente_mL": amounts["volume"],
                "Heating_Method": "Automated microwave synthesis",
                "Microwave_Power_W": power,
                "Workup": (
                    "Centrifuged; washed with reaction organic solvent, then acetone; "
                    "dried overnight at 60 °C (DMF additionally used when needed)."
                ),
                "Outcome_Class": outcome_class,
                "Outcome_Label": outcome_label,
                "Outcome_Raw_Score": score,
                "Outcome_Mapping_Rule": (
                    "Article-defined PXRD scale: 1=no powder; 2-5=amorphous/poor "
                    "crystallinity; 6-10=crystalline Al-PMOF."
                ),
                "Yield_percent": yield_pct,
                "PXRD_Confirmed": score >= 6,
                "SCXRD_Confirmed": False,
                "Evidence_Type": "PXRD qualitative categorical score",
                "Evidence_Statement": (
                    f"Supporting Table S{4 if generation == 1 else 5}: {sample}, "
                    f"article-assigned crystallinity score {score}/10."
                ),
                "Source_DOI": ARTICLE_DOI,
                "Source_Data_DOI": DATA_DOI,
                "Article_Title": ARTICLE_TITLE,
                "Extraction_Method": "manual_primary_supplementary_table",
                "Extraction_Note": (
                    f"Direct transcription of Table S{4 if generation == 1 else 5}; "
                    f"precursor quantities/volume resolved through Table S3, "
                    f"concentration code {concentration}."
                ),
                "Protocol_Completeness": "FULL_PROTOCOL",
                "Precursor_Qualification": "Formula and hydration explicitly reported in Table S3.",
                "Experiment_Generation": generation,
            })
    return pd.DataFrame(rows)


def main() -> None:
    frame = build()
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} verified records to {OUTPUT}")
    print(frame.Outcome_Class.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
