import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_v11_gold_dataset", ROOT / "tools" / "build_v11_gold_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v11_builder_enforces_article_level_doi_separation():
    gold, summary = MODULE.build()
    train = set(gold.loc[gold.Training_Eligibility, "Source_DOI"]) - {""}
    benchmark = set(gold.loc[~gold.Training_Eligibility, "Source_DOI"]) - {""}
    assert not train & benchmark
    assert summary["readiness_gates"]["no_doi_overlap_between_training_and_benchmark"]


def test_all_zif8_variants_from_locked_article_remain_external():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    zif = gold[gold.Source_DOI.eq("10.1039/C3CE42485E")]
    assert len(zif) == 8
    assert not zif.Training_Eligibility.astype(bool).any()
    assert zif.Partition_Role.str.startswith("LOCKED_EXTERNAL").all()


def test_cobalt_amino_bipyrazolate_protocol_is_complete_and_eligible():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    row = gold[gold.Record_ID.eq("V11-LIT-017")].iloc[0]
    assert row.MOF == "Co(BPZNH2)·DMF"
    assert row.Source_DOI == "10.1021/acs.inorgchem.0c00481"
    assert row.Temperatura_C == 120
    assert row.Tempo_ore == 24
    assert row.mmol_Legante == row.mmol_Sale == 1
    assert row.Volume_solvente_mL == 15
    assert bool(row.Training_Eligibility)
    assert bool(row.PXRD_Confirmed)


def test_only_quality_gated_laboratory_records_enter_gold_training_pool():
    source = pd.read_csv(ROOT / "data" / "laboratory_syntheses_normalized_v10_6.csv")
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    lab = gold[gold.Source_Type.eq("direct_laboratory_experiment")]
    assert len(lab) == int(source.Training_Status.eq("INCLUDE").sum()) == 17
    assert "LAB-DDS-001" not in set(lab.Record_ID)
    assert not {"LAB-DDS-012", "LAB-DDS-013"} & set(lab.Record_ID)


def test_repeated_conditions_are_retained_but_not_overweighted():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    train = gold[gold.Training_Eligibility.astype(bool)]
    per_condition = train.groupby("Condition_Signature").Training_Weight.sum()
    assert (per_condition <= 1.0 + 1e-12).all()
    repeated = train[train.Replicate_Group_Size.gt(1)]
    assert not repeated.empty
    assert repeated.Training_Weight.lt(1.0).all()


def test_readiness_gate_prevents_premature_retraining():
    summary = json.loads((ROOT / "reports" / "v11_dataset_readiness.json").read_text())
    assert not summary["promotion_gate_passed"]
    assert not summary["production_model_retrained"]
    assert summary["training_class_distribution"] == {"0": 13, "1": 65, "2": 85}
    assert summary["gold_records_total"] == 179
    assert summary["training_candidates"] == 163
    assert summary["training_ligand_families"] == 5
    assert summary["readiness_gates"]["minimum_5_ligand_families"]


def test_al_pmof_tranche_preserves_article_defined_pxrd_labels():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    rows = gold[gold.Source_DOI.eq("10.1038/s42004-022-00785-2")]
    assert len(rows) == 45
    assert rows.Outcome_Class.value_counts().sort_index().to_dict() == {0: 4, 1: 16, 2: 25}
    assert set(rows.Source_Data_DOI) == {"10.5281/zenodo.7186602"}
    assert set(rows.Source_Data_URL) == {"https://doi.org/10.5281/zenodo.7186602"}
    assert rows.Condition_Signature.nunique() == 45
    assert rows.Outcome_Mapping_Rule.str.contains("1=no powder", regex=False).all()


def test_al_pmof_best_yield_protocol_matches_supporting_table_s5():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    row = gold[gold.Record_ID.eq("V11-ALPMOF-G2S4")].iloc[0]
    assert row.MOF == "Al-PMOF (Al2(OH)2TCPP)"
    assert row.Sale_Metallico == "AlCl3·6H2O"
    assert row.Temperatura_C == 190
    assert abs(row.Tempo_ore - (50 / 60)) < 1e-12
    assert row.Microwave_Power_W == 250
    assert row.mmol_Legante == 0.051
    assert row.mmol_Sale == 0.099
    assert row.Volume_solvente_mL == 2
    assert row.Outcome_Raw_Score == 8
    assert row.Yield_percent == 75
    assert bool(row.PXRD_Confirmed)


def test_single_literature_source_cannot_dominate_future_training():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    train_literature = gold[
        gold.Training_Eligibility.astype(bool) & gold.Source_DOI.notna() & gold.Source_DOI.ne("")
    ]
    influence = train_literature.groupby("Source_DOI").Training_Weight.sum()
    assert (influence <= 20.0 + 1e-12).all()


def test_every_literature_url_is_derived_from_validated_doi():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv").fillna("")
    literature = gold[gold.Source_DOI.ne("")]
    assert all(MODULE.DOI_PATTERN.fullmatch(doi) for doi in literature.Source_DOI)
    assert all(url == f"https://doi.org/{doi}" for url, doi in zip(literature.Source_URL, literature.Source_DOI))


def test_hkust_robotic_campaign_preserves_scores_and_preregistered_mapping():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    rows = gold[gold.Source_DOI.eq("10.1038/s41467-019-08483-9")]
    assert len(rows) == 90
    assert rows.Outcome_Class.value_counts().sort_index().to_dict() == {0: 6, 1: 48, 2: 36}
    assert set(rows.Source_Data_DOI) == {"10.24435/materialscloud:2018.0011/v3"}
    assert set(rows.Curation_Policy) == {"V11-PXRD-NORMALIZED-0.30-0.80-v1"}
    assert rows.Outcome_Raw_Score.notna().all()
    assert rows.BET_m2_g.notna().all()
    assert rows.Evidence_Quality_Weight.eq(0.85).all()
    assert rows.mmol_Legante.isna().all() and rows.mmol_Sale.isna().all()


def test_hkust_boundary_examples_are_mapped_from_pxrd_score_not_bet():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    low = gold[gold.Record_ID.eq("V11-HKUST-G1-10")].iloc[0]
    high = gold[gold.Record_ID.eq("V11-HKUST-G2-04")].iloc[0]
    assert low.Outcome_Raw_Score == 0.1 and low.Outcome_Class == 0
    assert high.Outcome_Raw_Score == 0.95 and high.Outcome_Class == 2
    assert low.Curation_Policy == high.Curation_Policy


def test_hkust_raw_source_is_the_complete_published_table():
    raw = pd.read_csv(ROOT / "data" / "v11_source_hkust1_moesm5.csv", comment="#")
    assert len(raw) == 90
    assert raw.columns.tolist() == [
        "sample", "H2O", "DMF", "EeOH", "MeOH", "iPrOH", "ReaRatio",
        "Temp", "Power", "Time", "Crystallinity", "BET",
    ]


def test_mof321_and_mof322_protocols_are_direct_high_crystallinity_evidence():
    gold = pd.read_csv(ROOT / "data" / "v11_gold_synthesis_records.csv")
    rows = gold[gold.Source_DOI.eq("10.1021/acscentsci.3c01087")]
    assert len(rows) == 10
    assert rows.groupby("MOF").size().to_dict() == {"MOF-321": 5, "MOF-322": 5}
    assert rows.Outcome_Class.eq(2).all()
    assert rows.PXRD_Confirmed.astype(bool).all()
    assert rows.Outcome_Mapping_Rule.eq(
        "Direct article designation; no numerical threshold inferred."
    ).all()
    exp84 = rows[rows.Record_ID.eq("V11-MOF321-EXP084")].iloc[0]
    assert exp84.mmol_Sale == 0.75
    assert exp84.mmol_Additive == 1.75
    assert exp84.Volume_solvente_mL == 4.7
    assert exp84.Temperatura_C == 125
    assert exp84.Microwave_Power_W == 300
