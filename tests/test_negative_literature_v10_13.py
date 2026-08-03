from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)


def test_negative_amorphous_tranche_integrity():
    frame = pd.read_csv(ROOT / "data" / "v13_negative_amorphous_literature.csv")
    assert len(frame) == 759
    assert frame["Record_ID"].is_unique
    assert frame["Source_DOI"].map(lambda value: bool(DOI_RE.match(str(value)))).all()
    assert set(frame["Outcome_Class"]) == {0, 1}
    assert frame["Outcome_Class"].value_counts().to_dict() == {0: 673, 1: 86}
    assert not frame["Training_Eligibility"].astype(bool).any()
    assert frame["Outcome_Raw_Text"].notna().all()
    assert frame["Outcome_Mapping_Rule"].notna().all()


def test_unknown_phases_are_quarantined_not_negative():
    review = pd.read_csv(ROOT / "data" / "v13_literature_records_needing_review.csv")
    negative = pd.read_csv(ROOT / "data" / "v13_negative_amorphous_literature.csv")
    assert len(review) == 71
    assert review["Outcome_Class"].isna().all()
    assert set(review["Primary_Phase_Raw"].str.casefold()) == {"unknown phase"}
    assert not set(review["Record_ID"]).intersection(negative["Record_ID"])


def test_campaign_key_is_preserved_and_complete():
    source = pd.read_csv(ROOT / "data" / "v13_source_rare_earth_landscape.csv")
    assert len(source) == 1488
    assert source["Sample#"].is_unique
    assert source["SampleID"].is_unique
    assert (source["Primary Phase"].eq("no solid product").sum() == 673)
    assert (source["Primary Phase"].eq("no crystalline product").sum() == 85)


def test_verified_evidence_loader_includes_new_tranche():
    from src.engine import VERIFIED_EVIDENCE

    ids = set(VERIFIED_EVIDENCE["Evidence_ID"].astype(str))
    assert "V13-REHT-0001" in ids
    assert "V13-AMOF-WU-001" in ids
