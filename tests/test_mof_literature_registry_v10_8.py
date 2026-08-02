from pathlib import Path
import re

import pandas as pd

from src.mof_registry import (
    DOI_PATTERN,
    REGISTRY,
    canonical_ligand_key,
    known_mof_matches,
    validate_registry,
)


def query(ligand, metal, oxidation=2):
    return {"Legante": ligand, "Metallo": metal, "Oxidation_State": oxidation}


def test_registry_is_valid_and_every_link_is_derived_from_its_doi():
    assert validate_registry(REGISTRY)
    assert REGISTRY["Registry_ID"].is_unique
    assert all(DOI_PATTERN.fullmatch(doi) for doi in REGISTRY["Source_DOI"])
    assert all(url == f"https://doi.org/{doi}" for url, doi in zip(REGISTRY["DOI_URL"], REGISTRY["Source_DOI"]))


def test_common_aliases_resolve_to_exact_registry_identities():
    assert canonical_ligand_key("H2BDC | benzene-1,4-dicarboxylic acid") == "bdc"
    assert canonical_ligand_key("trimesic acid") == "btc"
    assert canonical_ligand_key("H4DOBDC") == "dobdc"
    assert canonical_ligand_key("H2BPZ") == "bpz"


def test_zirconium_bdc_returns_uio66_and_not_mof5():
    result = known_mof_matches(query("terephthalic acid", "Zr", 4))
    assert result["MOF_Name"].tolist() == ["UiO-66"]
    assert result.iloc[0]["Source_DOI"] == "10.1021/ja8057953"


def test_zinc_bdc_returns_mof5_and_not_uio66():
    result = known_mof_matches(query("H2BDC", "Zn", 2))
    assert result["MOF_Name"].tolist() == ["MOF-5 / IRMOF-1"]
    assert "UiO-66" not in result["MOF_Name"].tolist()


def test_copper_btc_returns_hkust1_with_verified_doi():
    result = known_mof_matches(query("H3BTC", "Cu", 2))
    assert result["MOF_Name"].tolist() == ["HKUST-1 / Cu-BTC / MOF-199"]
    assert result.iloc[0]["DOI_URL"] == "https://doi.org/10.1016/j.tca.2016.11.013"


def test_unsubstituted_and_functionalized_bipyrazoles_cannot_cross_match():
    base = known_mof_matches(query("4,4'-bipyrazole", "Ni", 2))
    methyl = known_mof_matches(query("3,3',5,5'-tetramethyl-4,4'-bipyrazole", "Ni", 2))
    wrong_metal = known_mof_matches(query("3-amino-4,4'-bipyrazole", "Zn", 2))
    assert base["MOF_Name"].tolist() == ["Ni(BPZ) framework"]
    assert methyl["MOF_Name"].tolist() == ["Ni-Me4BPZ layered MOF"]
    assert wrong_metal.empty


def test_oxidation_state_mismatch_is_exposed_not_hidden():
    result = known_mof_matches(query("2-methylimidazole", "Co", 3))
    assert result["MOF_Name"].tolist() == ["ZIF-67"]
    assert not bool(result.iloc[0]["Oxidation_State_Match"])
    assert "differs" in result.iloc[0]["Match_Level"]


def test_unknown_pair_has_no_claim_and_no_fallback_guess():
    assert known_mof_matches(query("unregistered ligand 123", "Zn", 2)).empty
    assert known_mof_matches(query("H3BTC", "Zr", 4)).empty


def test_app_uses_compatibility_safe_literature_loader():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'getattr(engine, "known_mof_matches", None)' in source
    assert "structural identification requires PXRD" not in source  # rendered copy is plain-language and explicit
    assert "not identification of the obtained phase" in source
