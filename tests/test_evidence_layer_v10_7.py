import pandas as pd

from src.chem import canonicalize_family, canonicalize_ligand_for_model
from src.engine import predict, verified_precedents


def test_public_vocabulary_maps_to_training_vocabulary():
    assert canonicalize_family("Carboxylate", "H2BDC") == "Carbossilati aromatici"
    assert canonicalize_family("Imidazolate/azolate", "2-methylimidazole") == "Imidazolati"
    assert canonicalize_ligand_for_model("H2BDC | benzene-1,4-dicarboxylic acid") == "1,4-Benzenedicarboxylic acid (H2BDC)"


def test_exact_literature_protocol_returns_crystalline_doi():
    record = pd.read_csv("data/literature_crystalline_challenge_v10_6.csv").iloc[0]
    evidence = verified_precedents(record.to_dict())
    strongest = evidence.iloc[0]
    assert strongest.Match_Level == "Exact verified protocol"
    assert strongest.Outcome_Class == 2
    assert strongest.Source_DOI == "10.1002/adsu.202500854"
    _, probabilities, predicted = predict(record.to_dict())
    assert predicted == 2
    assert probabilities[2] > 0.45


def test_exact_failed_laboratory_protocol_is_not_overridden_by_positive_neighbors():
    integrated = pd.read_csv("data/knowledge_database_integrated_v10_6.csv")
    record = integrated[integrated.ID.eq("LAB-DDS-007")].iloc[0]
    evidence = verified_precedents(record.to_dict())
    strongest = evidence.iloc[0]
    assert strongest.Match_Level == "Exact verified protocol"
    assert strongest.Outcome_Class == 0
    assert strongest.Verified_Outcome == "Failed"


def test_unseen_system_has_no_verified_precedent():
    evidence = verified_precedents({
        "Legante": "invented linker QX-999", "Famiglia_Legante": "Other/unknown",
        "Metallo": "Au", "Sale_Metallico": "AuCl3", "Solvente": "DMF",
        "Additivo_Colinker": "Nessuno", "Temperatura_C": 120,
        "Tempo_ore": 24, "Rapporto_LM": 1, "Volume solvente": 10,
    })
    assert evidence.empty
