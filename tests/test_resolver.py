from rdkit import Chem
from src.resolver import normalize_query, detect_input_type, resolve_ligand


def test_normalization():
    assert normalize_query(" 4,4′-bipyrazole ") == "4,4'-bipyrazole"


def test_direct_smiles():
    result = resolve_ligand("O=C(O)c1ccc(C(=O)O)cc1")
    assert result["success"]
    assert result["source"] == "direct SMILES / RDKit"
    assert result["descriptors"]["HBA"] >= 2


def test_input_detection():
    assert detect_input_type("100-21-0") == "CAS"
    assert detect_input_type("C8H6O4") == "molecular formula"


def test_amino_bipyrazole_curated_aliases():
    for query in ["3-amino-4,4'-bipyrazole", "3-amino-4,4′-bipyrazole", "BPZNH2", "H2BPZ-NH2"]:
        result = resolve_ligand(query)
        assert result["success"]
        assert result["molecular_formula"] == "C6H7N5"
        assert result["source"] == "curated MOF linker library / RDKit"
        assert result["confidence"] == "high"
        assert abs(result["molecular_weight"] - 149.157) < 0.02
        mol = Chem.MolFromSmiles(result["smiles"])
        amino = Chem.MolFromSmarts("[NX3;H2]-[c,n]")
        assert mol.HasSubstructMatch(amino)


def test_generic_aminobipyrazole_is_rejected_as_ambiguous():
    result = resolve_ligand("aminobipyrazole")
    assert not result["success"]
    assert result["confidence"] == "unresolved"
    assert "ambiguous" in result["message"].lower()
