from src.structural import canonicalize_smiles, ligand_vector, scaffold_smiles

def test_structural_features_are_deterministic():
    s='O=C(O)c1ccc(C(=O)O)cc1'
    assert canonicalize_smiles(s)
    assert ligand_vector(s).shape == (1034,)
    assert (ligand_vector(s) == ligand_vector(s)).all()
    assert scaffold_smiles(s) == 'c1ccccc1'
