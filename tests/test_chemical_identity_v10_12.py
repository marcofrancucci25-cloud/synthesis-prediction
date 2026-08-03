from src.chem import canonicalize_ligand_for_model


def test_bdc_positional_isomers_are_not_collapsed():
    para = canonicalize_ligand_for_model("BDC")
    meta = canonicalize_ligand_for_model("Isophthalic acid (1,3-BDC)")
    ortho = canonicalize_ligand_for_model("Phthalic acid (1,2-BDC)")
    assert para == "1,4-Benzenedicarboxylic acid (H2BDC)"
    assert meta == "Isophthalic acid (1,3-BDC)"
    assert ortho == "Phthalic acid (1,2-BDC)"
    assert len({para, meta, ortho}) == 3


def test_mixed_linker_identity_is_preserved():
    mixed = "2,6-NDC (0.5) / H3BTC (0.5)"
    assert canonicalize_ligand_for_model(mixed) == mixed


def test_resolver_pipe_alias_still_canonicalizes_exact_component():
    assert canonicalize_ligand_for_model("user label | H2BDC") == (
        "1,4-Benzenedicarboxylic acid (H2BDC)"
    )
