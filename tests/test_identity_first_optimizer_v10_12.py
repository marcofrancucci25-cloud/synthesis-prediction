from src.engine import POSITIVE_DB, optimize_joint
from src.optimizer import _select_positive_templates
from src.chem import build_row


def _bpz_values():
    return {
        "Legante": "3,5-diamino-4,4'-bipyrazole",
        "Famiglia_Legante": "Bipyrazole/pyrazole",
        "Metallo": "Zn",
        "Sale_Metallico": "Zn(NO3)2·6H2O",
        "Counterion_Class": "nitrate",
        "Hydration_Number": 6,
        "Oxidation_State": 2,
        "Solvente": "DMF",
        "Additivo_Colinker": "Nessuno",
        "Temperatura_C": 120,
        "Tempo_ore": 24,
        "mmol_Legante": 0.1,
        "mmol_Sale": 0.1,
        "Rapporto_LM": 1.0,
        "Volume solvente": 10.0,
    }


def test_exact_pair_templates_exclude_unrelated_linkers_when_evidence_is_sufficient():
    base = build_row(_bpz_values()).iloc[0].to_dict()
    templates = _select_positive_templates(POSITIVE_DB, base)
    assert len(templates) >= 3
    assert templates["Legante"].eq(base["Legante"]).all()
    assert templates["Metallo"].eq("Zn").all()


def test_bipyrazole_optimizer_no_longer_uses_bdc_template():
    result, _ = optimize_joint(_bpz_values(), n_samples=500, top_n=5)
    assert len(result)
    assert not result["Template_Positive_ID"].astype(str).eq("POS-0225").any()
    assert result["AD_score"].between(0, 1).all()


def test_volume_does_not_change_production_model_scores():
    from src.engine import predict

    values = _bpz_values()
    _, first, _ = predict(values)
    values["Volume solvente"] = 100.0
    _, second, _ = predict(values)
    assert (abs(first - second) < 1e-12).all()
