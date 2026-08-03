import numpy as np
import pandas as pd

import src.engine as engine


BASE = {
    "Legante": "terephthalic acid",
    "Famiglia_Legante": "Carboxylate",
    "Metallo": "Zr",
    "Sale_Metallico": "ZrCl4",
    "Counterion_Class": "chloride",
    "Hydration_Number": 0,
    "Oxidation_State": 4,
    "Solvente": "DMF",
    "Additivo_Colinker": "H2O",
    "Temperatura_C": 120,
    "Tempo_ore": 24,
    "mmol_Legante": 0.1,
    "mmol_Sale": 0.1,
    "Rapporto_LM": 1.0,
    "Volume solvente": 10.0,
}


def test_ratio_candidates_are_observed_and_exclude_artificial_point_one():
    candidates, support, _, limits = engine._supported_ratio_candidates(BASE)
    observed = set(pd.to_numeric(engine.DB.Rapporto_LM, errors="coerce").dropna().round(4))
    assert candidates
    assert 0.1 not in candidates
    assert all(value in observed for value in candidates)
    assert all(limits[0] <= value <= limits[1] for value in candidates)
    assert all(support[value] > 0 for value in candidates)


def test_ratio_sensitivity_rebalances_amounts_and_preserves_total(monkeypatch):
    calls = []

    def fake_predict(values):
        snapshot = dict(values)
        calls.append(snapshot)
        ratio = float(snapshot["Rapporto_LM"])
        crystalline = 0.8 / (1.0 + ratio)
        probability = np.array([0.1, 0.9 - crystalline, crystalline])
        return pd.DataFrame([snapshot]), probability, int(np.argmax(probability))

    monkeypatch.setattr(engine, "predict", fake_predict)
    influence, _ = engine.explain_prediction(BASE)
    row = influence[influence.Field.eq("Rapporto_LM")].iloc[0]
    ratio_calls = [call for call in calls if not np.isclose(float(call["Rapporto_LM"]), 1.0)]
    assert ratio_calls
    for call in ratio_calls:
        ligand = float(call["mmol_Legante"])
        metal = float(call["mmol_Sale"])
        ratio = float(call["Rapporto_LM"])
        assert np.isclose(ligand / metal, ratio)
        assert np.isclose(ligand + metal, 0.2)
        assert engine.prediction_validity(call)["reliable"]
    assert row.Best_alternative != 0.1
    assert row.Alternative_Validity_Score >= 0.72
    assert "constant total precursor amount" in row.Best_Alternative_Detail


def test_interface_does_not_present_sensitivity_as_optimizer_recommendation():
    source = open("app.py", encoding="utf-8").read()
    assert "best tested" not in source
    assert "Controlled local model sensitivity—not a synthesis recommendation" in source
    assert "best supported perturbation" in source


def test_no_reliable_perturbation_returns_an_empty_typed_result(monkeypatch):
    def fake_predict(values):
        return pd.DataFrame([values]), np.array([0.2, 0.4, 0.4]), 1

    monkeypatch.setattr(engine, "predict", fake_predict)
    extreme = dict(BASE, Temperatura_C=500, Tempo_ore=500)
    influence, base_probability = engine.explain_prediction(extreme)
    assert influence.empty
    assert "Influence" in influence.columns
    assert base_probability == 0.4
