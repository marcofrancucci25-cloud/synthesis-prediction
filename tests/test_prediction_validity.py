from src.engine import prediction_validity, applicability

BASE={
 'Legante':'terephthalic acid','Famiglia_Legante':'Carboxylate','Metallo':'Zn',
 'Sale_Metallico':'Zn(NO3)2·6H2O','Counterion_Class':'nitrate','Hydration_Number':6,
 'Oxidation_State':2,'Solvente':'DMF','Additivo_Colinker':'Nessuno',
 'Temperatura_C':120,'Tempo_ore':24,'mmol_Legante':0.1,'mmol_Sale':0.1,
 'Rapporto_LM':1.0,'Volume solvente':10.0,
}

def test_typical_conditions_are_supported():
    r=prediction_validity(BASE)
    assert r['reliable']
    assert r['score'] >= 0.7

def test_extreme_temperature_is_rejected():
    v=dict(BASE, Temperatura_C=280)
    r=prediction_validity(v)
    assert not r['reliable']
    assert 'Outside validated' in r['label']
    assert any('Temperature' in x for x in r['issues'])

def test_extreme_time_is_rejected():
    v=dict(BASE, Tempo_ore=0.1)
    r=prediction_validity(v)
    assert not r['reliable']

def test_ratio_amount_inconsistency_is_detected():
    v=dict(BASE, mmol_Legante=0.1, mmol_Sale=0.1, Rapporto_LM=8.0)
    r=prediction_validity(v)
    assert any('inconsistent' in x for x in r['issues'])

def test_numeric_extremes_reduce_applicability():
    normal=applicability(BASE)['score']
    extreme=applicability(dict(BASE, Temperatura_C=280, Tempo_ore=300))['score']
    assert extreme < normal
