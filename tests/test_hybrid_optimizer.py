from src.engine import optimize_joint, POSITIVE_DB

SAMPLE = {
    'Legante':'terephthalic acid','Famiglia_Legante':'Carboxylate','Metallo':'Zn',
    'Sale_Metallico':'Zn(NO3)2·6H2O','Counterion_Class':'nitrate','Hydration_Number':6,
    'Oxidation_State':2,'Solvente':'DMF','Additivo_Colinker':'Nessuno',
    'Temperatura_C':120,'Tempo_ore':24,'mmol_Legante':0.1,'mmol_Sale':0.1,
    'Rapporto_LM':1.0,'Volume solvente':10.0,
}

def test_positive_library_is_separate_and_positive_only():
    assert len(POSITIVE_DB) >= 600
    assert set(POSITIVE_DB['Esito_ML'].astype(int)) == {2}


def test_hybrid_optimizer_scores_positive_support_and_keeps_identity_fixed():
    out, meta = optimize_joint(SAMPLE, objective='Balanced conditions', n_samples=600, top_n=6)
    assert len(out) == 6
    assert set(out['Metallo']) == {'Zn'}
    assert set(out['Legante']) == {'terephthalic acid'}
    assert out['Positive_support_score'].between(0, 1).all()
    assert out['P_Crystalline'].between(0, 1).all()
    assert {'successful-template mutation', 'broad exploration'} & set(out['Generation_mode'])
    assert meta['positive_library_rows'] >= 600
    assert meta['template_candidates'] > 0
    assert meta['exploration_candidates'] > 0
