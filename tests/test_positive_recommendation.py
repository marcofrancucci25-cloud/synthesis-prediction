from src.engine import POSITIVE_DB, POSITIVE_MODEL, optimize_joint

def test_positive_model_loaded():
    assert POSITIVE_MODEL is not None
    assert POSITIVE_MODEL.get('version') == '10.4.0'
    assert len(POSITIVE_DB) == 656
    assert POSITIVE_DB['Condition_Signature'].astype(str).is_unique
    assert {'Evidence_Weight','Quality_Tier','Diversity_Weight'}.issubset(POSITIVE_DB.columns)

def test_optimizer_uses_quality_weighted_support():
    values={
        'Legante':'terephthalic acid','Famiglia_Legante':'Carboxylate','Metallo':'Zn',
        'Sale_Metallico':'Zn(NO3)2·6H2O','Counterion_Class':'nitrate','Hydration_Number':6,
        'Oxidation_State':2,'Solvente':'DMF','Additivo_Colinker':'Nessuno',
        'Temperatura_C':120,'Tempo_ore':24,'mmol_Legante':0.2,'mmol_Sale':0.1,
        'Rapporto_LM':2,'Volume solvente':10,'Procedura_Sintetica':'Solvothermal'
    }
    result,meta=optimize_joint(values,n_samples=300,top_n=5)
    assert len(result) <= 5
    assert result['Positive_support_score'].between(0,1).all()
    assert 'Nearest_positive_quality' in result.columns
    assert meta['optimizer_version']=='10.6.0'

if __name__=='__main__':
    test_positive_model_loaded(); test_optimizer_uses_quality_weighted_support(); print('positive recommendation tests passed')
