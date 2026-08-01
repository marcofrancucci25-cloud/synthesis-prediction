from src.engine import predict, optimize_joint


def _example():
    return {
        "Legante":"terephthalic acid","Ligand_User_Input":"terephthalic acid",
        "Ligand_SMILES":"O=C(O)c1ccc(C(=O)O)cc1","Famiglia_Legante":"Carboxylate",
        "Metallo":"Zn","Sale_Metallico":"Zn(NO3)2·6H2O","Counterion_Class":"nitrate",
        "Hydration_Number":6.0,"Oxidation_State":2,"Solvente":"DMF","Additivo_Colinker":"Nessuno",
        "Temperatura_C":120.0,"Tempo_ore":24.0,"mmol_Legante":0.1,"mmol_Sale":0.1,
        "Rapporto_LM":1.0,"Volume solvente":10.0,
    }


def test_prediction_does_not_mutate_input():
    values=_example(); original=dict(values)
    _,p,_=predict(values)
    assert values==original
    assert abs(float(sum(p))-1.0)<1e-8


def test_joint_optimizer_keeps_only_ligand_and_metal_fixed():
    values=_example()
    out,meta=optimize_joint(values,n_samples=350,top_n=5,constraints={"max_temperature":160,"max_time":72})
    assert len(out)>0
    assert set(out["Metallo"])=={"Zn"}
    assert set(out["Legante"])=={"terephthalic acid"}
    assert out["Temperatura_C"].max()<=160+1e-9
    assert out["Tempo_ore"].max()<=72+1e-9
    assert "pH" in meta["unsupported_not_optimized"]
