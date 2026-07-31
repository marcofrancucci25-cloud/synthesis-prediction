from src.engine import predict,optimize
v={'Legante':'terephthalic acid','Famiglia_Legante':'Carboxylate','Metallo':'Zn','Sale_Metallico':'Zn(NO3)2·6H2O','Solvente':'DMF','Additivo_Colinker':'Nessuno','Temperatura_C':120,'Tempo_ore':24,'mmol_Legante':0.1,'mmol_Sale':0.1,'Rapporto_LM':1,'Volume solvente':10,'Hydration_Number':6,'Oxidation_State':2}
_,p,k=predict(v)
assert abs(sum(p)-1)<1e-8
assert len(optimize(v,3))==3
print('Smoke test passed',p,k)
