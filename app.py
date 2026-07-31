import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.chem import METALS,FAMILIES,COUNTERIONS,infer_family,precursor_formula
from src.engine import predict,applicability,similar,optimize,DB
from src.pubchem import resolve_name

st.set_page_config(page_title='MOF Synthesis Assistant',page_icon='🧪',layout='wide')
st.title('🧪 MOF Synthesis Assistant v8.0')
st.caption('Chemistry-aware prediction, literature knowledge, applicability domain and synthesis optimization')

page=st.sidebar.radio('Module',['Predict synthesis','Knowledge engine','Optimizer','Model validation','About'])

def input_form(prefix='p'):
    st.subheader('Ligand identity')
    ligand=st.text_input('Ligand name, abbreviation, molecular formula or SMILES',key=prefix+'lig',placeholder="e.g. terephthalic acid, H2BDC, 4,4'-bipyrazole")
    c1,c2=st.columns([1,2])
    with c1:
        if st.button('Resolve through PubChem',key=prefix+'resolve',disabled=not ligand):
            try: st.session_state[prefix+'pubchem']=resolve_name(ligand)
            except Exception as e: st.warning(f'PubChem resolution unavailable: {e}')
    with c2:
        if prefix+'pubchem' in st.session_state: st.json(st.session_state[prefix+'pubchem'],expanded=False)
    inferred=infer_family(ligand)
    fam=st.selectbox('Ligand family',FAMILIES,index=FAMILIES.index(inferred) if inferred in FAMILIES else len(FAMILIES)-1,key=prefix+'fam')

    st.subheader('Metal precursor')
    a,b,c=st.columns(3)
    metal=a.selectbox('Metal',sorted(METALS),index=sorted(METALS).index('Zn'),key=prefix+'metal')
    oxidation=b.selectbox('Oxidation state',[1,2,3,4,5,6,'unknown'],index=1,key=prefix+'ox')
    counterion=c.selectbox('Counterion / precursor class',COUNTERIONS,key=prefix+'counter')
    hydration=st.number_input('Hydration number',min_value=0.0,max_value=20.0,value=0.0,step=0.5,key=prefix+'hyd')
    suggested=precursor_formula(metal,oxidation if oxidation!='unknown' else 2,counterion,hydration)
    salt=st.text_input('Full metal precursor formula',value=suggested,key=prefix+'salt',help='Editable. Examples: Zn(NO3)2·6H2O, Cu(OAc)2·H2O, ZrCl4.')

    st.subheader('Reaction conditions')
    c1,c2,c3=st.columns(3)
    solvent=c1.text_input('Solvent or solvent mixture',value='DMF',key=prefix+'solv')
    additive=c2.text_input('Additive / co-linker',value='Nessuno',key=prefix+'add')
    temp=c3.number_input('Temperature (°C)',20.0,300.0,120.0,key=prefix+'temp')
    c4,c5,c6=st.columns(3)
    hours=c4.number_input('Time (h)',0.1,500.0,24.0,key=prefix+'time')
    mmol_l=c5.number_input('Ligand amount (mmol)',0.0001,100.0,0.1,format='%.4f',key=prefix+'ml')
    mmol_m=c6.number_input('Metal precursor amount (mmol)',0.0001,100.0,0.1,format='%.4f',key=prefix+'mm')
    c7,c8=st.columns(2)
    ratio=c7.number_input('Ligand/metal molar ratio',0.01,100.0,float(mmol_l/mmol_m),key=prefix+'ratio')
    volume=c8.number_input('Solvent volume (mL)',0.0,1000.0,10.0,key=prefix+'vol')
    return {'Legante':ligand,'Famiglia_Legante':fam,'Metallo':metal,'Sale_Metallico':salt,'Counterion_Class':counterion,'Hydration_Number':hydration,'Oxidation_State':None if oxidation=='unknown' else oxidation,'Solvente':solvent,'Additivo_Colinker':additive,'Temperatura_C':temp,'Tempo_ore':hours,'mmol_Legante':mmol_l,'mmol_Sale':mmol_m,'Rapporto_LM':ratio,'Volume solvente':volume}

if page=='Predict synthesis':
    v=input_form('p')
    if st.button('Run prediction',type='primary'):
        _,p,pred=predict(v); ad=applicability(v)
        labels=['Failed/no useful product','Amorphous or uncertain product','Crystalline MOF']
        st.subheader(f'Predicted outcome: {labels[pred]}')
        st.bar_chart(pd.DataFrame({'Probability':p},index=labels))
        c1,c2,c3=st.columns(3); c1.metric('P(crystalline)',f'{p[2]:.1%}'); c2.metric('Applicability',ad['label']); c3.metric('AD score',f"{ad['score']:.2f}")
        if not ad['ligand_seen']: st.warning('The ligand was not observed exactly in training. The ligand-text branch uses textual similarity, but this remains extrapolation.')
        if not ad['metal_seen']: st.warning('The selected metal was not observed in training; periodic descriptors permit input but uncertainty is high.')
        st.dataframe(similar(v),use_container_width=True)
elif page=='Knowledge engine':
    st.subheader('Search the experimental database')
    q=st.text_input('Search ligand, metal, salt, solvent or family')
    d=DB.copy()
    if q:
        mask=d.astype(str).apply(lambda col: col.str.contains(q,case=False,na=False)).any(axis=1); d=d[mask]
    st.write(f'{len(d)} records')
    st.dataframe(d,use_container_width=True,height=520)
    st.download_button('Download filtered CSV',d.to_csv(index=False).encode(),'knowledge_results.csv','text/csv')
elif page=='Optimizer':
    v=input_form('o')
    n=st.slider('Number of proposed conditions',5,25,10)
    if st.button('Generate optimized conditions',type='primary'):
        out=optimize(v,n); st.dataframe(out,use_container_width=True)
        st.download_button('Download proposals',out.to_csv(index=False).encode(),'mof_optimizer_proposals.csv','text/csv')
elif page=='Model validation':
    root=Path(__file__).parent
    m=json.loads((root/'reports/external_metrics_v8_0.json').read_text())
    st.subheader('Ligand-group external test')
    st.json(m)
    st.dataframe(pd.read_csv(root/'reports/external_class_metrics_v8_0.csv'),use_container_width=True)
    st.dataframe(pd.read_csv(root/'reports/external_confusion_matrix_v8_0.csv',index_col=0),use_container_width=True)
else:
    st.markdown('''### Scope and scientific limitations
The v8.0 model accepts arbitrary ligand text, a broad metal list, full precursor formulas, counterion class, oxidation state and hydration number. Ligand names are represented by character n-grams; metal identity is supplemented with periodic descriptors; precursor formulas are parsed into hydration and counterion features.

A completely unseen ligand is therefore accepted, but **not equivalent to a structure-based prediction**. True structural extrapolation will require curated SMILES or molecular graphs for the training records. The applicability-domain warning must always be considered before experimental use.

Predictions and optimizer proposals are hypotheses for experimental prioritization, not guarantees of MOF formation.''')
