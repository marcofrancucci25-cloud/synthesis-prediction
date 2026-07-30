from __future__ import annotations
import pandas as pd
import streamlit as st


def input_form(engine, prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        family = st.selectbox("Famiglia del legante", engine.options("Famiglia_Legante"), key=prefix+"family")
        metal = st.selectbox("Metallo", engine.options("Metallo"), key=prefix+"metal")
        salt = st.selectbox("Sale metallico", engine.options("Sale_Metallico"), key=prefix+"salt")
        solvent = st.selectbox("Solvente", engine.options("Solvente"), key=prefix+"solvent")
    with c2:
        additive = st.selectbox("Additivo / co-linker", engine.options("Additivo_Colinker"), key=prefix+"additive")
        temp = st.number_input("Temperatura (°C)", value=engine.numeric_default("Temperatura_C"), step=5.0, key=prefix+"temp")
        hours = st.number_input("Tempo (h)", min_value=0.01, value=engine.numeric_default("Tempo_ore"), step=1.0, key=prefix+"hours")
        volume = st.number_input("Volume solvente (mL)", min_value=0.0, value=max(0.0, engine.numeric_default("Volume solvente")), step=1.0, key=prefix+"volume")
    with c3:
        ligand_mmol = st.number_input("mmol legante", min_value=0.0001, value=max(0.0001, engine.numeric_default("mmol_Legante")), format="%.4f", key=prefix+"ligmmol")
        salt_mmol = st.number_input("mmol sale", min_value=0.0001, value=max(0.0001, engine.numeric_default("mmol_Sale")), format="%.4f", key=prefix+"saltmmol")
        ratio = st.number_input("Rapporto L/M", min_value=0.001, value=max(0.001, engine.numeric_default("Rapporto_LM")), format="%.3f", key=prefix+"ratio")
    return {"Famiglia_Legante": family, "Metallo": metal, "Sale_Metallico": salt,
            "Solvente": solvent, "Additivo_Colinker": additive, "Temperatura_C": temp,
            "Tempo_ore": hours, "Volume solvente": volume, "mmol_Legante": ligand_mmol,
            "mmol_Sale": salt_mmol, "Rapporto_LM": ratio}


def probability_table(probabilities, labels):
    return pd.DataFrame({"Esito": [labels[i] for i in range(3)], "Probabilità": probabilities}).set_index("Esito")
