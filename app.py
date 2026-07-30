from __future__ import annotations
import streamlit as st
import pandas as pd
from src.assets import load_assets
from src.constants import LABELS, CLASS_SHORT
from src.engine import MOFSynthesisEngine
from src.ui import input_form, probability_table

st.set_page_config(page_title="MOF Synthesis Assistant", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
.small-note {font-size: .88rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Caricamento del modello...")
def cached_assets():
    return load_assets()

try:
    assets = cached_assets()
    engine = MOFSynthesisEngine(assets)
except Exception as exc:
    st.error("Impossibile inizializzare l'applicazione.")
    st.exception(exc)
    st.stop()

st.title("🧪 MOF Synthesis Assistant")
st.caption("Knowledge Engine · Prediction Engine · Applicability Domain · Local Explainability · Optimizer Engine")

with st.sidebar:
    st.header("Informazioni")
    st.write("Modello: Random Forest bilanciata, calibrata mediante sigmoid.")
    st.write("Validazione esterna: ligandi completamente separati dal Development Set.")
    st.warning("Le proposte sono ipotesi computazionali e devono essere verificate chimicamente e sperimentalmente.")
    st.caption("Versione applicativa 1.0 · modello v6.3 · motore v7.0")

tabs = st.tabs(["🔮 Predizione", "📚 Knowledge Engine", "🎯 Optimizer", "📊 Validazione", "ℹ️ Metodo e limiti"])

with tabs[0]:
    st.subheader("Prediction Engine")
    st.write("Inserisci le condizioni sperimentali per stimare l'esito e verificare il dominio di applicabilità.")
    row = input_form(engine, "pred_")
    if st.button("Calcola previsione", type="primary", use_container_width=True):
        result = engine.predict(row)
        a,b,c,d = st.columns(4)
        a.metric("Esito previsto", CLASS_SHORT[result.predicted_class])
        b.metric("P(MOF cristallino)", f"{result.probabilities[2]*100:.1f}%")
        c.metric("Affidabilità AD", f"{result.domain_score*100:.0f}%")
        d.metric("Dominio", result.domain_label)
        st.bar_chart(probability_table(result.probabilities, LABELS))
        st.caption(f"Distanza media dai 5 vicini più prossimi: {result.distance:.3f}")
        if result.unseen_categories:
            st.warning("Categorie non osservate nel training: " + ", ".join(result.unseen_categories))
        sensitivity = engine.local_sensitivity(row)
        sensitivity["Variazione P(cristallino)"] = sensitivity["Variazione P(cristallino)"].map(lambda x: f"{x:+.3f}")
        st.subheader("Spiegazione locale per analisi di sensibilità")
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        st.info("Questa analisi sostituisce una feature alla volta con il riferimento del database; non rappresenta causalità.")

with tabs[1]:
    st.subheader("Knowledge Engine")
    c1,c2,c3 = st.columns(3)
    ligand = c1.selectbox("Legante specifico", ["Tutti"] + engine.options("Legante"))
    family = c2.selectbox("Famiglia", ["Tutte"] + engine.options("Famiglia_Legante"))
    metal = c3.selectbox("Metallo", ["Tutti"] + engine.options("Metallo"))
    sub = engine.knowledge_filter(ligand, family, metal)
    if sub.empty:
        st.info("Nessuna sintesi corrisponde ai filtri selezionati.")
    else:
        counts = sub["Esito_ML"].value_counts().reindex([0,1,2], fill_value=0)
        a,b,c,d = st.columns(4)
        a.metric("Sintesi", len(sub)); b.metric("Ligandi", sub["Legante"].nunique())
        c.metric("Tasso cristallino", f"{(sub['Esito_ML']==2).mean()*100:.1f}%"); d.metric("Metalli", sub["Metallo"].nunique())
        st.bar_chart(pd.DataFrame({"Numero": counts.values}, index=[LABELS[i] for i in counts.index]))
        numeric_cols = ["Temperatura_C","Tempo_ore","Rapporto_LM","Volume solvente"]
        st.subheader("Intervalli sperimentali osservati")
        st.dataframe(sub[numeric_cols].describe().T, use_container_width=True)
        show_cols = ["ID","Legante","Metallo","Sale_Metallico","Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM","Esito_ML"]
        st.subheader("Sintesi recuperate")
        st.dataframe(sub[show_cols].head(500), use_container_width=True, hide_index=True)
        st.download_button("Scarica risultati filtrati", sub.to_csv(index=False).encode("utf-8"), "knowledge_results.csv", "text/csv")

with tabs[2]:
    st.subheader("Optimizer Engine")
    st.write("Genera combinazioni vicine alla condizione iniziale e le ordina usando probabilità di cristallinità e dominio di applicabilità.")
    base = input_form(engine, "opt_")
    c1,c2,c3 = st.columns(3)
    n = c1.slider("Numero di proposte", 5, 30, 10)
    vary_solvent = c2.checkbox("Varia solvente", True)
    vary_additive = c3.checkbox("Varia additivo", True)
    if st.button("Ottimizza condizioni", type="primary", use_container_width=True):
        with st.spinner("Valutazione delle combinazioni..."):
            result = engine.optimize(base, n, vary_solvent, vary_additive)
        display = result[["Solvente","Additivo_Colinker","Temperatura_C","Tempo_ore","Rapporto_LM","P_cristallino","AD_score","Dominio","Ranking_score"]].copy()
        for col in ["P_cristallino","AD_score","Ranking_score"]:
            display[col] = (display[col]*100).round(1)
        display = display.rename(columns={"P_cristallino":"P cristallino (%)","AD_score":"AD (%)","Ranking_score":"Score (%)"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Scarica proposte complete", result.to_csv(index=False).encode("utf-8"), "MOF_optimized_conditions.csv", "text/csv")
        st.warning("Controllare sempre solubilità, stabilità termica, compatibilità dei reagenti e sicurezza prima della prova sperimentale.")

with tabs[3]:
    st.subheader("Validazione esterna definitiva")
    m = assets.external_metrics
    a,b,c,d = st.columns(4)
    a.metric("Macro-F1", f"{m['Macro_F1']:.3f}"); b.metric("Balanced accuracy", f"{m['Balanced_Accuracy']:.3f}")
    c.metric("MCC", f"{m['MCC']:.3f}"); d.metric("Log-loss", f"{m['Log_Loss']:.3f}")
    e,f,g = st.columns(3)
    e.metric("Accuracy", f"{m['Accuracy']:.3f}"); f.metric("Brier score", f"{m['Multiclass_Brier']:.3f}"); g.metric("ECE", f"{m['ECE_10_bins']:.3f}")
    st.subheader("Metriche per classe")
    st.dataframe(assets.class_metrics, use_container_width=True, hide_index=True)
    st.subheader("Matrice di confusione")
    st.dataframe(assets.confusion_matrix, use_container_width=True)
    st.caption("External Test: 269 sintesi con ligandi non presenti nel Development Set.")

with tabs[4]:
    st.subheader("Metodo")
    st.markdown("""
- Classificazione a tre classi: fallimento, prodotto amorfo/scarsamente cristallino, MOF cristallino.
- Random Forest con pesi di classe e calibrazione sigmoid.
- Preprocessing incorporato nel modello; `Procedura_Sintetica` esclusa per evitare leakage.
- Validazione raggruppata per legante e successivo External Test su ligandi mai utilizzati nello sviluppo.
- Applicability Domain basato sulla distanza dai vicini nel medesimo spazio preprocessato.
- L'Optimizer esplora perturbazioni locali e non esegue una simulazione chimico-fisica.
""")
    st.subheader("Limiti")
    st.markdown("""
- La qualità dipende dalla copertura e dalla correttezza del database di letteratura.
- Una probabilità elevata non garantisce cristallinità o purezza di fase.
- Condizioni fuori dominio sono da considerare esplorative.
- L'analisi di sensibilità locale non dimostra relazioni causali.
- Il modello non verifica automaticamente rischi, incompatibilità o fattibilità pratica.
""")
