import json
import html
from pathlib import Path
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from src.chem import METALS, FAMILIES, COUNTERIONS, infer_family, precursor_formula, parse_salt
from src.engine import predict, applicability, similar, optimize, explain_prediction, DB
from src.resolver import resolve_ligand
from src.literature import search_literature

# Temporary Tavily deployment key. Replace this value when rotating the key.
TAVILY_DEPLOYMENT_KEY = "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"

APP_VERSION = "9.5.0"

st.set_page_config(page_title="MOF Synthesis Assistant", page_icon="🧪", layout="wide")
st.title("🧪 MOF Synthesis Assistant v9.5")
st.caption("Version 9.5.0 · Refined ligand identity card and integrated literature search")
st.caption("Prediction, intuitive condition diagnosis, contextual optimization and recent literature search")
page = st.sidebar.radio("Module", ["Predict synthesis", "Literature search", "Model validation", "About"])


def _format_formula(formula):
    if not formula:
        return "Unavailable"
    import re
    return re.sub(r"(\d+)", r"<sub>\1</sub>", html.escape(str(formula)))


def _molecule_image(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Draw.rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=(430, 300), kekulize=True)


def _descriptor_value(descriptors, key, suffix=""):
    value = (descriptors or {}).get(key)
    return "—" if value is None else f"{value}{suffix}"


def _resolved_identity(prefix, typed_ligand):
    state_key = prefix + "resolved_ligand"
    if st.button("Resolve ligand", key=prefix + "resolve", disabled=not typed_ligand, type="secondary"):
        with st.spinner("Resolving through local parsing, NCI Cactus and PubChem..."):
            st.session_state[state_key] = resolve_ligand(typed_ligand)

    result = st.session_state.get(state_key)
    if result and result.get("query") != typed_ligand:
        result = None
    if not result:
        return None

    if not result.get("success"):
        st.warning(result.get("message", "Ligand resolution failed."))
        return result

    if result.get("ambiguity_warning"):
        st.warning(result["ambiguity_warning"])

    title = result.get("title") or result.get("iupac_name") or typed_ligand or "Resolved ligand"
    iupac = result.get("iupac_name") or "Not available"
    formula = _format_formula(result.get("molecular_formula"))
    mw = result.get("molecular_weight")
    mw_text = f"{mw:.2f} g/mol" if isinstance(mw, (int, float)) else "Unavailable"
    inchikey = result.get("inchikey") or "Unavailable"
    source = result.get("source") or "Chemical resolver"
    descriptors = result.get("descriptors") or {}

    card = f"""
    <div style="border:1px solid #d9e2ec;border-radius:16px;padding:20px 22px;margin:8px 0 16px 0;
                background:linear-gradient(135deg,#ffffff 0%,#f8fbff 100%);box-shadow:0 4px 16px rgba(15,23,42,0.06);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;">
        <div>
          <div style="font-size:0.82rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#2e7d5b;">
            ✓ Ligand successfully identified
          </div>
          <div style="font-size:1.55rem;font-weight:750;color:#172033;margin-top:5px;">{html.escape(str(title))}</div>
          <div style="font-size:0.93rem;color:#5f6b7a;margin-top:4px;max-width:760px;">{html.escape(str(iupac))}</div>
        </div>
        <div style="background:#e9f8ef;color:#24724d;border-radius:999px;padding:7px 12px;font-size:0.82rem;font-weight:700;">
          Resolved identity
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:18px;">
        <div style="background:#fff;border:1px solid #e7edf3;border-radius:11px;padding:12px;">
          <div style="font-size:.75rem;color:#718096;text-transform:uppercase;font-weight:700;">Formula</div>
          <div style="font-size:1.08rem;font-weight:700;color:#1f2937;margin-top:4px;">{formula}</div>
        </div>
        <div style="background:#fff;border:1px solid #e7edf3;border-radius:11px;padding:12px;">
          <div style="font-size:.75rem;color:#718096;text-transform:uppercase;font-weight:700;">Molecular weight</div>
          <div style="font-size:1.08rem;font-weight:700;color:#1f2937;margin-top:4px;">{html.escape(mw_text)}</div>
        </div>
        <div style="background:#fff;border:1px solid #e7edf3;border-radius:11px;padding:12px;">
          <div style="font-size:.75rem;color:#718096;text-transform:uppercase;font-weight:700;">Source</div>
          <div style="font-size:.95rem;font-weight:650;color:#1f2937;margin-top:4px;">{html.escape(str(source))}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        image = _molecule_image(result.get("smiles"))
        if image is not None:
            st.image(image, caption="2D molecular structure", use_container_width=True)
        else:
            st.info("A 2D structure preview is not available for this result.")

    with right:
        st.markdown("#### Molecular identifiers")
        id_table = pd.DataFrame([
            ["InChIKey", inchikey],
            ["Canonical SMILES", result.get("smiles") or "Unavailable"],
        ], columns=["Identifier", "Value"])
        st.dataframe(id_table, hide_index=True, use_container_width=True)

        st.markdown("#### Key molecular descriptors")
        descriptor_table = pd.DataFrame([
            ["TPSA", _descriptor_value(descriptors, "TPSA", " Å²")],
            ["LogP", _descriptor_value(descriptors, "LogP")],
            ["H-bond donors", _descriptor_value(descriptors, "HBD")],
            ["H-bond acceptors", _descriptor_value(descriptors, "HBA")],
            ["Aromatic rings", _descriptor_value(descriptors, "AromaticRings")],
            ["Rotatable bonds", _descriptor_value(descriptors, "RotatableBonds")],
        ], columns=["Descriptor", "Value"])
        st.dataframe(descriptor_table, hide_index=True, use_container_width=True)

    with st.expander("View complete resolver details"):
        st.caption(result.get("message") or "Structure resolved and validated.")
        extra = pd.DataFrame([
            ["Input type", result.get("input_type") or "—"],
            ["Normalized query", result.get("normalized_query") or "—"],
            ["Connectivity SMILES", result.get("connectivity_smiles") or "—"],
            ["Exact mass", _descriptor_value(descriptors, "ExactMass")],
            ["Heavy atoms", _descriptor_value(descriptors, "HeavyAtoms")],
            ["Formal charge", _descriptor_value(descriptors, "FormalCharge")],
        ], columns=["Field", "Value"])
        st.dataframe(extra, hide_index=True, use_container_width=True)

    return result


def input_form(prefix="p"):
    st.subheader("Ligand identity")
    mode=st.radio("Input type",["Name / abbreviation / CAS / formula","SMILES"],horizontal=True,key=prefix+"ligand_mode")
    placeholder="e.g. terephthalic acid, H2BDC, 100-21-0 or C8H6O4" if mode.startswith("Name") else "e.g. O=C(O)c1ccc(C(=O)O)cc1"
    ligand=st.text_input("Ligand identifier",key=prefix+"lig",placeholder=placeholder)
    resolved=_resolved_identity(prefix,ligand)
    canonical_title=(resolved.get("title") or resolved.get("iupac_name")) if resolved and resolved.get("success") else None
    inferred=infer_family(" ".join(filter(None,[ligand,canonical_title])))
    fam=st.selectbox("Ligand family",FAMILIES,index=FAMILIES.index(inferred) if inferred in FAMILIES else len(FAMILIES)-1,key=prefix+"fam")
    st.subheader("Metal precursor")
    a,b,c=st.columns(3); metals=sorted(METALS)
    metal=a.selectbox("Metal",metals,index=metals.index("Zn"),key=prefix+"metal")
    oxidation=b.selectbox("Oxidation state",[1,2,3,4,5,6,"unknown"],index=1,key=prefix+"ox")
    counterion=c.selectbox("Counterion / precursor class",COUNTERIONS,key=prefix+"counter")
    hydration=st.number_input("Hydration number",0.0,20.0,0.0,0.5,key=prefix+"hyd")
    suggested=precursor_formula(metal,oxidation if oxidation!="unknown" else 2,counterion,hydration)
    salt=st.text_input("Full metal precursor formula",value=suggested,key=prefix+"salt")
    parsed=parse_salt(salt)
    st.caption(f"Parsed precursor: counterion = {parsed.get('Counterion_Class')}; hydration = {parsed.get('Hydration_Number')}; oxidation state = {parsed.get('Oxidation_State')}")
    st.subheader("Reaction conditions")
    c1,c2,c3=st.columns(3)
    solvent=c1.text_input("Solvent or solvent mixture",value="DMF",key=prefix+"solv")
    additive=c2.text_input("Additive / co-linker",value="Nessuno",key=prefix+"add")
    temp=c3.number_input("Temperature (°C)",20.0,300.0,120.0,key=prefix+"temp")
    c4,c5,c6=st.columns(3)
    hours=c4.number_input("Time (h)",0.1,500.0,24.0,key=prefix+"time")
    mmol_l=c5.number_input("Ligand amount (mmol)",0.0001,100.0,0.1,format="%.4f",key=prefix+"ml")
    mmol_m=c6.number_input("Metal precursor amount (mmol)",0.0001,100.0,0.1,format="%.4f",key=prefix+"mm")
    c7,c8=st.columns(2)
    ratio=c7.number_input("Ligand/metal molar ratio",0.01,100.0,float(mmol_l/mmol_m),key=prefix+"ratio")
    volume=c8.number_input("Solvent volume (mL)",0.0,1000.0,10.0,key=prefix+"vol")
    model_ligand=ligand
    if canonical_title and canonical_title.casefold() not in ligand.casefold(): model_ligand=f"{ligand} | {canonical_title}"
    return {"Legante":model_ligand,"Ligand_User_Input":ligand,"Ligand_SMILES":resolved.get("smiles") if resolved and resolved.get("success") else None,"Ligand_Resolution_Source":resolved.get("source") if resolved else None,"Famiglia_Legante":fam,"Metallo":metal,"Sale_Metallico":salt,"Counterion_Class":counterion,"Hydration_Number":hydration,"Oxidation_State":None if oxidation=="unknown" else oxidation,"Solvente":solvent,"Additivo_Colinker":additive,"Temperatura_C":temp,"Tempo_ore":hours,"mmol_Legante":mmol_l,"mmol_Sale":mmol_m,"Rapporto_LM":ratio,"Volume solvente":volume}


def render_prediction(result):
    values=result['values']; probabilities=result['probabilities']; predicted=result['predicted']; ad=result['ad']
    labels=["Failed/no useful product","Amorphous or uncertain product","Crystalline MOF"]
    pcr=float(probabilities[2])
    if pcr>=0.75: signal="🟢 Very favorable conditions"
    elif pcr>=0.50: signal="🟡 Moderately favorable conditions"
    elif pcr>=0.30: signal="🟠 Challenging synthesis"
    else: signal="🔴 Crystallization unlikely under current conditions"
    st.divider(); st.subheader(signal)
    st.write(f"**Predicted outcome:** {labels[predicted]}")
    c1,c2,c3=st.columns(3)
    c1.metric("Probability of crystalline MOF",f"{pcr:.1%}")
    c2.metric("Applicability domain",ad['label'])
    c3.metric("AD score",f"{ad['score']:.2f}")
    with st.expander("View all class probabilities"):
        st.bar_chart(pd.DataFrame({"Probability":probabilities},index=labels))
    influence=result['influence']
    st.subheader("Why did the model reach this prediction?")
    st.caption("Local sensitivity analysis: each editable condition is varied independently over plausible, data-derived alternatives. It is descriptive, not causal.")
    if not influence.empty:
        chart=influence.set_index('Parameter')[['Influence']]
        st.bar_chart(chart)
        left,right=st.columns(2)
        limiting=influence[influence.Direction=='Limiting'].head(4)
        favorable=influence[influence.Direction=='Favorable'].head(4)
        with left:
            st.markdown("### 🔴 Main limiting factors")
            if limiting.empty: st.write("No strong limiting condition was detected locally.")
            for _,r in limiting.iterrows():
                st.write(f"**{r.Parameter}:** current `{r.Current}` → best tested `{r.Best_alternative}` (up to {r.Best_P_crystalline:.1%} crystalline probability)")
        with right:
            st.markdown("### 🟢 Factors supporting crystallization")
            if favorable.empty: st.write("No strongly favorable condition was isolated locally.")
            for _,r in favorable.iterrows(): st.write(f"**{r.Parameter}:** current value `{r.Current}` is locally favorable.")
    if not ad['ligand_seen']: st.warning("The ligand was not observed exactly in training. Chemical recognition does not remove model extrapolation.")
    if not ad['metal_seen']: st.warning("The selected metal was not observed in training; uncertainty remains high.")
    st.subheader("Next step")
    button_label="Optimize synthesis conditions" if pcr<0.65 else "Explore alternative conditions"
    if st.button(button_label,type="primary",key="context_optimize"):
        with st.spinner("Searching plausible conditions while keeping the metal–ligand identity fixed..."):
            st.session_state['optimization_results']=optimize(values,10)
    out=st.session_state.get('optimization_results')
    if out is not None:
        st.subheader("Recommended conditions")
        best=float(out.iloc[0]['P_Crystalline']) if len(out) else pcr
        a,b,c=st.columns(3); a.metric("Current P(crystalline)",f"{pcr:.1%}"); b.metric("Best proposed",f"{best:.1%}"); c.metric("Expected improvement",f"{best-pcr:+.1%}")
        show_cols=['Temperatura_C','Tempo_ore','Rapporto_LM','Solvente','Additivo_Colinker','P_Crystalline','AD_score','Optimized_score']
        st.dataframe(out[[c for c in show_cols if c in out.columns]],use_container_width=True,hide_index=True)
        st.download_button("Download optimized conditions",out.to_csv(index=False).encode(),"mof_optimized_conditions.csv","text/csv")
    with st.expander("Similar experimental records"):
        st.dataframe(similar(values),use_container_width=True)


if page=="Predict synthesis":
    values=input_form("p")
    if st.button("Run prediction",type="primary"):
        _,probabilities,predicted=predict(values); ad=applicability(values); influence,_=explain_prediction(values)
        st.session_state['prediction_result']={'values':values,'probabilities':probabilities,'predicted':predicted,'ad':ad,'influence':influence}
        st.session_state.pop('optimization_results',None)
    if 'prediction_result' in st.session_state: render_prediction(st.session_state['prediction_result'])
elif page=="Literature search":
    st.subheader("🔎 Recent scientific literature")
    st.write("Search recent articles from selected scholarly publishers and repositories using Tavily.")
    st.caption("Results are restricted to scientific domains, but relevance and bibliographic details should still be verified on the publisher page.")

    with st.form("literature_search_form"):
        query = st.text_input(
            "Keyword or research question",
            placeholder="e.g. bipyrazole MOF oxygen evolution reaction",
        )
        c1, c2, c3 = st.columns(3)
        years_back = c1.selectbox("Publication window", [1, 2, 3, 5, 10], index=3, format_func=lambda x: f"Last {x} year" if x == 1 else f"Last {x} years")
        max_results = c2.slider("Number of articles", 5, 20, 10)
        mof_focus = c3.checkbox("Add MOF context", value=True, help="Adds MOF and synthesis terms to improve materials-science relevance.")
        submitted = st.form_submit_button("Search literature", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a keyword or research question.")
        else:
            with st.spinner("Searching recent scholarly sources..."):
                try:
                    results = search_literature(
                        query,
                        years_back=years_back,
                        max_results=max_results,
                        mof_focus=mof_focus,
                        api_key=TAVILY_DEPLOYMENT_KEY,
                    )
                    st.session_state["literature_results"] = results
                    st.session_state["literature_query"] = query
                except Exception as exc:
                    st.error(f"Literature search failed: {exc}")

    results = st.session_state.get("literature_results", [])
    if results:
        st.success(f"Found {len(results)} selected results for: {st.session_state.get('literature_query', '')}")
        for i, article in enumerate(results, start=1):
            date_text = article.get("published_date") or "Date not supplied by source"
            st.markdown(f"### {i}. [{article['title']}]({article['url']})")
            st.caption(f"{article['source']} · {date_text} · Tavily relevance {article['score']:.2f}")
            if article.get("doi"):
                st.code(article["doi"], language=None)
            if article.get("summary"):
                st.write(article["summary"])
            st.divider()

        export = pd.DataFrame(results)
        st.download_button(
            "Download literature results (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            "mof_literature_results.csv",
            "text/csv",
        )
elif page=="Model validation":
    root=Path(__file__).parent; metrics=json.loads((root/"reports/external_metrics_v8_0.json").read_text())
    st.subheader("Ligand-group external test of the current predictive core")
    st.info("v9.4 updates the user workflow and adds a curated Tavily literature-search interface and local sensitivity display. The frozen predictive core and external validation remain v8.0.")
    st.json(metrics); st.dataframe(pd.read_csv(root/"reports/external_class_metrics_v8_0.csv"),use_container_width=True); st.dataframe(pd.read_csv(root/"reports/external_confusion_matrix_v8_0.csv",index_col=0),use_container_width=True)
else:
    st.markdown("""### Scope and scientific limitations
Version 9.4 integrates prediction and optimization into one workflow and replaces the default technical explanation with an intuitive local-sensitivity summary. The optimizer keeps the selected metal–ligand identity fixed and varies only experimental conditions.

The explanation is **model-based and descriptive, not causal**. Optimized conditions are hypotheses for experimental prioritization, not guarantees of MOF formation. The predictive core remains the frozen v8.0 ensemble; the literature module is a retrieval aid and this interface update does not constitute a new external validation.""")
