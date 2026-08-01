import json
import html
from pathlib import Path
import pandas as pd
import streamlit as st
from rdkit import Chem
from src.chem import METALS, FAMILIES, COUNTERIONS, infer_family, precursor_formula, parse_salt
from src.engine import predict, applicability, similar, optimize_joint, explain_prediction, DB
from src.resolver import resolve_ligand, confirmed_entry
from src.literature import search_literature

# Temporary Tavily deployment key. Replace this value when rotating the key.
TAVILY_DEPLOYMENT_KEY = "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"

APP_VERSION = "10.0.0"

st.set_page_config(page_title="MOF Synthesis Assistant", page_icon="🧪", layout="wide")
st.title("🧪 MOF Synthesis Assistant v10.0.0")
st.caption("Version 10.0.0 · Separate prediction engine and joint multivariable synthesis optimizer")
st.caption("Predict entered conditions first; then jointly optimize every model-supported variable while keeping only ligand and metal fixed")
page = st.sidebar.radio("Module", ["Predict synthesis", "Literature search", "Model validation", "About"])


def _format_formula(formula):
    if not formula:
        return "Unavailable"
    import re
    return re.sub(r"(\d+)", r"<sub>\1</sub>", html.escape(str(formula)))


def _molecule_image(smiles):
    """Render a 2D structure when RDKit drawing libraries are available.

    RDKit's drawing backend may depend on Linux X-render libraries that are not
    present in every Streamlit image. Importing it lazily prevents the entire
    application from failing when those optional system libraries are missing.
    """
    if not smiles:
        return None
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Draw.rdDepictor.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=(430, 300), kekulize=True)
    except (ImportError, OSError, RuntimeError):
        return None


def _descriptor_value(descriptors, key, suffix=""):
    value = (descriptors or {}).get(key)
    return "—" if value is None else f"{value}{suffix}"


def _resolved_identity(prefix, typed_ligand):
    state_key = prefix + "resolved_ligand"
    if st.button("Resolve ligand", key=prefix + "resolve", disabled=not typed_ligand, type="secondary"):
        with st.spinner("Resolving through curated entries, OPSIN, PubChem and NCI Cactus..."):
            cache_json = json.dumps(st.session_state.get("confirmed_ligands_cache", {}), sort_keys=True)
            st.session_state[state_key] = resolve_ligand(typed_ligand, user_cache_json=cache_json)

    result = st.session_state.get(state_key)
    if result and result.get("query") != typed_ligand:
        result = None
    if not result:
        return None

    if not result.get("success"):
        st.warning(result.get("message", "Ligand resolution failed."))
        candidates = result.get("candidates") or []
        if result.get("needs_confirmation") and candidates:
            st.markdown("#### Candidate structures")
            st.caption("The resolver did not silently choose an isomer. Compare the candidates and confirm the correct structure.")
            labels = []
            for idx, candidate in enumerate(candidates, start=1):
                name = candidate.get("title") or candidate.get("iupac_name") or f"Candidate {idx}"
                formula = candidate.get("molecular_formula") or "formula unavailable"
                sources = candidate.get("source") or "unknown source"
                score = candidate.get("consensus_score")
                score_text = f" · consensus {score:.0f}/100" if isinstance(score, (int, float)) else ""
                labels.append(f"{idx}. {name} · {formula} · {sources}{score_text}")
            chosen_index = st.selectbox(
                "Select the structure matching your intended ligand",
                range(len(candidates)),
                format_func=lambda i: labels[i],
                key=prefix + "candidate_choice",
            )
            chosen = candidates[chosen_index]
            preview_left, preview_right = st.columns([0.75, 1.25], gap="large")
            with preview_left:
                image = _molecule_image(chosen.get("smiles"))
                if image is not None:
                    st.image(image, caption="Candidate 2D structure", use_container_width=True)
            with preview_right:
                st.write(f"**IUPAC name:** {chosen.get('iupac_name') or 'Unavailable'}")
                st.write(f"**Formula:** {chosen.get('molecular_formula') or 'Unavailable'}")
                st.write(f"**Molecular weight:** {chosen.get('molecular_weight') or 'Unavailable'}")
                st.write(f"**Sources:** {chosen.get('source') or 'Unavailable'}")
                st.code(chosen.get("smiles") or "SMILES unavailable", language=None)
            if st.button("Confirm selected structure", type="primary", key=prefix + "confirm_candidate"):
                confirmed = dict(chosen)
                confirmed["success"] = True
                confirmed["needs_confirmation"] = False
                confirmed["query"] = typed_ligand
                confirmed["normalized_query"] = result.get("normalized_query") or typed_ligand
                confirmed["confidence"] = "user confirmed"
                notes = list(confirmed.get("validation_notes") or [])
                notes.append("Structure selected and confirmed by the user from resolver candidates.")
                confirmed["validation_notes"] = notes
                confirmed["message"] = "Candidate identity confirmed by the user after consensus resolution."
                st.session_state[state_key] = confirmed
                cache = dict(st.session_state.get("confirmed_ligands_cache", {}))
                cache.update(confirmed_entry(typed_ligand, confirmed))
                st.session_state["confirmed_ligands_cache"] = cache
                st.rerun()
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
    confidence = (result.get("confidence") or "medium").capitalize()
    confidence_color = {"High": ("#e9f8ef", "#24724d"), "Medium": ("#fff7df", "#8a6116"), "Low": ("#fff0f0", "#a33a3a"), "User confirmed": ("#e8f1ff", "#2457a6")}.get(confidence, ("#eef2f7", "#475569"))
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
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <div style="background:#e9f8ef;color:#24724d;border-radius:999px;padding:7px 12px;font-size:0.82rem;font-weight:700;">Resolved identity</div>
          <div style="background:{confidence_color[0]};color:{confidence_color[1]};border-radius:999px;padding:7px 12px;font-size:0.82rem;font-weight:700;">{confidence} confidence</div>
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

    if st.session_state.get("confirmed_ligands_cache"):
        cache_payload = json.dumps(st.session_state["confirmed_ligands_cache"], indent=2, ensure_ascii=False)
        st.download_button(
            "Download confirmed ligand cache", cache_payload,
            file_name="confirmed_ligands.json", mime="application/json",
            key=prefix + "download_confirmed_cache",
            help="Add this file to data/confirmed_ligands.json in the repository to preserve confirmed identities across deployments.",
        )

    with st.expander("View complete resolver details"):
        st.caption(result.get("message") or "Structure resolved and validated.")
        notes = result.get("validation_notes") or []
        if notes:
            st.markdown("**Identity validation checks**")
            for note in notes:
                st.markdown(f"- {note}")
        extra = pd.DataFrame([
            ["Input type", result.get("input_type") or "—"],
            ["Resolution confidence", result.get("confidence") or "—"],
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
    st.subheader("Joint synthesis optimizer")
    st.caption("Prediction and optimization are separate: the prediction above evaluates exactly what you entered. The optimizer below keeps only ligand and metal fixed and searches all other variables learned by the frozen model together.")
    with st.expander("Configure multivariable optimization", expanded=pcr < 0.65):
        objective=st.selectbox("Optimization objective",[
            "Maximum crystallinity","Balanced conditions","Conservative optimization","Green synthesis","Fast synthesis"
        ],index=1,key="joint_objective")
        o1,o2,o3=st.columns(3)
        max_temp=o1.number_input("Maximum temperature (°C)",40.0,300.0,180.0,key="joint_max_temp")
        max_time=o2.number_input("Maximum time (h)",0.5,500.0,96.0,key="joint_max_time")
        samples=o3.select_slider("Search depth",options=[750,1500,2500,4000,6000],value=2500,key="joint_samples")
        c1,c2,c3=st.columns(3)
        keep_precursor=c1.checkbox("Keep current metal precursor",value=False,key="joint_keep_precursor")
        keep_solvent=c2.checkbox("Keep current solvent",value=False,key="joint_keep_solvent")
        keep_additive=c3.checkbox("Keep current additive",value=False,key="joint_keep_additive")
        banned_text=st.text_input("Excluded solvents (comma-separated, optional)",placeholder="e.g. DMF, NMP",key="joint_banned")
        st.caption("Variables not learned by the frozen model—such as pH, heating ramp, cooling rate and addition order—are not optimized yet and are explicitly reported as unsupported.")
        button_label="Optimize synthesis conditions" if pcr<0.65 else "Explore joint alternatives"
        if st.button(button_label,type="primary",key="context_optimize"):
            constraints={
                "max_temperature":max_temp,"max_time":max_time,
                "keep_precursor":keep_precursor,"keep_solvent":keep_solvent,"keep_additive":keep_additive,
                "banned_solvents":[x.strip() for x in banned_text.split(',') if x.strip()],
            }
            with st.spinner("Jointly exploring precursor, hydration, oxidation state, solvent, additive, temperature, time, amounts, ratio and volume..."):
                try:
                    out,meta=optimize_joint(values,objective=objective,n_samples=samples,top_n=12,constraints=constraints)
                    st.session_state['optimization_results']=out
                    st.session_state['optimization_metadata']=meta
                except Exception as exc:
                    st.error(f"Joint optimization failed: {exc}")
    out=st.session_state.get('optimization_results')
    meta=st.session_state.get('optimization_metadata')
    if out is not None and len(out):
        st.subheader("Pareto-ranked experimental proposals")
        best=float(out['P_Crystalline'].max())
        a,b,c,d=st.columns(4)
        a.metric("Current P(crystalline)",f"{pcr:.1%}")
        b.metric("Best proposed",f"{best:.1%}")
        c.metric("Expected improvement",f"{best-pcr:+.1%}")
        d.metric("Feasible candidates searched",f"{(meta or {}).get('feasible_candidates',len(out)):,}")
        show_cols=['Rank','Strategy','Sale_Metallico','Oxidation_State','Hydration_Number','Solvente','Additivo_Colinker','Temperatura_C','Tempo_ore','mmol_Legante','mmol_Sale','Rapporto_LM','Volume solvente','P_Failed','P_Amorphous','P_Crystalline','AD_score','Feasibility_score','Pareto_optimal','Optimization_score']
        display=out[[c for c in show_cols if c in out.columns]].copy()
        for col in ['P_Failed','P_Amorphous','P_Crystalline','AD_score','Feasibility_score','Optimization_score']:
            if col in display: display[col]=display[col].map(lambda x:f"{float(x):.1%}" if col.startswith('P_') else f"{float(x):.3f}")
        for col in ['Temperatura_C','Tempo_ore','mmol_Legante','mmol_Sale','Rapporto_LM','Volume solvente','Hydration_Number']:
            if col in display: display[col]=pd.to_numeric(display[col],errors='coerce').round(3)
        st.dataframe(display,use_container_width=True,hide_index=True)
        with st.expander("Scientific scope of this optimization"):
            st.write("**Fixed:**",", ".join((meta or {}).get('fixed_variables',[])))
            st.write("**Jointly optimized:**",", ".join((meta or {}).get('optimized_variables',[])))
            st.write("**Not optimized yet because absent from the frozen model:**",", ".join((meta or {}).get('unsupported_not_optimized',[])))
            st.warning("These are model-ranked experimental hypotheses, not guaranteed synthesis conditions. Confirm experimentally and feed outcomes into a future active-learning dataset.")
        st.download_button("Download joint experimental plan",out.to_csv(index=False).encode(),"mof_joint_optimization_plan.csv","text/csv")
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
    st.info("v10.0 separates exact-condition prediction from joint multivariable optimization. The predictive core and its external validation remain frozen v8.0; optimizer outputs are ranked hypotheses over model-supported variables.")
    st.json(metrics); st.dataframe(pd.read_csv(root/"reports/external_class_metrics_v8_0.csv"),use_container_width=True); st.dataframe(pd.read_csv(root/"reports/external_confusion_matrix_v8_0.csv",index_col=0),use_container_width=True)
else:
    st.markdown("""### Scope and scientific limitations
Version 10.0 separates the Prediction Engine from the Joint Optimization Engine. Prediction evaluates the exact user-entered conditions. Optimization keeps ligand and metal fixed while jointly varying all other variables learned by the frozen predictive core.

The explanation is **model-based and descriptive, not causal**. Optimized conditions are hypotheses for experimental prioritization, not guarantees of MOF formation. The predictive core remains the frozen v8.0 ensemble; the literature module is a retrieval aid and this interface update does not constitute a new external validation.""")
