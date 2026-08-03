import json
import html
from pathlib import Path
import pandas as pd
import streamlit as st
from rdkit import Chem
from src.chem import METALS, FAMILIES, COUNTERIONS, infer_family, precursor_formula, parse_salt, hsab_acid_class
import src.engine as engine

predict = engine.predict
applicability = engine.applicability
prediction_validity = engine.prediction_validity
similar = engine.similar
explain_prediction = engine.explain_prediction
DB = engine.DB

def verified_precedents(values):
    """Compatibility-safe evidence loader for mixed Streamlit deployments.

    Git-backed deployments can briefly expose a new app.py together with an
    older cached engine.py.  In that window the optional evidence panel is
    disabled instead of preventing the entire application from starting.
    """
    fn = getattr(engine, "verified_precedents", None)
    return fn(values) if callable(fn) else pd.DataFrame()

def known_mof_matches(values):
    """Compatibility-safe curated literature matcher."""
    fn = getattr(engine, "known_mof_matches", None)
    return fn(values) if callable(fn) else pd.DataFrame()

def optimize_joint(*args, **kwargs):
    """Compatibility-safe joint optimizer loader.

    Keeps the app online even if Streamlit briefly serves a mixed Git revision
    during deployment. The preferred implementation is engine.optimize_joint;
    the older engine.optimize wrapper is used only as a temporary fallback.
    """
    fn = getattr(engine, "optimize_joint", None)
    if fn is not None:
        return fn(*args, **kwargs)
    legacy = getattr(engine, "optimize", None)
    if legacy is None:
        raise RuntimeError(
            "Joint optimizer module is unavailable. Replace src/engine.py and "
            "src/optimizer.py with the files from the same release."
        )
    values = args[0] if args else kwargs.get("values")
    top_n = kwargs.get("top_n", 10)
    result = legacy(values, top_n=top_n)
    return result, {
        "compatibility_fallback": True,
        "message": "Legacy optimizer fallback used; upload the complete v10.0.1 src folder."
    }
from src.resolver import resolve_ligand, confirmed_entry
from src.literature import search_literature

APP_VERSION = "10.11.2"

st.set_page_config(page_title="MOF Synthesis Assistant", page_icon="🧪", layout="wide")
st.title("🧪 MOF Synthesis Assistant v10.11.2")
st.caption("Version 10.11.9 · Ligand resolver: relevance filter on the Tavily fallback (rejects unrelated compounds); metal+family-specific numeric bounds; miscibility, pKa, solubility & vessel-condition checks")
st.caption("Prediction evaluates the exact entered conditions. Optimization separately combines three-class risk, successful precedents, feasibility and applicability while keeping only ligand and metal fixed.")
page = st.sidebar.radio("Module", ["Predict synthesis", "Literature search", "About"])


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


def _resolve_ligand_from_input(prefix):
    """Resolve the current ligand input when the user presses Enter.

    Streamlit invokes this callback before rerunning the page, so the resolved
    identity is already available when the interface is rendered again.
    """
    typed_ligand = str(st.session_state.get(prefix + "lig", "")).strip()
    state_key = prefix + "resolved_ligand"
    if not typed_ligand:
        st.session_state.pop(state_key, None)
        return
    cache_json = json.dumps(st.session_state.get("confirmed_ligands_cache", {}), sort_keys=True)
    st.session_state[state_key] = resolve_ligand(typed_ligand, user_cache_json=cache_json)


def _reset_prediction_inputs(prefix="p"):
    """Restore prediction widgets to their declared defaults."""
    widget_suffixes = [
        "ligand_mode", "lig", "resolved_ligand", "candidate_choice",
        "fam", "metal", "ox", "counter", "hyd", "salt",
        "solv", "add", "temp", "time", "ml", "mm", "ratio", "vol", "proc",
    ]
    for suffix in widget_suffixes:
        st.session_state.pop(prefix + suffix, None)
    for key in [
        "prediction_result", "optimization_results", "optimization_metadata",
        "joint_objective", "joint_max_temp", "joint_max_time", "joint_samples",
        "joint_keep_precursor", "joint_keep_solvent", "joint_keep_additive",
        "joint_banned",
    ]:
        st.session_state.pop(key, None)


def _reset_optimizer_inputs():
    """Restore optimizer controls without changing the synthesis input form."""
    for key in [
        "optimization_results", "optimization_metadata", "joint_objective",
        "joint_max_temp", "joint_max_time", "joint_samples",
        "joint_keep_precursor", "joint_keep_solvent", "joint_keep_additive",
        "joint_banned",
    ]:
        st.session_state.pop(key, None)


def _reset_literature_inputs():
    """Clear the literature search form and any results currently shown."""
    for key in [
        "literature_results", "literature_query",
        "lit_query", "lit_years", "lit_max_results", "lit_mof_focus",
    ]:
        st.session_state.pop(key, None)


def _resolved_identity(prefix, typed_ligand):
    state_key = prefix + "resolved_ligand"
    if st.button("Resolve ligand", key=prefix + "resolve", disabled=not typed_ligand, type="secondary"):
        with st.spinner("Resolving through curated entries, OPSIN, PubChem and NCI Cactus..."):
            _resolve_ligand_from_input(prefix)

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
    ligand=st.text_input("Ligand identifier",key=prefix+"lig",placeholder=placeholder,on_change=_resolve_ligand_from_input,args=(prefix,),help="Type a name, abbreviation, CAS, formula or SMILES and press Enter to resolve it.")
    resolved=_resolved_identity(prefix,ligand)
    canonical_title=(resolved.get("title") or resolved.get("iupac_name")) if resolved and resolved.get("success") else None
    inferred=infer_family(" ".join(filter(None,[ligand,canonical_title])))
    fam=st.selectbox("Ligand family",FAMILIES,index=FAMILIES.index(inferred) if inferred in FAMILIES else len(FAMILIES)-1,key=prefix+"fam")
    st.subheader("Metal precursor")
    a,b,c=st.columns(3); metals=sorted(m for m in METALS if m!='H')
    selected_oxidation=st.session_state.get(prefix+"ox",2)
    metal=a.selectbox(
        "Metal ion (HSAB class)",metals,index=metals.index("Zn"),key=prefix+"metal",
        format_func=lambda symbol:f"{symbol} — {hsab_acid_class(symbol,selected_oxidation)}",
        help="Indicative Pearson HSAB classification for the selected oxidation state.",
    )
    oxidation=b.selectbox("Oxidation state",[1,2,3,4,5,6,"unknown"],index=1,key=prefix+"ox")
    counterion=c.selectbox("Counterion / precursor class",COUNTERIONS,key=prefix+"counter")
    hydration=st.number_input("Hydration number",0.0,20.0,0.0,0.5,key=prefix+"hyd")
    suggested=precursor_formula(metal,oxidation if oxidation!="unknown" else 2,counterion,hydration)
    salt=st.text_input("Full metal precursor formula",value=suggested,key=prefix+"salt")
    parsed=parse_salt(salt)
    st.caption(f"Parsed precursor: counterion = {parsed.get('Counterion_Class')}; hydration = {parsed.get('Hydration_Number')}; oxidation state = {parsed.get('Oxidation_State')}")
    st.caption(f"HSAB classification: {metal}({oxidation if oxidation!='unknown' else '?'}) is treated as **{hsab_acid_class(metal,oxidation).lower()}**. This is an indicative classification and can depend on oxidation state and coordination environment.")
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
    procedure_options=["Not specified","Solvothermal","Hydrothermal","Room Temperature","Microwave","Sonochemical","Precipitation"]
    procedure=st.selectbox(
        "Synthesis procedure (optional)",procedure_options,key=prefix+"proc",
        help="Matched against successful-synthesis precedents in the optimizer. The core three-class crystallinity model was trained without this field and does not use it.",
    )
    st.caption("Note: the crystallinity prediction above evaluates temperature, time and composition only; it does not distinguish solvothermal, microwave, hydrothermal or other procedures. Selecting a procedure here only sharpens which successful precedents are shown as supporting evidence in the optimizer.")
    model_ligand=ligand
    if canonical_title and canonical_title.casefold() not in ligand.casefold(): model_ligand=f"{ligand} | {canonical_title}"
    result={"Legante":model_ligand,"Ligand_User_Input":ligand,"Ligand_SMILES":resolved.get("smiles") if resolved and resolved.get("success") else None,"Ligand_Resolution_Source":resolved.get("source") if resolved else None,"Famiglia_Legante":fam,"Metallo":metal,"Sale_Metallico":salt,"Counterion_Class":counterion,"Hydration_Number":hydration,"Oxidation_State":None if oxidation=="unknown" else oxidation,"Solvente":solvent,"Additivo_Colinker":additive,"Temperatura_C":temp,"Tempo_ore":hours,"mmol_Legante":mmol_l,"mmol_Sale":mmol_m,"Rapporto_LM":ratio,"Volume solvente":volume}
    # Only set the key when the user actually picked a procedure: an absent
    # key preserves the exact pre-existing "Unknown" fallback used internally
    # by the optimizer's precedent matching, rather than passing an explicit
    # None that would reach the model's preprocessor as a raw null value.
    if procedure!="Not specified":
        result["Procedura_Sintetica"]=procedure
    return result


def render_prediction(result):
    values=result['values']; probabilities=result['probabilities']; predicted=result['predicted']; ad=result['ad']; validity=result.get('validity') or ad.get('validity',{})
    precedents=result.get('precedents',pd.DataFrame())
    known_mofs=result.get('known_mofs',pd.DataFrame())
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
    c3.metric("Prediction validity",validity.get('label','Not assessed'))
    if not known_mofs.empty:
        st.markdown("### 📚 Known metal–linker system in the curated literature")
        st.success(
            "The entered metal–linker combination has one or more DOI-verified "
            "framework precedents in the curated registry."
        )
        for _, reference in known_mofs.iterrows():
            doi = str(reference['Source_DOI'])
            oxidation = reference.get('Reported_Oxidation_State')
            oxidation_text = f"{reference['Metal']}({int(oxidation)})" if pd.notna(oxidation) else reference['Metal']
            st.markdown(
                f"**{reference['MOF_Name']}** · documented pair: `{oxidation_text}` – "
                f"`{reference['Canonical_Ligand_Name']}`  \n"
                f"{reference['Reference_Title']}  \n"
                f"[Open DOI {doi}]({reference['DOI_URL']})"
            )
            if not bool(reference.get('Oxidation_State_Match', True)):
                st.warning(
                    "The oxidation state selected in the form differs from the state "
                    "reported for this framework. The elemental metal–linker pair matches, "
                    "but the ionic specification does not."
                )
        st.caption(
            "This is a literature precedent, not identification of the obtained phase. "
            "A metal–linker pair can form multiple networks; confirm the product by PXRD "
            "comparison with a simulated/reference pattern or by SCXRD."
        )
    strong_precedents=precedents[precedents['Match_Level'].isin(['Exact verified protocol','Close verified protocol'])] if not precedents.empty else precedents
    if not strong_precedents.empty:
        strongest=strong_precedents.iloc[0]
        outcome=strongest['Verified_Outcome']
        source=strongest['Evidence_Source']
        match=strongest['Match_Level']
        if strongest['Outcome_Class']==2:
            st.success(f"Verified experimental evidence: **{outcome}** · {match} · {source}. This evidence is reported separately and takes interpretive priority over an uncertain classifier result.")
        elif strongest['Outcome_Class']==0:
            st.error(f"Verified experimental evidence: **{outcome}** · {match} · {source}. The matching experiment did not yield a useful crystalline product.")
        else:
            st.warning(f"Verified experimental evidence: **{outcome}** · {match} · {source}.")
        if pd.notna(strongest.get('Source_DOI')):
            doi=str(strongest['Source_DOI']).removeprefix('https://doi.org/')
            st.markdown(f"Source: [https://doi.org/{doi}](https://doi.org/{doi})")
        st.caption("The probability above remains the model estimate; experimental evidence is not converted into a fictitious calibrated probability.")
    if not validity.get('reliable', True):
        st.error("The entered conditions are outside or near the edge of the experimentally validated range. The numerical probabilities are shown for transparency, but should not be interpreted as reliable success estimates.")
        for issue in validity.get('issues', [])[:6]:
            st.write(f"- {issue}")
    else:
        st.caption(f"Validated-range score: {validity.get('score',1.0):.2f} · Applicability score: {ad['score']:.2f}")
    with st.expander("View all class probabilities"):
        st.bar_chart(pd.DataFrame({"Probability":probabilities},index=labels))
    influence=result['influence']
    st.subheader("Why did the model reach this prediction?")
    st.caption(
        "Controlled local model sensitivity—not a synthesis recommendation. Only numerically reliable "
        "alternatives are evaluated. For L:M changes, ligand and metal amounts are rebalanced while "
        "keeping the total precursor amount constant. Use the joint optimizer below for complete proposals."
    )
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
                st.write(f"**{r.Parameter}:** current `{r.Current}` → best supported perturbation `{r.Best_alternative}` (model response up to {r.Best_P_crystalline:.1%} crystalline probability)")
                if r.Field=='Rapporto_LM' and pd.notna(r.get('Best_Alternative_Detail')):
                    st.caption(str(r.Best_Alternative_Detail))
        with right:
            st.markdown("### 🟢 Factors supporting crystallization")
            if favorable.empty: st.write("No strongly favorable condition was isolated locally.")
            for _,r in favorable.iterrows(): st.write(f"**{r.Parameter}:** current value `{r.Current}` is locally favorable.")
    if not ad['ligand_seen']: st.warning("The ligand was not observed exactly in training. Chemical recognition does not remove model extrapolation.")
    if not ad['metal_seen']: st.warning("The selected metal was not observed in training; uncertainty remains high.")
    if ad.get('family_mismatch'): st.warning(f"Declared ligand family (\"{ad.get('declared_family')}\") does not match the family inferred from the ligand name (\"{ad.get('inferred_family')}\"). This is a model input and can change the prediction; verify the selection is intentional.")
    solubility=ad.get('solubility') or {}
    if not solubility.get('smiles_resolved'):
        st.info("Ligand/solvent solubility could not be screened (no resolved ligand structure). Verify solubility manually before selecting a solvent.")
    elif solubility.get('water_solubility_flag')=='likely poorly soluble' and 'water' in str(values.get('Solvente','')).casefold():
        st.warning(f"The selected solvent (water) is estimated to be a poor match for this ligand (ESOL estimated log S ≈ {solubility.get('logS_water'):.1f}, water). This is a computed screening estimate (RDKit/ESOL), not a lab measurement — verify experimentally, especially for rigid symmetric aromatic acids where this estimate is known to be optimistic.")
    elif solubility.get('solubility_penalty',0) > 0.5:
        st.warning("The selected solvent is estimated to be a chemically poor match for this ligand's polarity (computed screening estimate, not a lab measurement). Consider a less/more polar alternative or verify solubility experimentally.")
    vessel=ad.get('vessel') or {}
    if vessel.get('requires_sealed_vessel'):
        st.info(f"Vessel note: {vessel.get('note')}")
    if ad.get('vessel_mismatch'):
        st.warning(f"Declared synthesis procedure (\"{values.get('Procedura_Sintetica')}\") is inconsistent with the entered temperature: {vessel.get('note')} Verify this is intentional.")
    modulator=ad.get('modulator') or {}
    if modulator.get('checked'):
        st.info(f"Modulator note: {modulator.get('note')}")
    elif modulator.get('role') not in (None, 'none'):
        st.caption(f"Modulator note: {modulator.get('note')}")
    miscibility=ad.get('miscibility') or {}
    if miscibility.get('flag')=='immiscible':
        st.warning(f"Solvent mixture note: {miscibility.get('note')}")
    elif miscibility.get('flag')=='partially_miscible':
        st.info(f"Solvent mixture note: {miscibility.get('note')}")
    st.subheader("Hybrid joint synthesis optimizer")
    st.caption("Prediction and optimization are separate. The optimizer keeps only ligand and metal fixed, generates coherent condition sets from successful precedents, explores new combinations, and re-scores every proposal with the balanced three-class predictor.")
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
        optimize_col,reset_col=st.columns(2)
        with optimize_col:
            if st.button(button_label,type="primary",key="context_optimize",use_container_width=True):
                constraints={
                    "max_temperature":max_temp,"max_time":max_time,
                    "keep_precursor":keep_precursor,"keep_solvent":keep_solvent,"keep_additive":keep_additive,
                    "banned_solvents":[x.strip() for x in banned_text.split(',') if x.strip()],
                }
                with st.spinner("Combining successful synthesis templates with broad multivariable exploration, then evaluating risk, feasibility and applicability..."):
                    try:
                        out,meta=optimize_joint(values,objective=objective,n_samples=samples,top_n=12,constraints=constraints)
                        st.session_state['optimization_results']=out
                        st.session_state['optimization_metadata']=meta
                    except Exception as exc:
                        st.error(f"Joint optimization failed: {exc}")
        with reset_col:
            st.button(
                "Reset optimizer fields",on_click=_reset_optimizer_inputs,
                key="reset_optimizer",use_container_width=True,
            )
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
        n_metal_rows=(meta or {}).get('metal_specific_evidence_rows',0)
        n_metal_family_rows=(meta or {}).get('metal_family_specific_evidence_rows',0)
        if n_metal_family_rows >= 10:
            st.caption(f"Numeric ranges (temperature/time/ratio/volume/hydration/oxidation state) are informed by {n_metal_family_rows} records for this specific metal+ligand-family combination, not just the dataset-wide distribution.")
        elif n_metal_rows < 10:
            st.caption(f"Note: only {n_metal_rows} metal-specific evidence row(s) were available for this metal, so temperature/time/ratio/volume/hydration/oxidation-state ranges fall back to the dataset-wide distribution rather than this metal's own typical conditions.")
        for warning_text in (meta or {}).get('warnings', []):
            st.warning(warning_text)
        if "Solubility_penalty" in out.columns and out["Solubility_penalty"].max() > 0.5:
            st.caption("Note: proposals are ranked with an estimated ligand/solvent solubility penalty (RDKit/ESOL-based screening, see the 'Solubility_penalty' column) — lower is better. It is a computed estimate, not a lab measurement.")
        if "Requires_Sealed_Vessel" in out.columns and out["Requires_Sealed_Vessel"].fillna(False).any():
            n_sealed = int(out["Requires_Sealed_Vessel"].fillna(False).sum())
            st.caption(f"Note: {n_sealed} of {len(out)} proposals are at/above the solvent's estimated boiling point ('Requires_Sealed_Vessel' column) — these need a sealed autoclave/vial (solvothermal), not open-vessel reflux. This is informational, not a penalty: solvothermal synthesis is normal and often preferable.")
        if "Modulator_Note" in out.columns and (out["Modulator_Note"] == "modulator_too_weak").any():
            st.caption("Note: for some proposals ('Modulator_Note' column = 'modulator_too_weak'), the chosen additive is estimated to be a weaker acid than typical for this ligand family and may compete only weakly for coordination sites. This is a coarse, family-level pKa estimate, not a penalty on the ranking — see CHANGELOG_FIXES.md for its limitations.")
        if "Miscibility_Flag" in out.columns and (out["Miscibility_Flag"] == "partially_miscible").any():
            st.caption("Note: some proposals use a water/solvent combination with only limited mutual solubility ('Miscibility_Flag' column = 'partially_miscible') and may separate into two phases depending on the exact ratio used.")
        # Keep internal recommendation metadata available for downloads and scientific
        # diagnostics, but present researchers with only the five strongest, directly
        # actionable experimental proposals.
        ranked = out.copy()
        if 'Optimization_score' in ranked.columns:
            ranked = ranked.sort_values('Optimization_score', ascending=False)
        ranked = ranked.head(5).reset_index(drop=True)
        ranked['Rank'] = range(1, len(ranked) + 1)
        show_cols=['Rank','Sale_Metallico','Oxidation_State','Hydration_Number','Solvente','Additivo_Colinker','Temperatura_C','Tempo_ore','mmol_Legante','mmol_Sale','Rapporto_LM','Volume solvente','P_Failed','P_Amorphous','P_Crystalline','AD_score','Feasibility_score','Optimization_score']
        display=ranked[[c for c in show_cols if c in ranked.columns]].copy()
        rename_cols={
            'Sale_Metallico':'Metal precursor', 'Oxidation_State':'Ox. state',
            'Hydration_Number':'Hydration', 'Solvente':'Solvent',
            'Additivo_Colinker':'Additive', 'Temperatura_C':'Temperature (°C)',
            'Tempo_ore':'Time (h)', 'mmol_Legante':'Ligand (mmol)',
            'mmol_Sale':'Metal (mmol)', 'Rapporto_LM':'L:M ratio',
            'Volume solvente':'Solvent volume (mL)', 'P_Failed':'P(failed)',
            'P_Amorphous':'P(amorphous)', 'P_Crystalline':'P(crystalline)',
            'AD_score':'Domain score', 'Feasibility_score':'Feasibility',
            'Optimization_score':'Overall score',
        }
        display=display.rename(columns=rename_cols)
        for col in ['P(failed)','P(amorphous)','P(crystalline)']:
            if col in display: display[col]=display[col].map(lambda x:f"{float(x):.1%}")
        for col in ['Domain score','Feasibility','Overall score']:
            if col in display: display[col]=display[col].map(lambda x:f"{float(x):.3f}")
        for col in ['Temperature (°C)','Time (h)','Ligand (mmol)','Metal (mmol)','L:M ratio','Solvent volume (mL)','Hydration']:
            if col in display: display[col]=pd.to_numeric(display[col],errors='coerce').round(3)
        st.caption("Top five proposals ranked by the overall hybrid optimization score.")
        st.dataframe(display,use_container_width=True,hide_index=True)
        st.download_button("Download joint experimental plan",out.to_csv(index=False).encode(),"mof_joint_optimization_plan.csv","text/csv")
    if not precedents.empty:
        with st.expander("Verified laboratory and literature precedents"):
            evidence_columns=['Evidence_ID','Match_Level','Verified_Outcome','Evidence_Source','Source_DOI','Evidence_Statement','Sale_Metallico','Solvente','Additivo_Colinker','Temperatura_C','Tempo_ore','Rapporto_LM','Volume solvente']
            st.dataframe(precedents[[c for c in evidence_columns if c in precedents]],use_container_width=True,hide_index=True)


if page=="Predict synthesis":
    values=input_form("p")
    action_left, action_right = st.columns([1, 1])
    with action_left:
        run_prediction = st.button("Run prediction", type="primary", use_container_width=True)
    with action_right:
        st.button("Reset parameters", on_click=_reset_prediction_inputs, args=("p",), use_container_width=True)
    if run_prediction:
        _,probabilities,predicted=predict(values); validity=prediction_validity(values); ad=applicability(values); influence,_=explain_prediction(values); precedents=verified_precedents(values); known_mofs=known_mof_matches(values)
        st.session_state['prediction_result']={'values':values,'probabilities':probabilities,'predicted':predicted,'ad':ad,'validity':validity,'influence':influence,'precedents':precedents,'known_mofs':known_mofs}
        st.session_state.pop('optimization_results',None)
        st.session_state.pop('optimization_metadata',None)
    if 'prediction_result' in st.session_state: render_prediction(st.session_state['prediction_result'])
elif page=="Literature search":
    st.subheader("🔎 Recent scientific literature")
    st.write("Search recent articles from selected scholarly publishers and repositories using Tavily.")
    st.caption("Results are restricted to scientific domains, but relevance and bibliographic details should still be verified on the publisher page.")

    query = st.text_input(
        "Keyword or research question",
        placeholder="e.g. bipyrazole MOF oxygen evolution reaction",
        key="lit_query",
    )
    c1, c2, c3 = st.columns(3)
    years_back = c1.selectbox("Publication window", [1, 2, 3, 5, 10], index=3, format_func=lambda x: f"Last {x} year" if x == 1 else f"Last {x} years", key="lit_years")
    max_results = c2.slider("Number of articles", 5, 20, 10, key="lit_max_results")
    mof_focus = c3.checkbox("Add MOF context", value=True, help="Adds MOF and synthesis terms to improve materials-science relevance.", key="lit_mof_focus")

    search_col, reset_col = st.columns(2)
    with search_col:
        submitted = st.button("Search literature", type="primary", use_container_width=True)
    with reset_col:
        st.button("Reset search", on_click=_reset_literature_inputs, key="reset_literature", use_container_width=True)

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
            title = html.escape(str(article.get("title") or "Untitled"))
            url = article.get("url") or ""
            # 1) Title — largest text, clickable.
            st.markdown(f"### {i}. [{title}]({url})" if url else f"### {i}. {title}")
            # 2) DOI, directly under the title, in a smaller font than the title.
            if article.get("doi"):
                st.caption(f"DOI: {html.escape(str(article['doi']))}")
            st.caption(f"{html.escape(str(article.get('source','')))} · {date_text} · Tavily relevance {article['score']:.2f}")
            # 3) Abstract — small font, but a touch larger than the caption
            # lines above so a multi-line abstract stays comfortably readable.
            if article.get("summary"):
                summary = html.escape(str(article["summary"]))
                st.markdown(
                    f"<div style='font-size:0.92rem; line-height:1.5; color:inherit;'>{summary}</div>",
                    unsafe_allow_html=True,
                )
            st.divider()

        export = pd.DataFrame(results)
        st.download_button(
            "Download literature results (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            "mof_literature_results.csv",
            "text/csv",
        )
else:
    st.markdown("""### Scope and scientific limitations
Version 10.11.9 fixes a resolver bug reported by the user: the last-resort Tavily fallback used to discover alternate ligand identifiers could extract a CAS number/abbreviation/quoted string from ANYWHERE in a search result's text, including results not actually about the queried ligand -- presenting a completely unrelated compound as a plausible-looking candidate structure. A relevance filter now requires most of the query's distinctive words to actually appear in a result before any identifier from it is trusted; see CHANGELOG_FIXES.md, including a note on a related, deliberately unresolved gap (a diamino-bipyrazole variant with no verified local structure). It also carries forward the four-tier metal+ligand-family-specific numeric-range cascade (10.11.7-10.11.8) and the four-part optimizer chemistry-screening set: water/nonpolar-solvent miscibility, modulator/ligand pKa compatibility, temperature/solvent/vessel-type physical consistency, and RDKit/ESOL-based ligand/solvent solubility (10.11.3-10.11.6; heuristic layers on top of the optimizer's scoring/output, not trained model features; only solubility is folded into the ranking score). It also carries forward the reworked literature search page (reset control, results reordered as title, DOI, then abstract), canonical public ligand families and common linker aliases before prediction, verified laboratory or literature precedents reported independently from model probabilities, oxidation-state-aware HSAB labels on the metal selector, exact curated metal–linker pairs with DOI-derived article links, and stoichiometrically coherent, domain-filtered local L:M sensitivity. The provenance-first v11 gold dataset contains 179 records, including a 90-experiment HKUST-1 campaign with its continuous PXRD-derived score and ten directly designated high-crystallinity MOF-321/MOF-322 protocols. Training candidates and the external benchmark remain separated at DOI level. Framework names are literature candidates, never structural identification from composition alone.

The explanation is **model-based and descriptive, not causal**. Optimized conditions are hypotheses for experimental prioritization, not guarantees of MOF formation. The predictive core remains the validated frozen v8.0 ensemble: two retraining candidates were rejected because their gain in crystalline recall reduced three-class specificity. The v11 foundation is not activated for training because its eligible records still lack sufficient independent-source and minority-class coverage; verified evidence is therefore displayed separately rather than being converted into an uncalibrated probability.""")
