## Version 9.5.0

Updated ligand identity card with a 2D RDKit structure, clean molecular identifiers, key descriptors, and expandable technical details.

# MOF Synthesis Assistant v9.2

Interactive Streamlit application for literature-aware MOF synthesis prediction and condition prioritization.

## New in v9.2

- ligand input by name, abbreviation, CAS number, molecular formula or SMILES;
- robust resolver chain: local RDKit → MOF aliases → NCI Cactus → PubChem;
- corrected PubChem properties (`SMILES` and `ConnectivitySMILES`);
- RDKit validation and molecular descriptors;
- explicit warning for ambiguous formula-only queries;
- metal precursor formula, counterion, oxidation state and hydration fields;
- predictive, knowledge, applicability-domain and optimizer modules.

## Deploy on Streamlit Community Cloud

- Repository branch: `main`
- Main file: `app.py`
- Recommended Python: 3.12 or 3.13

The application requires outbound HTTPS access to resolve chemical names. Direct SMILES input and the predictive core remain available without the external resolver.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Scientific boundary

The v9.2 release improves chemical identity resolution but retains the frozen v8.0 predictive core. Successful structure resolution does not imply that a ligand is inside the model applicability domain.


## v9.2 interface workflow
Prediction and optimization are now integrated in one page. After a prediction, the app displays a traffic-light assessment, a local sensitivity explanation of favorable and limiting experimental parameters, and a contextual button for generating improved conditions while keeping the selected metal–ligand identity fixed.


## Tavily literature search

Version 9.3 removes the API-key field from the user interface. The key is read only from Streamlit Secrets or the local private secrets file.

The sidebar includes a recent-literature search restricted to selected scientific domains. Configure the API key in Streamlit Cloud under **App settings → Secrets**:

```toml
TAVILY_API_KEY = "tvly-your-key"
```

The application does not store keys entered in the temporary local password field. Search results remain retrieval outputs and should be checked against the original publisher record.


The key is never stored in source code. Configure `TAVILY_API_KEY` only through Streamlit Secrets or an environment variable.

## Ligand consensus resolver (v9.6.0)
Ligand names are resolved using curated linker entries, OPSIN, PubChem and NCI Cactus. The app compares independently obtained structures and only accepts an identity automatically when the evidence is sufficiently consistent. Ambiguous results are displayed as candidate structures for explicit user confirmation.


## v10.0 joint optimization
Prediction evaluates exact input conditions. The optimizer keeps only ligand and metal fixed and jointly searches all other model-supported variables, with feasibility, domain and Pareto controls. See `reports/SCIENTIFIC_UPDATE_v10_0.md`.

## v10.3 — Hybrid successful-synthesis optimizer

Version 10.3 keeps the outcome predictor and optimizer scientifically separate.

- The **Prediction Engine** remains the frozen, balanced three-class model and estimates failed/no useful product, amorphous/uncertain product, and crystalline MOF.
- The **Hybrid Optimization Engine** uses a separate positive synthesis library to generate coherent joint-condition templates and to score successful-synthesis precedent.
- Positive-only records never replace the balanced predictor and are not interpreted as an absolute probability of success.
- Ligand and metal remain fixed; precursor, hydration, oxidation state, solvent, additive, temperature, time, amounts, ratio, and solvent volume are jointly searched.
- Every proposal is ranked using predicted crystallinity, positive precedent, applicability domain, feasibility, and objective-specific penalties.

The included positive library contains 694 deduplicated crystalline-condition records covering 92 ligands and 20 metals. It is designed to be expanded with additional curated literature syntheses in future releases.


## Interface update v10.4.1
- Press Enter in the ligand field to start resolution.
- Reset parameters button restores prediction defaults.
- Optimizer displays only the five strongest actionable proposals.
- Internal recommendation metadata remains in the downloadable CSV and scientific scope panel.


## Prediction validity gate (v10.4.1)

The exact-condition predictor now reports whether numerical inputs lie within the experimentally supported range. Extreme temperature, reaction time, stoichiometry, reagent amount, volume, concentration, or mutually inconsistent ratio/amount inputs trigger an explicit reliability warning. The underlying probability is retained for transparency but is not presented as a validated success estimate outside the supported range. The applicability-domain score now combines chemical identity support with numerical-range support.

## Scientific audit and structural pipeline (v10.5.0)

- Added a provenance-aware ligand structure registry and Morgan/ECFP + RDKit descriptor pipeline.
- Added deterministic label-quality, duplicate-condition and outcome-conflict auditing.
- Added grouped validation by unseen ligand and unseen Bemis-Murcko scaffold.
- The structural model is retained as an experimental artifact and is **not activated in the app** because scaffold-held-out performance did not pass the scientific promotion gate.
- The externally validated v8 predictor remains the production predictor until source-level labels, DOI provenance and PXRD evidence are curated.

## Laboratory evidence integration (v10.6.0)

- Normalized 20 unique DDS laboratory experiments into a flat, provenance-aware schema.
- Integrated 17 eligible records into `knowledge_database_integrated_v10_6.csv`: 3 failed, 1 amorphous/uncertain and 13 crystalline.
- Kept DDS1 in review because its notes conflict on whether a solid existed.
- Kept DDS10,2 and DDS10,3 in a separate in-situ ibuprofen dataset.
- Consolidated one exact duplicate and aggregated repeated positive condition signatures instead of treating them as independent templates.
- Added 10 unique PXRD-supported laboratory conditions to the positive precedent library.
- The original v8 training database and external-test claims remain unchanged. Laboratory evidence is used for precedent retrieval and optimization support only.

## Predictive robustness and verified evidence (v10.7.0)

- Public UI families are translated into the historical model vocabulary before prediction.
- Common linker aliases such as H2BDC, terephthalic acid and benzene-1,4-dicarboxylic acid share one model-facing identity.
- Nine PXRD/XRD-supported literature protocols and the 17 eligible laboratory records form a separate verified-evidence layer.
- Exact and close experimental precedents are shown independently from model probabilities, with DOI provenance where available.
- Three retraining strategies were evaluated and rejected for global deployment because improvements in crystalline recall reduced three-class specificity and balanced accuracy.
- The production predictor therefore remains the frozen v8.0 ensemble; verified evidence is never converted into a fictitious calibrated probability.

### Deployment hotfix v10.7.1

The optional verified-evidence loader now uses a compatibility-safe lookup. If Streamlit briefly combines the new `app.py` with a cached older `src/engine.py`, the evidence panel is temporarily disabled instead of raising an `AttributeError` and stopping the application.

### Interface hotfix v10.7.2

- Removed the public “Scientific scope of this optimization” and “Similar experimental records” expanders.
- Added a reset button dedicated to optimizer controls and results.
- Added oxidation-state-aware Pearson HSAB labels (`Hard acid`, `Borderline acid`, `Soft acid`) to the metal-ion selector.
- Removed hydrogen from the metal-ion list while keeping the underlying model data unchanged.
