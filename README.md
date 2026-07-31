# MOF Synthesis Assistant v8.0

Streamlit platform for literature-informed MOF synthesis prediction and optimization.

## New chemistry-aware inputs

- Free ligand name, abbreviation, molecular formula or SMILES text
- Optional PubChem resolution
- Extended MOF-relevant metal selection
- Oxidation state
- Counterion / precursor class
- Full precursor formula
- Hydration number

The model combines a structured Random Forest with a ligand-text Logistic Regression. Metals are represented both categorically and through periodic descriptors; precursor formulas contribute counterion, oxidation-state and hydration features.

## Deploy on Streamlit Community Cloud

- Repository: this repository
- Branch: `main`
- Main file: `app.py`
- Recommended Python: **3.12 or 3.13**

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important limitation

Free-text ligand input enables name/formula similarity and accepts unseen ligands, but it is not a substitute for a molecular-structure model. Predictions outside the training domain are explicitly flagged.
