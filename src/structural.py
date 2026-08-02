"""RDKit-based ligand representation used by the v10.5 predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.base import BaseEstimator, TransformerMixin

MORGAN_BITS = 1024
DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotBonds",
    "AromaticRings", "FractionCSP3", "FormalCharge", "HeavyAtoms",
)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def scaffold_smiles(smiles: str) -> str | None:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    value = Chem.MolToSmiles(scaffold, canonical=True)
    return value or Chem.MolToSmiles(mol, canonical=True)


def ligand_vector(smiles: str, radius: int = 2, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError("A valid ligand SMILES is required for structural prediction.")
    fp = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    bits = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, bits)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol), Chem.GetFormalCharge(mol), mol.GetNumHeavyAtoms(),
    ], dtype=np.float32)
    return np.concatenate([bits, desc])


def structural_matrix(smiles_values) -> np.ndarray:
    return np.vstack([ligand_vector(s) for s in smiles_values])


class LigandStructureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible Morgan + descriptor transformer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        values = np.asarray(X).reshape(-1)
        return structural_matrix(values)


def add_structure_columns(df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    cols = ["Legante", "Ligand_SMILES", "Ligand_InChIKey", "Structure_Status", "Structure_Source"]
    lookup = registry[[c for c in cols if c in registry]].drop_duplicates("Legante")
    out = df.drop(columns=[c for c in cols[1:] if c in df], errors="ignore").merge(lookup, on="Legante", how="left")
    out["Ligand_Scaffold"] = out["Ligand_SMILES"].map(scaffold_smiles)
    return out
