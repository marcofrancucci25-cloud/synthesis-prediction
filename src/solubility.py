"""Ligand/solvent compatibility estimation, independent of the frozen predictor.

The frozen classifier (models/MOF_ChemAware_Ensemble_v8_0.joblib) has no
solubility feature of any kind: solvent choice is driven purely by how often
a solvent co-occurs with a given metal in the historical database. This
module adds a second, chemistry-based signal, computed directly from the
ligand's SMILES with RDKit, and combines it with the optimizer's existing
scoring the same way Green_penalty/Speed_penalty already are: as a
transparent, literature-grounded heuristic layered on top of the model, not
a claim about what the model itself has learned.

Two distinct estimates are produced, and they are NOT of the same quality:

1. Aqueous solubility (`estimate_logS_water`) uses the ESOL/Delaney (2004)
   QSPR equation, a widely used, peer-reviewed estimator:
       log S = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
   This is a real, citable quantitative estimate, specifically for water.

2. Compatibility with any OTHER solvent (`solvent_polarity_penalty`) has no
   equivalent peer-reviewed per-solvent solubility equation available here.
   It instead compares the ligand's computed lipophilicity (Crippen logP)
   against each solvent's Snyder polarity index P' -- a standard, tabulated
   analytical-chemistry polarity scale, not an invented number -- as a
   coarse "like dissolves like" proxy. This is a heuristic screen intended
   to catch clear mismatches (e.g. a highly lipophilic porphyrin-type
   ligand paired with pure water), not a calibrated solubility prediction.
   It is deliberately weighted less confidently than the water-specific
   ESOL estimate wherever the two disagree.

Both estimates ignore synthesis-time chemistry that can improve real
solubility beyond what the neutral molecule alone suggests: deprotonation
by an added base, elevated temperature, or a co-solvent/modulator. They are
therefore intentionally framed as a caution flag on top of the optimizer's
existing soft-gate architecture (feasibility, applicability domain, family
consistency), never as a hard block on a candidate.
"""
from __future__ import annotations

from typing import Optional
import re

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - environment without rdkit
    _RDKIT_AVAILABLE = False

# Snyder polarity index P' (standard analytical-chemistry solvent polarity
# scale; see L.R. Snyder, J. Chromatogr. Sci. 16 (1978) 223). Water is the
# most polar common synthesis solvent (P' ~ 10.2); alkanes are the least
# polar (P' ~ 0.1). Values are for the pure solvent.
SOLVENT_POLARITY_INDEX = {
    "water": 10.2, "h2o": 10.2,
    "dmso": 7.2,
    "acetonitrile": 5.8, "mecn": 5.8,
    "dmf": 6.4, "def": 6.0, "dma": 6.5, "nmp": 6.7,
    "methanol": 5.1, "meoh": 5.1,
    "ethanol": 4.3, "etoh": 4.3,
    "acetone": 5.1,
    "2-propanol": 3.9, "isopropanol": 3.9, "ipa": 3.9,
    "ethylene glycol": 6.9,
    "dioxane": 4.8,
    "thf": 4.0,
    "ethyl acetate": 4.4,
    "pyridine": 5.3,
    "dichloromethane": 3.1, "ch2cl2": 3.1, "dcm": 3.1,
    "chloroform": 4.1,
    "toluene": 2.4,
    "hexane": 0.1, "heptane": 0.1,
}
_MAX_POLARITY_INDEX = 10.2  # water; used to normalize to a 0-1 scale
_DEFAULT_POLARITY_INDEX = 5.0  # neutral mid-polarity fallback for unlisted solvents


def _split_mixture(solvent_text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[/+;,:]", str(solvent_text or "")) if c.strip()]


def _polarity_index(component: str) -> float:
    comp = component.strip().casefold()
    matches = [v for k, v in SOLVENT_POLARITY_INDEX.items() if k in comp]
    return max(matches) if matches else _DEFAULT_POLARITY_INDEX


def ligand_descriptors(smiles: Optional[str]) -> Optional[dict]:
    """Compute the RDKit descriptors ESOL needs, or None if unavailable/invalid."""
    if not smiles or not _RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return None
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "aromatic_proportion": float(aromatic_atoms) / float(heavy),
    }


def estimate_logS_water(smiles: Optional[str]) -> Optional[float]:
    """ESOL (Delaney, 2004) estimated aqueous solubility, log10(mol/L).

    Returns None if the SMILES cannot be parsed or RDKit is unavailable.
    As a rough reading guide: logS > -2 is generally freely/very soluble,
    -2 to -4 is moderately soluble, below -4 is poorly soluble in water.
    """
    d = ligand_descriptors(smiles)
    if d is None:
        return None
    return 0.16 - 0.63 * d["clogp"] - 0.0062 * d["mw"] + 0.066 * d["rotatable_bonds"] - 0.74 * d["aromatic_proportion"]


def water_solubility_flag(logS: Optional[float]) -> Optional[str]:
    if logS is None:
        return None
    if logS >= -2.0:
        return "likely soluble"
    if logS >= -4.0:
        return "moderately soluble"
    return "likely poorly soluble"


def solvent_polarity_penalty(smiles: Optional[str], solvent_text: str) -> float:
    """Coarse 0-1 mismatch penalty between ligand lipophilicity and solvent polarity.

    0 = no evident mismatch (or nothing to check against); 1 = strong
    mismatch (a strongly lipophilic ligand in the most polar solvent, or
    vice versa). This is the fallback heuristic used for every solvent that
    is not water; see module docstring for why it is weaker evidence than
    the ESOL estimate.
    """
    d = ligand_descriptors(smiles)
    if d is None:
        return 0.0
    # Normalize clogP to 0 (hydrophilic, clogP<=-2) .. 1 (lipophilic, clogP>=8).
    hydrophobicity = min(max((d["clogp"] + 2.0) / 10.0, 0.0), 1.0)
    components = _split_mixture(solvent_text)
    if not components:
        return 0.0
    penalties = []
    for comp in components:
        polarity = min(_polarity_index(comp) / _MAX_POLARITY_INDEX, 1.0)
        penalties.append(hydrophobicity * polarity)
    return float(sum(penalties) / len(penalties))


def solubility_penalty(smiles: Optional[str], solvent_text: str) -> float:
    """Combined 0-1 penalty used by the optimizer's scoring function.

    For water specifically, the real ESOL estimate dominates (it is an
    actual quantitative solubility estimate). For every other solvent, or
    when no SMILES is available, only the coarser polarity heuristic
    applies (and is 0.0, i.e. no penalty, when there is no ligand structure
    to check against at all -- absence of evidence is not evidence of a
    problem).
    """
    components = _split_mixture(solvent_text)
    has_water = any(_polarity_index(c) == SOLVENT_POLARITY_INDEX["water"] for c in components) if components else False
    polarity_penalty = solvent_polarity_penalty(smiles, solvent_text)
    if has_water:
        logS = estimate_logS_water(smiles)
        if logS is not None:
            # Map logS onto 0-1: >=-2 -> 0.0 penalty, <=-6 -> 1.0 penalty.
            esol_penalty = min(max((-2.0 - logS) / 4.0, 0.0), 1.0)
            # The real ESOL number, where available, is trusted more than
            # the coarse polarity heuristic for the water-specific case.
            return float(0.75 * esol_penalty + 0.25 * polarity_penalty)
    return polarity_penalty


def describe(smiles: Optional[str], solvent_text: str) -> dict:
    """Human-readable summary for UI display / logging."""
    logS = estimate_logS_water(smiles)
    return {
        "logS_water": logS,
        "water_solubility_flag": water_solubility_flag(logS),
        "solubility_penalty": solubility_penalty(smiles, solvent_text),
        "rdkit_available": _RDKIT_AVAILABLE,
        "smiles_resolved": bool(ligand_descriptors(smiles)),
    }
