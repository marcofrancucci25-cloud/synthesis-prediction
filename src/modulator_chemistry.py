"""Modulator / ligand acidity ("competitive modulation") compatibility check.

Independent of the frozen predictor, in the same spirit as
src/solubility.py and src/vessel_conditions.py: a chemistry-grounded
heuristic layered on top of the optimizer and the single-prediction view,
not a trained model feature.

Background: in modulated MOF synthesis, a monotopic acid ("modulator", e.g.
acetic or benzoic acid) is added in excess to compete with the polytopic
ligand for coordination sites on the metal, slowing nucleation and often
improving crystallinity and crystal size (see e.g. the extensive literature
on modulated synthesis of Zr-MOFs). For competitive modulation to work as
intended, the modulator's acidity needs to be broadly comparable to the
ligand's -- a modulator that is far weaker will not compete effectively,
one that is far stronger can still work (this is a normal, deliberate
strategy for some systems) but changes the mechanism.

This module is weaker evidence than src/solubility.py's ESOL estimate or
src/vessel_conditions.py's boiling points, for two compounding reasons that
are stated here explicitly rather than hidden in a confident-looking score:

1. The ligand side is NOT the specific ligand's actual pKa (no general,
   reliable pKa predictor from SMILES is available in this environment).
   It is a single REPRESENTATIVE literature pKa for the ligand's declared
   family (e.g. "aromatic dicarboxylic acids are typically pKa1 ~3.5-4.5").
   Two ligands in the same family can differ by a full pKa unit or more.
2. "Comparable acidity" as a predictor of modulation effectiveness is
   qualitative guidance from the synthesis literature, not a quantitative
   law -- unlike solubility (thermodynamics) or boiling point (a physical
   constant), there is no single accepted equation to plug numbers into.

For both reasons this check is purely informational: it is never added to
the optimizer's Optimization_score, and its message is phrased as a
question to consider, not a verdict.
"""
from __future__ import annotations

from typing import Optional

from .chem import canonicalize_family

# Representative pKa of the ligand family's coordinating acidic/basic group,
# keyed on the INTERNAL canonical family vocabulary produced by
# chem.canonicalize_family() -- the same normalization build_row() and the
# optimizer already apply, so this table is looked up post-canonicalization
# rather than duplicating a second, easily-drifting public-label mapping.
# Values are literature pKa for typical members of each family (e.g.
# terephthalic acid pKa1=3.51 for aromatic carboxylates; pyrazole pKaH~2.5
# for bipyrazole/pyrazole; imidazole pKaH~6.95 for imidazolates; 1,2,4-triazole
# pKaH~2.4 for triazoles; pyridine pKaH~5.2 for pyridyl donors; phosphonic
# acid pKa1~1-2; sulfonic acids pKa~-1). Categories left out (curcumin/
# beta-diketonate, organometallic variants, "not specified") have no single
# value representative enough to be worth stating.
LIGAND_FAMILY_PKA = {
    "Carbossilati aromatici": 4.0,
    "Carbossilati alifatici": 3.0,
    "Bipyrazole": 2.5,
    "Pyrazole carbossilati": 2.8,
    "Imidazolati": 7.0,
    "Triazole": 2.4,
    "Pyridyl/N-donor": 5.2,
    "Phosphonate": 2.0,
    "Sulfonate": -1.0,
}

# (role, pKa or None). pKa values are standard aqueous pKa of the acid form;
# for strong acids (fully dissociated) a very negative indicative value is
# used only to rank ordering, not as a literal pKa. Matched by substring on
# the casefolded additive text, so both Italian and English spellings work.
MODULATOR_REFERENCE = {
    "acido acetico": ("acid_modulator", 4.76), "acetic acid": ("acid_modulator", 4.76),
    "acido benzoico": ("acid_modulator", 4.20), "benzoic acid": ("acid_modulator", 4.20),
    "acido formico": ("acid_modulator", 3.75), "formic acid": ("acid_modulator", 3.75),
    "tfa": ("acid_modulator", 0.3), "trifluoroacetic": ("acid_modulator", 0.3),
    "monochloroacetic": ("acid_modulator", 2.86),
    "hcl": ("strong_acid", -7.0),
    "hno3": ("strong_acid", -1.4), "nitric acid": ("strong_acid", -1.4),
    "trietilammina": ("base", 10.75), "triethylamine": ("base", 10.75), " tea": ("base", 10.75),
    "ammoni": ("base", 9.25),
    "piridina": ("base", 5.2), "pyridine": ("base", 5.2),
    "bdc": ("co_linker", None), "btc": ("co_linker", None),
    "mesitilene": ("inert_spacer", None), "mesitylene": ("inert_spacer", None),
}

_NO_ADDITIVE_TOKENS = {"nessuno", "none", "n/a", "na", "-", ""}


def _classify_modulator(additive_text: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Return (role, pKa, matched_key) for the additive text, or (None, None, None) if unrecognized."""
    text = str(additive_text or "").strip().casefold()
    if text in _NO_ADDITIVE_TOKENS:
        return "none", None, None
    for key, (role, pka) in MODULATOR_REFERENCE.items():
        if key.strip() in text:
            return role, pka, key
    return None, None, None


def modulator_compatibility(ligand_family: str, additive_text: str, ligand_text: str = "") -> dict:
    """Informational note on modulator/ligand acidity compatibility.

    ``ligand_family`` may be either a public UI family label (src/chem.py
    FAMILIES) or an already-canonicalized internal training-vocabulary
    label -- both are normalized here via the same chem.canonicalize_family()
    used by build_row()/the optimizer, so the lookup always uses the more
    specific internal category (this matters most for "Imidazolate/azolate",
    which covers two internal categories -- imidazolates and triazoles --
    with very different representative pKa; ``ligand_text`` lets that
    disambiguation happen the same way it does during prediction).

    Returns a dict with at least 'checked' (bool) and 'note' (str). See the
    module docstring for why this is deliberately weaker, hedged evidence
    compared to the solubility and vessel-condition checks, and why it is
    never turned into a score penalty.
    """
    role, mod_pka, matched = _classify_modulator(additive_text)

    if role == "none":
        return {"checked": False, "role": "none", "note": "No modulator/additive in use; nothing to compare."}

    if role is None:
        return {"checked": False, "role": None,
                "note": "Additive not in the reference table; acidity compatibility could not be estimated."}

    if role == "co_linker":
        return {"checked": False, "role": role,
                "note": ("This looks like a second full linker (mixed-linker strategy), not a monotopic "
                         "capping modulator -- the competitive-modulation acidity comparison does not apply here.")}

    if role == "inert_spacer":
        return {"checked": False, "role": role,
                "note": "This additive is not an acid/base modulator (space-filling/templating role); acidity comparison does not apply."}

    if role == "base":
        return {"checked": False, "role": role,
                "note": ("This additive is a base, not a competitive acid modulator: its usual role is to help "
                         "deprotonate the ligand and drive coordination, not to compete for binding sites. "
                         "Acidity-matching guidance below does not apply to this role.")}

    ligand_pka = LIGAND_FAMILY_PKA.get(canonicalize_family(ligand_family, ligand_text))
    if ligand_pka is None:
        return {"checked": False, "role": role, "modulator_pka": mod_pka,
                "note": ("No representative pKa is available for this ligand family, so modulator/ligand "
                         "acidity compatibility could not be estimated.")}

    delta = mod_pka - ligand_pka  # positive: modulator is weaker (higher pKa) than the ligand family
    if abs(delta) <= 2.0:
        verdict = "comparable_acidity"
        note = (f"Modulator pKa (~{mod_pka:.1f}) is comparable to the typical pKa for this ligand family "
                 f"(~{ligand_pka:.1f}): a reasonable candidate for competitive modulation, in the sense the "
                 f"synthesis literature generally uses the term.")
    elif delta > 2.0:
        verdict = "modulator_too_weak"
        note = (f"Modulator pKa (~{mod_pka:.1f}) is notably higher (weaker acid) than the typical pKa for this "
                 f"ligand family (~{ligand_pka:.1f}): it may compete only weakly for coordination sites. This is "
                 f"a coarse, family-level estimate -- worth checking against literature for this specific ligand.")
    else:
        verdict = "modulator_much_stronger"
        note = (f"Modulator pKa (~{mod_pka:.1f}) is notably lower (stronger acid) than the typical pKa for this "
                 f"ligand family (~{ligand_pka:.1f}). This can still work deliberately (e.g. TFA modulation is "
                 f"common for some systems) but changes the mechanism from simple site competition; verify against "
                 f"literature for this specific ligand/metal system.")

    return {"checked": True, "role": role, "modulator_pka": mod_pka, "ligand_family_pka": ligand_pka,
            "delta_pka": delta, "verdict": verdict, "note": note}
