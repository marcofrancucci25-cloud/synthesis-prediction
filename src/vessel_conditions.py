"""Temperature / solvent / vessel-type physical consistency check.

Independent of the frozen predictor, exactly like src/solubility.py: this is
a chemistry-grounded heuristic layered on top of the optimizer and the
single-prediction view, not a trained model feature.

The optimizer proposes a temperature and a solvent independently of each
other. Nothing today checks whether the pair is physically achievable in an
open vessel at ambient pressure: if the proposed temperature exceeds the
solvent's normal boiling point, the synthesis is still entirely valid
chemistry (solvothermal/hydrothermal syntheses in a sealed autoclave or
sealed vial routinely run above the solvent's normal boiling point -- this
is one of the most common ways to make MOFs), but it is NOT achievable in an
open flask at reflux. This module flags that distinction so the user knows
which glassware/vessel a given proposal actually requires, instead of
silently proposing e.g. 150 degC in water (bp 100 degC) without comment.

This is deliberately informational, not a penalty added to the optimizer's
score: a sealed-vessel requirement is not a defect of a proposal (many of
the best-performing MOF syntheses in the training data are solvothermal),
it is a fact the user needs in order to plan the experiment. The one place
this DOES turn into an actionable flag is when it contradicts something the
user declared explicitly (Procedura_Sintetica = "Room Temperature" while the
proposed/entered temperature already exceeds the solvent's boiling point) --
that is treated the same way as the ligand-family mismatch check in
src/engine.py: a caution about self-contradictory inputs, not a claim that
the chemistry itself is impossible.

Boiling points below are standard normal (1 atm) boiling points from common
reference sources (e.g. CRC Handbook of Chemistry and Physics). For a
solvent mixture, the estimate conservatively uses the LOWEST boiling
component: real mixtures can deviate substantially from ideal behavior
(azeotropes in particular can boil below every pure component), so this is
a caution threshold, not a precise mixture boiling point calculation.
"""
from __future__ import annotations

from typing import Optional
import re

# Normal boiling point in degrees Celsius at 1 atm.
SOLVENT_BOILING_POINT_C = {
    "water": 100.0, "h2o": 100.0,
    "methanol": 64.7, "meoh": 64.7,
    "ethanol": 78.4, "etoh": 78.4,
    "2-propanol": 82.6, "isopropanol": 82.6, "ipa": 82.6,
    "acetone": 56.0,
    "acetonitrile": 82.0, "mecn": 82.0,
    "dmf": 153.0,
    "def": 176.0,
    "dma": 165.0,
    "nmp": 202.0,
    "dmso": 189.0,
    "thf": 66.0,
    "dioxane": 101.1,
    "ethyl acetate": 77.1,
    "ethylene glycol": 197.3,
    "pyridine": 115.2,
    "dichloromethane": 39.6, "ch2cl2": 39.6, "dcm": 39.6,
    "chloroform": 61.2,
    "toluene": 110.6,
    "hexane": 68.7, "heptane": 98.4,
}
_DEFAULT_BP_C = None  # unlisted solvent: no claim is made rather than a guess


def _split_mixture(solvent_text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[/+;,:]", str(solvent_text or "")) if c.strip()]


def _component_bp(component: str) -> Optional[float]:
    comp = component.strip().casefold()
    matches = [v for k, v in SOLVENT_BOILING_POINT_C.items() if k in comp]
    return min(matches) if matches else _DEFAULT_BP_C


def estimate_mixture_bp(solvent_text: str) -> Optional[float]:
    """Conservative (lowest-component) boiling-point estimate for a solvent or mixture."""
    components = _split_mixture(solvent_text)
    bps = [b for b in (_component_bp(c) for c in components) if b is not None]
    return min(bps) if bps else None


def vessel_requirement(solvent_text: str, temperature_c: Optional[float], margin_c: float = 5.0) -> dict:
    """Whether the given temperature is achievable in an open vessel for this solvent.

    Returns a dict with:
      - estimated_bp_c: conservative boiling point estimate, or None if unknown
      - requires_sealed_vessel: True if temperature_c is at/above (bp - margin_c)
      - note: short human-readable explanation
    A None estimated_bp_c means the solvent (or one of its mixture
    components) is not in the reference table; requires_sealed_vessel is
    then also None (unknown), not False, since absence of data is not
    evidence that a sealed vessel is unnecessary.
    """
    bp = estimate_mixture_bp(solvent_text)
    if bp is None or temperature_c is None:
        return {"estimated_bp_c": bp, "requires_sealed_vessel": None,
                "note": "Boiling point unknown for this solvent; vessel requirement could not be checked."}
    try:
        t = float(temperature_c)
    except (TypeError, ValueError):
        return {"estimated_bp_c": bp, "requires_sealed_vessel": None,
                "note": "Temperature could not be parsed; vessel requirement could not be checked."}
    requires_sealed = t >= (bp - margin_c)
    if requires_sealed:
        note = (f"Estimated solvent boiling point ~{bp:.0f} degC at 1 atm: {t:.0f} degC requires a "
                f"sealed vessel (solvothermal/hydrothermal autoclave or sealed vial), not open reflux.")
    else:
        note = f"Estimated solvent boiling point ~{bp:.0f} degC at 1 atm: {t:.0f} degC is achievable at open-vessel reflux."
    return {"estimated_bp_c": bp, "requires_sealed_vessel": requires_sealed, "note": note}
