"""Water/nonpolar-solvent miscibility screen for solvent mixtures.

Independent of the frozen predictor, in the same spirit as
src/solubility.py, src/vessel_conditions.py and src/modulator_chemistry.py:
a chemistry-grounded heuristic layered on top of the optimizer and the
single-prediction view, not a trained model feature.

Scope, deliberately narrow: the optimizer's own solvent pool only ever
proposes mixture strings that already appear verbatim in the historical
database (see optimizer._pool()), and every such mixture that exists there
today is a chemically ordinary, fully-miscible combination (checked during
development: DMF/H2O, H2O/EtOH, MeOH/Toluene, etc. are all genuinely
miscible). The realistic risk is instead the free-text "Solvente" field in
the single-prediction form, and any future allowed_solvents constraint,
where nothing stops someone from entering a mixture that would separate
into two phases in real life, e.g. "Water/Toluene".

This module therefore checks for exactly one well-established, high-
confidence case: water combined with a solvent it is not freely miscible
with in all proportions (the classic biphasic-with-water solvents from
general chemistry -- toluene, hexane/heptane, dichloromethane, chloroform;
ethyl acetate is flagged separately as only partially miscible). It does
NOT attempt a full pairwise miscibility matrix across every solvent this
app knows about: most of the remaining pairs (DMF, DMA, DMSO, NMP,
acetonitrile, the alcohols, acetone, THF, dioxane, ethylene glycol,
pyridine) are miscible with each other and with water in essentially all
the combinations this app's solvent list can produce, so asserting
"miscible" for an untested pair would overstate what is actually known
here -- an unmatched pair returns "not flagged", not "confirmed miscible".
"""
from __future__ import annotations

import re

# Solvents not freely miscible with water in all proportions (classic
# biphasic-with-water solvents from general/analytical chemistry).
WATER_IMMISCIBLE = {
    "toluene", "hexane", "heptane", "dichloromethane", "ch2cl2", "dcm", "chloroform",
}
# Limited mutual solubility with water (a two-phase system still typically
# forms, though with more mutual uptake than the fully immiscible solvents
# above) -- flagged with a softer message.
WATER_PARTIALLY_MISCIBLE = {"ethyl acetate"}


def _split_mixture(solvent_text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[/+;,:]", str(solvent_text or "")) if c.strip()]


def _is_water(component: str) -> bool:
    comp = component.strip().casefold()
    return comp in ("water", "h2o") or comp.startswith("water") or comp.startswith("h2o")


def miscibility_check(solvent_text: str) -> dict:
    """Flag a water + classically-immiscible-solvent mixture.

    Returns a dict with 'checked' (bool, True only when there were at least
    two components to compare), 'flag' (None / 'immiscible' / 'partially_miscible'),
    and 'note' (str). A single solvent, or a mixture this narrow check has no
    concern about, returns checked=True (there was something to look at) but
    flag=None -- this is NOT a claim that the pair is confirmed miscible,
    only that it is not one of the specific known problem pairs checked here.
    """
    components = _split_mixture(solvent_text)
    if len(components) < 2:
        return {"checked": False, "flag": None, "note": "Single solvent; no mixture to screen."}

    casefolded = [c.casefold() for c in components]
    has_water = any(_is_water(c) for c in components)
    if not has_water:
        return {"checked": True, "flag": None,
                "note": "No water component detected; this narrow screen only checks water/nonpolar-solvent pairs."}

    immiscible_hits = [c for c in components if any(k in c.casefold() for k in WATER_IMMISCIBLE)]
    partial_hits = [c for c in components if any(k in c.casefold() for k in WATER_PARTIALLY_MISCIBLE)]

    if immiscible_hits:
        return {"checked": True, "flag": "immiscible",
                "note": (f"Water and {', '.join(immiscible_hits)} are not freely miscible in all proportions "
                         f"(classic biphasic pair): this mixture would likely separate into two liquid phases "
                         f"rather than forming a single homogeneous reaction medium.")}
    if partial_hits:
        return {"checked": True, "flag": "partially_miscible",
                "note": (f"Water and {', '.join(partial_hits)} have only limited mutual solubility: this "
                         f"mixture may separate into two phases depending on the ratio used.")}
    return {"checked": True, "flag": None,
            "note": "No known water-immiscibility issue detected for this mixture (narrow screen; not a full miscibility check)."}
