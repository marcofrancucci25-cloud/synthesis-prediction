"""Curated literature matches for known metal-linker combinations.

This module intentionally performs deterministic local matching.  It does not
infer a framework name from search-engine snippets and it does not treat a
metal-linker pair as a structural identification.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd

from .chem import canonicalize_ligand_for_model, normalize_ligand


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "known_mof_literature_registry_v10_8.csv"
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)


def _identity_key(value):
    """Return a punctuation-insensitive identity key, not a similarity key."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.replace("′", "'").replace("’", "'").replace("`", "'")
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


LIGAND_ALIASES = {
    "bdc": {
        "h2bdc", "bdc", "terephthalic acid", "terephthalate",
        "benzene-1,4-dicarboxylic acid", "1,4-benzenedicarboxylic acid",
        "1,4-benzenedicarboxylic acid (h2bdc)",
    },
    "btc": {
        "h3btc", "btc", "trimesic acid", "benzene-1,3,5-tricarboxylic acid",
        "1,3,5-benzenetricarboxylic acid", "1,3,5-benzenetricarboxylic acid (h3btc)",
    },
    "dobdc": {
        "h4dobdc", "dobdc", "2,5-dihydroxyterephthalic acid",
        "2,5-dihydroxyterephthalic acid (h4dobdc)",
    },
    "2mim": {"2-methylimidazole", "2-methyl-1h-imidazole", "hmim", "hmemim"},
    "bpz": {
        "bpz", "h2bpz", "4,4-bipyrazole", "4,4'-bipyrazole",
        "4,4'-bi-1h-pyrazole", "4,4'-bipyrazole (h2bpz)",
    },
    "me4bpz": {
        "h2me4bpz", "me4bpz", "3,3',5,5'-tetramethyl-4,4'-bipyrazole",
        "3,3',5,5'-tetramethyl-4,4'-bipyrazole (h2me4bpz)",
    },
    "amino_bpz": {
        "h2bpznh2", "bpznh2", "3-amino-4,4'-bipyrazole",
        "3-amino-4,4'-bipyrazole (h2bpznh2)",
    },
    "nitro_bpz": {
        "h2bpzno2", "bpzno2", "3-nitro-4,4'-bipyrazole",
        "3-nitro-4,4'-bipyrazole (h2bpzno2)",
    },
    "tcpp": {
        "tcpp", "h2tcpp", "tetrakis(4-carboxyphenyl)porphyrin",
        "tcpp (tetrakis(4-carboxyphenyl)porphyrin)",
        "meso-tetra(4-carboxyphenyl)porphine",
        "meso-tetra(4-carboxyphenyl)porphine (h2tcpp)",
    },
    "pzvdc": {
        "pzvdc", "h2pzvdc", "pyrazine vinyl dicarboxylic acid",
        "pyrazine vinyl dicarboxylate",
    },
    "tvdc": {
        "tvdc", "h2tvdc", "thiophene vinyl dicarboxylic acid",
        "thiophene vinyl dicarboxylate",
    },
}
_ALIAS_LOOKUP = {
    _identity_key(alias): canonical
    for canonical, aliases in LIGAND_ALIASES.items()
    for alias in aliases
}


def canonical_ligand_key(value):
    """Map only exact, curated aliases to a registry identity.

    Functionalized bipyrazoles are deliberately checked by exact normalized
    identity, so an amino/nitro/tetramethyl derivative cannot fall through to
    the unsubstituted BPZ entry.
    """
    raw = " ".join(str(value or "").strip().split())
    parts = [part.strip() for part in raw.split("|") if part.strip()] or [raw]
    candidates = []
    for part in parts:
        candidates.extend(
            [part, normalize_ligand(part), canonicalize_ligand_for_model(normalize_ligand(part))]
        )
    for candidate in candidates:
        match = _ALIAS_LOOKUP.get(_identity_key(candidate))
        if match:
            return match
    return None


def _normalise_doi(value):
    doi = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            doi = doi[len(prefix):].strip()
            break
    return doi


def validate_registry(frame):
    """Raise on any registry condition that could create a wrong/broken link."""
    required = {
        "Registry_ID", "Canonical_Ligand_Key", "Canonical_Ligand_Name", "Metal",
        "Reported_Oxidation_State", "MOF_Name", "Reference_Title", "Source_DOI",
        "Reference_Role", "Verification_Status", "Scope_Note",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Literature registry is missing columns: {', '.join(missing)}")
    if frame["Registry_ID"].astype(str).duplicated().any():
        raise ValueError("Literature registry contains duplicate Registry_ID values.")
    if frame[["MOF_Name", "Reference_Title", "Source_DOI"]].fillna("").eq("").any().any():
        raise ValueError("Literature registry contains an incomplete framework/reference entry.")
    dois = frame["Source_DOI"].map(_normalise_doi)
    invalid = frame.loc[~dois.map(lambda doi: bool(DOI_PATTERN.fullmatch(doi))), "Registry_ID"].tolist()
    if invalid:
        raise ValueError(f"Literature registry contains invalid DOI syntax: {', '.join(invalid)}")
    unknown_keys = sorted(set(frame["Canonical_Ligand_Key"]) - set(LIGAND_ALIASES))
    if unknown_keys:
        raise ValueError(f"Literature registry contains unmapped ligand keys: {', '.join(unknown_keys)}")
    return True


def load_registry(path=REGISTRY_PATH):
    frame = pd.read_csv(path, keep_default_na=False)
    validate_registry(frame)
    frame = frame.copy()
    frame["Source_DOI"] = frame["Source_DOI"].map(_normalise_doi)
    frame["DOI_URL"] = frame["Source_DOI"].map(lambda doi: f"https://doi.org/{doi}")
    frame["Reported_Oxidation_State"] = pd.to_numeric(
        frame["Reported_Oxidation_State"], errors="coerce"
    ).astype("Int64")
    return frame


REGISTRY = load_registry()


def known_mof_matches(values, n=8):
    """Return curated literature candidates for an exact metal-linker pair.

    The returned names are candidates documented for the pair, not an
    identification of the user's product.  Oxidation-state agreement is
    reported and used for ranking, never silently ignored.
    """
    ligand_key = canonical_ligand_key(values.get("Legante") or values.get("Ligand_User_Input"))
    metal = str(values.get("Metallo") or "").strip()
    if not ligand_key or not metal:
        return REGISTRY.iloc[0:0].copy()
    matches = REGISTRY[
        REGISTRY["Canonical_Ligand_Key"].eq(ligand_key)
        & REGISTRY["Metal"].astype(str).eq(metal)
    ].copy()
    if matches.empty:
        return matches
    try:
        query_oxidation = int(values.get("Oxidation_State"))
    except (TypeError, ValueError):
        query_oxidation = None
    matches["Oxidation_State_Match"] = (
        True if query_oxidation is None
        else matches["Reported_Oxidation_State"].eq(query_oxidation)
    )
    matches["Match_Level"] = matches["Oxidation_State_Match"].map(
        {
            True: "Exact curated metal-linker match (oxidation state consistent)",
            False: "Curated metal-linker match (reported oxidation state differs)",
        }
    )
    matches["Identification_Level"] = "Literature candidate — structural identification requires PXRD/SCXRD"
    return (
        matches.sort_values(["Oxidation_State_Match", "Registry_ID"], ascending=[False, True])
        .head(int(n))
        .reset_index(drop=True)
    )
