from __future__ import annotations

import re
import json
import time
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any
from pathlib import Path
from urllib.parse import quote

import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey

USER_AGENT = "MOF-Synthesis-Assistant/9.0 (chemical resolver; academic research)"
DEFAULT_TIMEOUT = 5
CONFIRMED_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "confirmed_ligands.json"

# Small curated alias layer: it accelerates common MOF linkers but is not a ligand database.
ALIASES: dict[str, str] = {
    "bdc": "terephthalic acid",
    "h2bdc": "terephthalic acid",
    "1,4-bdc": "terephthalic acid",
    "btc": "benzene-1,3,5-tricarboxylic acid",
    "h3btc": "benzene-1,3,5-tricarboxylic acid",
    "trimesic acid": "benzene-1,3,5-tricarboxylic acid",
    "bpdc": "biphenyl-4,4'-dicarboxylic acid",
    "dobdc": "2,5-dihydroxyterephthalic acid",
    "h4dobdc": "2,5-dihydroxyterephthalic acid",
    "bpz": "4,4'-bipyrazole",
    "h2bpz": "4,4'-bipyrazole",
    "hmim": "2-methylimidazole",
    "mim": "2-methylimidazole",
    "bpy": "2,2'-bipyridine",
    "4,4-bpy": "4,4'-bipyridine",
}


# Curated structures for specialist MOF linkers that are often absent or
# ambiguously indexed in general-purpose chemical databases.  These entries
# are resolved locally before external API calls.
LOCAL_STRUCTURES: dict[str, dict[str, Any]] = {
    "3-amino-4,4'-bipyrazole": {
        "title": "3-amino-4,4′-bipyrazole",
        "iupac_name": "4-(1H-pyrazol-4-yl)-1H-pyrazol-3-amine",
        "smiles": "Nc1n[nH]cc1-c1cn[nH]c1",
        "expected_formula": "C6H7N5",
        "expected_molecular_weight": 149.157,
        "required_smarts": ["[NX3;H2]-[c,n]", "c1n[nH]cc1", "c1cn[nH]c1"],
    },
}

# Only unambiguous aliases are accepted. Generic labels such as
# "aminobipyrazole" are intentionally excluded because they do not specify
# substitution position or the 4,4′ connectivity and may resolve to another isomer.
LOCAL_STRUCTURE_ALIASES: dict[str, str] = {
    "3-amino-4,4'-bipyrazole": "3-amino-4,4'-bipyrazole",
    "3-amino-4,4-bipyrazole": "3-amino-4,4'-bipyrazole",
    "3-amino-4,4'-bipyrazol": "3-amino-4,4'-bipyrazole",
    "3-amino-4,4-bipyrazol": "3-amino-4,4'-bipyrazole",
    "3-aminobipirazolo-4,4'": "3-amino-4,4'-bipyrazole",
    "3-amino-4,4'-bipirazolo": "3-amino-4,4'-bipyrazole",
    "3-amino-4,4-bipirazolo": "3-amino-4,4'-bipyrazole",
    "bpznh2": "3-amino-4,4'-bipyrazole",
    "bpz-nh2": "3-amino-4,4'-bipyrazole",
    "3-nh2-bpz": "3-amino-4,4'-bipyrazole",
    "h2bpznh2": "3-amino-4,4'-bipyrazole",
    "h2bpz-nh2": "3-amino-4,4'-bipyrazole",
}

SPECIALIST_AMBIGUOUS_PATTERNS = (
    "aminobipyrazole", "amino bipyrazole", "aminobipirazolo", "amino bipirazolo",
    "nitrobipyrazole", "nitro bipyrazole", "diaminobipyrazole", "dinitrobipyrazole",
)

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){2,}(?:[+-]\d*)?$")


def _load_confirmed_cache(extra_cache_json: str = "") -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    try:
        if CONFIRMED_CACHE_PATH.exists():
            payload = json.loads(CONFIRMED_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cache.update(payload)
    except Exception:
        pass
    if extra_cache_json:
        try:
            payload = json.loads(extra_cache_json)
            if isinstance(payload, dict):
                cache.update(payload)
        except Exception:
            pass
    return cache


def _query_variants(query: str) -> list[str]:
    """Generate conservative alternate identifiers without deleting meaningful locants."""
    q = normalize_query(query)
    variants = [q]
    variants += [
        q.replace("'", "′"),
        q.replace("′", "'"),
        q.replace("-", " "),
        re.sub(r"\s+", "", q),
    ]
    # Acid-form abbreviations: H2BDC -> BDC, H3BTC -> BTC.
    compact = re.sub(r"[^A-Za-z0-9]+", "", q)
    stripped = re.sub(r"^[Hh]\d+", "", compact)
    if stripped and stripped != compact:
        variants.append(stripped)
    # Extract exact abbreviation in parentheses, preserving the original query.
    match = re.search(r"\(([^)]+)\)", q)
    if match:
        variants.append(match.group(1).strip())
    # Common spelling variants used in MOF papers.
    substitutions = {
        "bipirazolo": "bipyrazole", "bipirazolo": "bipyrazole",
        "bipyrazol": "bipyrazole", "benzenedicarboxylic": "phthalic",
    }
    low = q.casefold()
    for old, new in substitutions.items():
        if old in low:
            variants.append(re.sub(old, new, q, flags=re.I))
    out=[]
    seen=set()
    for item in variants:
        item=normalize_query(item)
        key=item.casefold()
        if item and key not in seen:
            seen.add(key); out.append(item)
    return out[:12]


def _standardize_smiles(smiles: str) -> tuple[str, str] | None:
    """RDKit parent/charge/tautomer standardization for candidate deduplication."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Normalizer().normalize(mol)
        mol = rdMolStandardize.Reionizer().reionize(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        pass
    full = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return full, connectivity


def confirmed_entry(query: str, candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a portable cache record for a user-confirmed ligand identity."""
    normalized = normalize_query(query).casefold()
    return {
        normalized: {
            "query": query,
            "title": candidate.get("title") or candidate.get("iupac_name") or query,
            "iupac_name": candidate.get("iupac_name"),
            "smiles": candidate.get("smiles"),
            "molecular_formula": candidate.get("molecular_formula"),
            "inchikey": candidate.get("inchikey"),
            "source": candidate.get("source") or "user-confirmed resolver candidate",
            "aliases": sorted(set([query, candidate.get("title") or "", candidate.get("iupac_name") or ""]) - {""}),
            "confidence": "user confirmed",
        }
    }


@dataclass
class ResolutionResult:
    success: bool
    query: str
    normalized_query: str
    input_type: str
    source: str | None = None
    title: str | None = None
    iupac_name: str | None = None
    molecular_formula: str | None = None
    smiles: str | None = None
    connectivity_smiles: str | None = None
    inchikey: str | None = None
    molecular_weight: float | None = None
    message: str | None = None
    ambiguity_warning: str | None = None
    descriptors: dict[str, float | int] | None = None
    confidence: str | None = None
    validation_notes: list[str] | None = None
    needs_confirmation: bool = False
    candidates: list[dict[str, Any]] | None = None
    consensus_sources: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_query(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    replacements = {
        "’": "'", "‘": "'", "′": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-", "·": ".",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.strip().split())
    return text


def _alias(query: str) -> str:
    lowered = query.casefold()
    if lowered in ALIASES:
        return ALIASES[lowered]
    match = re.search(r"\(([^)]+)\)", query)
    if match:
        abbreviation = match.group(1).strip().casefold()
        stripped = re.sub(r"^h\d+", "", abbreviation)
        if abbreviation in ALIASES:
            return ALIASES[abbreviation]
        if stripped in ALIASES:
            return ALIASES[stripped]
    return query


def detect_input_type(query: str) -> str:
    if CAS_RE.fullmatch(query):
        return "CAS"
    mol = Chem.MolFromSmiles(query)
    if mol is not None and any(token in query for token in "[]=#()@+-\\/"):
        return "SMILES"
    compact = query.replace(" ", "")
    if FORMULA_RE.fullmatch(compact):
        return "molecular formula"
    return "chemical name or abbreviation"


def _descriptors(smiles: str) -> dict[str, float | int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolecularWeight_RDKit": round(float(Descriptors.MolWt(mol)), 4),
        "ExactMass": round(float(Descriptors.ExactMolWt(mol)), 4),
        "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        "LogP": round(float(Crippen.MolLogP(mol)), 4),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
        "FormalCharge": int(Chem.GetFormalCharge(mol)),
    }


def _canonicalize_smiles(smiles: str) -> tuple[str, str] | None:
    return _standardize_smiles(smiles)


def _request_json(url: str, timeout: int) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                time.sleep(0.8)
                continue
            return None
        except requests.RequestException:
            if attempt == 0:
                time.sleep(0.5)
                continue
            return None
    return None


def _cactus_smiles(query: str, timeout: int) -> str | None:
    url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(query, safe='')}/smiles"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        candidate = response.text.strip()
        if not candidate or "page not found" in candidate.casefold():
            return None
        canonical = _canonicalize_smiles(candidate)
        return canonical[0] if canonical else None
    except requests.RequestException:
        return None



def _opsin_candidate(query: str, timeout: int) -> dict[str, Any] | None:
    """Resolve a systematic chemical name through the official EMBL-EBI OPSIN web service."""
    url = f"https://www.ebi.ac.uk/opsin/ws/{quote(query, safe='')}.json"
    payload = _request_json(url, timeout)
    if not payload or str(payload.get("status", "")).upper() not in {"SUCCESS", "WARNING"}:
        return None
    smiles = payload.get("smiles")
    canonical = _canonicalize_smiles(smiles) if smiles else None
    if not canonical:
        return None
    return {
        "Title": query,
        "IUPACName": query,
        "SMILES": canonical[0],
        "ConnectivitySMILES": canonical[1],
        "InChIKey": payload.get("stdinchikey"),
        "_source": "OPSIN",
        "_warning": "; ".join(payload.get("warnings") or []) or None,
    }


def _candidate_from_smiles(query: str, normalized: str, input_type: str, smiles: str,
                           source: str, metadata: dict[str, Any] | None = None) -> ResolutionResult | None:
    canonical = _canonicalize_smiles(smiles)
    if not canonical:
        return None
    full, connectivity = canonical
    mol = Chem.MolFromSmiles(full)
    if mol is None:
        return None
    metadata = metadata or {}
    formula = metadata.get("MolecularFormula") or rdMolDescriptors.CalcMolFormula(mol)
    mw = metadata.get("MolecularWeight")
    try:
        mw = float(mw) if mw is not None else float(Descriptors.MolWt(mol))
    except (TypeError, ValueError):
        mw = float(Descriptors.MolWt(mol))
    inchikey = metadata.get("InChIKey") or MolToInchiKey(mol)
    confidence, notes = _identity_consistency(normalized, metadata)
    warning = metadata.get("_warning")
    if warning:
        notes.append(str(warning))
        confidence = "medium" if confidence == "high" else confidence
    return ResolutionResult(
        success=True, query=query, normalized_query=normalized, input_type=input_type,
        source=source, title=metadata.get("Title") or normalized,
        iupac_name=metadata.get("IUPACName"), molecular_formula=formula,
        smiles=full, connectivity_smiles=connectivity, inchikey=inchikey,
        molecular_weight=round(mw, 4), descriptors=_descriptors(full),
        confidence=confidence, validation_notes=notes,
        consensus_sources=[source],
        message=f"Candidate resolved through {source} and validated with RDKit.",
    )


def _candidate_key(result: ResolutionResult) -> str:
    return result.inchikey or result.connectivity_smiles or result.smiles or ""


def _merge_candidates(candidates: list[ResolutionResult]) -> list[ResolutionResult]:
    merged: dict[str, ResolutionResult] = {}
    for candidate in candidates:
        key = _candidate_key(candidate)
        if not key:
            continue
        if key not in merged:
            merged[key] = candidate
            continue
        current = merged[key]
        sources = list(dict.fromkeys((current.consensus_sources or [current.source]) + (candidate.consensus_sources or [candidate.source])))
        current.consensus_sources = [x for x in sources if x]
        current.source = " + ".join(current.consensus_sources)
        current.validation_notes = list(dict.fromkeys((current.validation_notes or []) + (candidate.validation_notes or [])))
        if not current.iupac_name and candidate.iupac_name:
            current.iupac_name = candidate.iupac_name
        if not current.title and candidate.title:
            current.title = candidate.title
    return list(merged.values())


def _score_candidate(candidate: ResolutionResult) -> float:
    sources = candidate.consensus_sources or ([candidate.source] if candidate.source else [])
    score = 40.0 + 22.0 * max(0, len(sources) - 1)
    if "OPSIN" in sources:
        score += 12
    if any("PubChem" in x for x in sources):
        score += 10
    if any("Cactus" in x for x in sources):
        score += 5
    notes = candidate.validation_notes or []
    score -= 18 * len([n for n in notes if "does not" in n.lower() or "mismatch" in n.lower()])
    for note in notes:
        if note.startswith("Synonym/name match bonus:"):
            try: score += float(note.split(":",1)[1].strip())
            except Exception: pass
    return max(0.0, min(100.0, score))


def _candidate_dict(candidate: ResolutionResult) -> dict[str, Any]:
    data = candidate.to_dict()
    data["consensus_score"] = round(_score_candidate(candidate), 1)
    return data

def _pubchem_properties(namespace: str, identifier: str, timeout: int) -> dict[str, Any] | None:
    # PubChem renamed the historical CanonicalSMILES/IsomericSMILES properties.
    properties = "Title,IUPACName,MolecularFormula,SMILES,ConnectivitySMILES,InChIKey,MolecularWeight"
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
        f"{namespace}/{quote(identifier, safe='')}/property/{properties}/JSON"
    )
    payload = _request_json(url, timeout)
    if not payload:
        return None
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else None


def _pubchem_cids(namespace: str, identifier: str, timeout: int, *, word_search: bool = False, max_records: int = 12) -> list[int]:
    suffix = "?name_type=word" if word_search and namespace == "name" else ""
    joiner = "&" if suffix else "?"
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"{namespace}/{quote(identifier, safe='')}/cids/JSON{suffix}{joiner}MaxRecords={max_records}")
    payload = _request_json(url, timeout)
    return list((payload or {}).get("IdentifierList", {}).get("CID", []))[:max_records]


def _pubchem_synonyms(cid: int | str, timeout: int) -> list[str]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    payload = _request_json(url, timeout)
    rows = (payload or {}).get("InformationList", {}).get("Information", [])
    synonyms = rows[0].get("Synonym", []) if rows else []
    return [str(x) for x in synonyms[:80]]


def _pubchem_candidates_for_name(name: str, timeout: int) -> list[dict[str, Any]]:
    cids = _pubchem_cids("name", name, timeout, word_search=False)
    if not cids:
        cids = _pubchem_cids("name", name, timeout, word_search=True)
    rows=[]
    for cid in cids[:4]:
        row=_pubchem_properties("cid", str(cid), timeout)
        if row:
            row["CID"] = cid
            row["Synonyms"] = _pubchem_synonyms(cid, min(timeout, 3))
            rows.append(row)
    return rows


def _synonym_match_score(query_variants: list[str], row: dict[str, Any]) -> float:
    terms=[normalize_query(x).casefold() for x in query_variants]
    names=[row.get("Title") or "", row.get("IUPACName") or ""] + list(row.get("Synonyms") or [])
    names=[normalize_query(x).casefold() for x in names if x]
    score=0.0
    for q in terms:
        if q in names:
            score=max(score, 30.0)
        elif any(q and (q in n or n in q) for n in names):
            score=max(score, 16.0)
    return score


def _pubchem_formula_candidate(formula: str, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    # Formula searches are inherently ambiguous. Retrieve a short CID list and use the
    # first PubChem result only as an editable candidate, while surfacing the ambiguity.
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/"
        f"{quote(formula, safe='')}/cids/JSON?MaxRecords=10"
    )
    payload = _request_json(url, timeout)
    cids = (payload or {}).get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, None
    row = _pubchem_properties("cid", str(cids[0]), timeout)
    warning = (
        f"The molecular formula matched {len(cids)} PubChem candidate(s) in the first result page. "
        "The proposed structure must be checked manually or replaced with a name, CAS number or SMILES."
    )
    return row, warning


def _pubchem_formula_candidates(formula: str, timeout: int) -> list[dict[str, Any]]:
    cids = _pubchem_cids("formula", formula, timeout, max_records=12)
    rows=[]
    for cid in cids[:10]:
        row=_pubchem_properties("cid", str(cid), timeout)
        if row:
            row["CID"] = cid
            rows.append(row)
    return rows


def _result_from_pubchem(query: str, normalized: str, input_type: str, row: dict[str, Any], source: str,
                         warning: str | None = None) -> ResolutionResult:
    smiles = row.get("SMILES") or row.get("ConnectivitySMILES")
    canonical = _canonicalize_smiles(smiles) if smiles else None
    full = canonical[0] if canonical else smiles
    connectivity = canonical[1] if canonical else row.get("ConnectivitySMILES")
    confidence, consistency_notes = _identity_consistency(normalized, row)
    if warning:
        confidence = "low"
        consistency_notes.append(warning)
    return ResolutionResult(
        success=bool(full), query=query, normalized_query=normalized, input_type=input_type,
        source=source, title=row.get("Title"), iupac_name=row.get("IUPACName"),
        molecular_formula=row.get("MolecularFormula"), smiles=full,
        connectivity_smiles=connectivity, inchikey=row.get("InChIKey"),
        molecular_weight=float(row["MolecularWeight"]) if row.get("MolecularWeight") is not None else None,
        ambiguity_warning=warning, descriptors=_descriptors(full) if full else None,
        confidence=confidence, validation_notes=consistency_notes,
        message=("Structure resolved and parsed with RDKit. Confirm low-confidence API identities manually."
                 if full else "The service returned no valid SMILES.")
    )



def _validate_curated_entry(entry: dict[str, Any], smiles: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ["The curated SMILES could not be parsed by RDKit."]

    formula = rdMolDescriptors.CalcMolFormula(mol)
    expected_formula = entry.get("expected_formula")
    if expected_formula and formula != expected_formula:
        notes.append(f"Formula mismatch: calculated {formula}, expected {expected_formula}.")

    expected_mw = entry.get("expected_molecular_weight")
    calculated_mw = float(Descriptors.MolWt(mol))
    if expected_mw is not None and abs(calculated_mw - float(expected_mw)) > 0.05:
        notes.append(f"Molecular-weight mismatch: calculated {calculated_mw:.3f}, expected {float(expected_mw):.3f}.")

    for smarts in entry.get("required_smarts", []):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None or not mol.HasSubstructMatch(pattern):
            notes.append(f"Required structural motif not found: {smarts}.")

    return not notes, notes


def _is_ambiguous_specialist_name(query: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
    return any(pattern in compact for pattern in SPECIALIST_AMBIGUOUS_PATTERNS)


def _identity_consistency(query: str, row: dict[str, Any]) -> tuple[str, list[str]]:
    """Conservative consistency screen for API-derived identities.

    It does not prove structural identity, but prevents high-confidence display
    when a substituted specialist linker loses its functional-group or scaffold
    keywords during resolution.
    """
    text = " ".join(str(row.get(k) or "") for k in ("Title", "IUPACName")).casefold()
    query_l = query.casefold()
    notes: list[str] = []
    functional_terms = [t for t in ("amino", "nitro", "hydroxy", "carbox", "methyl", "ethyl") if t in query_l]
    for term in functional_terms:
        if term not in text and not (term == "amino" and "amine" in text):
            notes.append(f"The resolved name does not preserve the requested '{term}' functionality.")
    scaffold_terms = [t for t in ("bipyraz", "bipyrid", "imidazol", "terephthal", "trimes") if t in query_l]
    for term in scaffold_terms:
        if term not in text.replace("-", ""):
            notes.append(f"The resolved name does not clearly preserve the requested '{term}' scaffold.")
    confidence = "high" if not notes else "low"
    return confidence, notes

def _local_structure_result(original: str, normalized: str, key: str) -> ResolutionResult | None:
    canonical_key = LOCAL_STRUCTURE_ALIASES.get(key.casefold())
    if not canonical_key:
        return None
    entry = LOCAL_STRUCTURES[canonical_key]
    valid, validation_notes = _validate_curated_entry(entry, entry["smiles"])
    if not valid:
        return None
    canonical = _canonicalize_smiles(entry["smiles"])
    if not canonical:
        return None
    full, connectivity = canonical
    mol = Chem.MolFromSmiles(full)
    formula = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None
    mw = float(Descriptors.MolWt(mol)) if mol is not None else None
    inchikey = MolToInchiKey(mol) if mol is not None else None
    return ResolutionResult(
        success=True, query=original, normalized_query=normalized,
        input_type="curated MOF ligand", source="curated MOF linker library / RDKit",
        title=entry["title"], iupac_name=entry.get("iupac_name"),
        molecular_formula=formula, smiles=full, connectivity_smiles=connectivity,
        inchikey=inchikey, molecular_weight=round(mw, 4) if mw is not None else None,
        descriptors=_descriptors(full), confidence="high",
        validation_notes=[
            "Exact curated alias match.",
            f"Calculated formula confirmed as {formula}.",
            "Connectivity and required functional motifs validated with RDKit.",
        ],
        message="Resolved locally from a curated MOF-linker entry and validated against formula, mass and structural motifs.",
    )

@lru_cache(maxsize=1024)
def resolve_ligand(query: str, timeout: int = DEFAULT_TIMEOUT, user_cache_json: str = "") -> dict[str, Any]:
    original = str(query or "")
    normalized = normalize_query(original)
    if not normalized:
        return ResolutionResult(False, original, normalized, "empty", message="Enter a ligand identifier.").to_dict()

    cache = _load_confirmed_cache(user_cache_json)
    cache_key = normalized.casefold()
    cached = cache.get(cache_key)
    if not cached:
        for entry in cache.values():
            aliases = [normalize_query(x).casefold() for x in entry.get("aliases", [])]
            if cache_key in aliases:
                cached = entry; break
    if cached and cached.get("smiles"):
        candidate = _candidate_from_smiles(original, normalized, "confirmed ligand cache", cached["smiles"],
                                            "confirmed ligand cache / RDKit", cached)
        if candidate:
            candidate.confidence = "user confirmed"
            candidate.validation_notes = ["Exact match in the portable confirmed-ligand cache."]
            return candidate.to_dict()

    local = _local_structure_result(original, normalized, normalized.casefold())
    if local:
        local.consensus_sources = ["Curated MOF linker library"]
        return local.to_dict()

    aliased = _alias(normalized)
    local = _local_structure_result(original, normalized, aliased.casefold())
    if local:
        local.consensus_sources = ["Curated MOF linker library"]
        return local.to_dict()
    input_type = detect_input_type(aliased)

    if _is_ambiguous_specialist_name(normalized) and normalized.casefold() not in LOCAL_STRUCTURE_ALIASES:
        return ResolutionResult(
            False, original, normalized, input_type, confidence="unresolved",
            validation_notes=["Substitution positions or linker connectivity are not uniquely specified."],
            message=("The name is chemically ambiguous. Enter the complete locanted name, an exact curated "
                     "abbreviation, or a SMILES to prevent assignment of the wrong isomer.")
        ).to_dict()

    if input_type == "SMILES":
        candidate = _candidate_from_smiles(original, normalized, input_type, aliased, "direct SMILES / RDKit")
        if candidate:
            candidate.confidence = "high"
            candidate.validation_notes = ["Direct structure input parsed and canonicalized with RDKit."]
            return candidate.to_dict()

    if input_type == "molecular formula":
        rows = _pubchem_formula_candidates(aliased.replace(" ", ""), timeout)
        candidates=[]
        for row in rows:
            smiles=row.get("SMILES") or row.get("ConnectivitySMILES")
            if smiles:
                c=_candidate_from_smiles(original, normalized, input_type, smiles, "PubChem formula search", row)
                if c: candidates.append(c)
        candidates=_merge_candidates(candidates)
        candidates.sort(key=_score_candidate, reverse=True)
        if candidates:
            return ResolutionResult(
                False, original, normalized, input_type, confidence="confirmation required",
                needs_confirmation=True, candidates=[_candidate_dict(c) for c in candidates[:10]],
                ambiguity_warning=f"The formula matched {len(candidates)} distinct standardized structures.",
                message="A molecular formula is not structurally unique. Select a candidate or enter a name/SMILES."
            ).to_dict()

    candidates: list[ResolutionResult] = []
    variants = _query_variants(aliased)

    # OPSIN is strongest for systematic names and retains locants.
    for variant in variants[:2]:
        opsin = _opsin_candidate(variant, timeout)
        if opsin:
            c = _candidate_from_smiles(original, normalized, input_type, opsin["SMILES"], "OPSIN", opsin)
            if c:
                c.validation_notes = list(c.validation_notes or []) + [f"Resolved from query variant: {variant}"]
                candidates.append(c)

    # PubChem exact and word searches may expose multiple CIDs and synonyms.
    for variant in variants[:3]:
        for row in _pubchem_candidates_for_name(variant, timeout):
            smiles = row.get("SMILES") or row.get("ConnectivitySMILES")
            if not smiles:
                continue
            c = _candidate_from_smiles(original, normalized, input_type, smiles, "PubChem PUG REST", row)
            if c:
                bonus = _synonym_match_score(variants, row)
                c.validation_notes = list(c.validation_notes or []) + [
                    f"PubChem CID: {row.get('CID')}", f"Resolved from query variant: {variant}",
                    f"Synonym/name match bonus: {bonus}"
                ]
                candidates.append(c)
        if len(_merge_candidates(candidates)) >= 4:
            break

    # Cactus is retained as a secondary identifier resolver.
    for variant in variants[:2] if len(_merge_candidates(candidates)) < 2 else []:
        cactus = _cactus_smiles(variant, timeout)
        if cactus:
            metadata = _pubchem_properties("smiles", cactus, timeout) or {"Title": variant}
            c = _candidate_from_smiles(original, normalized, input_type, cactus, "NCI Cactus", metadata)
            if c:
                c.validation_notes = list(c.validation_notes or []) + [f"Resolved from query variant: {variant}"]
                candidates.append(c)

    # Last fallback: Tavily is used only to discover alternate textual identifiers.
    if not candidates:
        try:
            from src.literature import discover_ligand_identifiers
            discovered = discover_ligand_identifiers(normalized, max_identifiers=8)
        except Exception:
            discovered = []
        for identifier in discovered:
            for row in _pubchem_candidates_for_name(identifier, timeout):
                smiles = row.get("SMILES") or row.get("ConnectivitySMILES")
                if smiles:
                    c = _candidate_from_smiles(original, normalized, input_type, smiles,
                                               "Tavily-discovered identifier + PubChem", row)
                    if c:
                        c.validation_notes = list(c.validation_notes or []) + [
                            f"Alternate identifier discovered from scholarly web results: {identifier}"
                        ]
                        candidates.append(c)

    candidates = _merge_candidates(candidates)
    candidates.sort(key=_score_candidate, reverse=True)

    if not candidates:
        return ResolutionResult(
            False, original, normalized, input_type,
            message=("No valid structure was found. Enter a SMILES, CAS number, or a more systematic name. "
                     "You may continue by selecting the ligand family manually.")
        ).to_dict()

    best = candidates[0]
    best_score = _score_candidate(best)
    second_score = _score_candidate(candidates[1]) if len(candidates) > 1 else -1
    consensus_count = len(best.consensus_sources or [])

    # Automatic acceptance requires either independent source agreement or a unique, strong OPSIN result.
    auto_accept = consensus_count >= 2 and best_score >= 60
    auto_accept = auto_accept or (len(candidates) == 1 and "OPSIN" in (best.consensus_sources or []) and best_score >= 52 and best.confidence != "low")

    if auto_accept and (len(candidates) == 1 or best_score - second_score >= 8):
        best.confidence = "high" if consensus_count >= 2 else "medium"
        best.message = (f"Identity accepted from {consensus_count} agreeing source(s) and validated with RDKit."
                        if consensus_count >= 2 else "Unique OPSIN candidate validated with RDKit.")
        return best.to_dict()

    return ResolutionResult(
        False, original, normalized, input_type, confidence="confirmation required",
        needs_confirmation=True, candidates=[_candidate_dict(c) for c in candidates[:5]],
        message=("Multiple or insufficiently corroborated structures were found. Select and confirm the correct "
                 "candidate before using it for prediction.")
    ).to_dict()

