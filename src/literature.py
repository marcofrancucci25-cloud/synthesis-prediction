"""Reliable literature search utilities powered by Tavily.

The module deliberately restricts results to scholarly publishers, indexing
services and preprint repositories. Search output is still retrieval-based and
must be critically evaluated by the user.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any
import os
from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "pubs.acs.org",
    "pubs.rsc.org",
    "sciencedirect.com",
    "link.springer.com",
    "nature.com",
    "onlinelibrary.wiley.com",
    "academic.oup.com",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "acs.figshare.com",
    "chemrxiv.org",
    "arxiv.org",
    "mdpi.com",
    "tandfonline.com",
    "journals.iucr.org",
    "science.org",
    "pnas.org",
    "cell.com",
]

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _doi(text: str) -> str | None:
    match = DOI_RE.search(text or "")
    return match.group(0).rstrip(".,;)") if match else None


def _api_key(explicit_key: str | None = None) -> str:
    """Read Tavily credentials at runtime; credentials are never bundled in source."""
    key = str(explicit_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if not key:
        try:
            import streamlit as st
            key = str(st.secrets.get("TAVILY_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not configured in Streamlit Secrets or the environment.")
    return key


def search_literature(
    keyword: str,
    *,
    years_back: int = 5,
    max_results: int = 10,
    api_key: str | None = None,
    mof_focus: bool = True,
) -> list[dict[str, Any]]:
    """Search recent scholarly literature and return normalized results.

    Parameters
    ----------
    keyword:
        User topic, compound, material or process.
    years_back:
        Earliest publication/update date to request from Tavily.
    max_results:
        Number of results to request (5–20 recommended).
    api_key:
        Optional runtime override. If omitted, the bundled deployment key is used.
    mof_focus:
        Add MOF/materials-science context to reduce generic matches.
    """
    query = (keyword or "").strip()
    if not query:
        return []

    key = _api_key(api_key)
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("The tavily-python package is not installed.") from exc

    context = " metal-organic framework MOF synthesis" if mof_focus else ""
    scholarly_query = f'{query}{context} research article journal DOI'
    start = date.today() - timedelta(days=365 * max(1, int(years_back)))

    client = TavilyClient(api_key=key)
    response = client.search(
        query=scholarly_query,
        search_depth="advanced",
        topic="general",
        max_results=max(5, min(int(max_results) * 2, 40)),
        include_answer=False,
        include_raw_content=False,
        include_domains=TRUSTED_DOMAINS,
        start_date=start.isoformat(),
    )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in response.get("results", []):
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "Untitled result").strip()
        if not url or url in seen:
            continue
        domain = _domain(url)
        if not any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS):
            continue
        seen.add(url)
        content = str(item.get("content") or "").strip()
        published = item.get("published_date") or item.get("date") or ""
        normalized.append(
            {
                "title": title,
                "url": url,
                "source": domain,
                "published_date": str(published),
                "score": float(item.get("score") or 0.0),
                "summary": content,
                "doi": _doi(" ".join([title, url, content])),
            }
        )

    # Prefer dated/recent records, while using Tavily relevance as a tie-breaker.
    normalized.sort(
        key=lambda r: (
            bool(r["published_date"]),
            r["published_date"],
            r["score"],
        ),
        reverse=True,
    )
    return normalized[: int(max_results)]


CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
ABBREV_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9'′\-]{1,20})\)")
QUOTED_RE = re.compile(r'["“]([^"”]{5,120})["”]')

# Generic words injected into every discover_ligand_identifiers() search query
# (see the query string built there): matching these tells us nothing about
# whether a given search RESULT is actually about the ligand the person
# asked for, so they are excluded before checking relevance.
_QUERY_NOISE_WORDS = {
    "ligand", "mof", "linker", "synonym", "cas", "smiles", "chemical", "name",
    "and", "or", "the", "of", "a", "an", "for",
}
_ALPHA_TOKEN_RE = re.compile(r"[a-z]+")


def _significant_tokens(text: str) -> set[str]:
    """Alphabetic tokens (length >= 3) from `text`, excluding generic/noise words.

    Deliberately excludes bare numbers: locants like "3" or "4" are far too
    common in chemistry text to serve as a relevance signal on their own,
    so only the more distinctive word tokens (e.g. "amino", "bipyrazole")
    are used to decide whether a search result is actually discussing the
    queried ligand.
    """
    tokens = _ALPHA_TOKEN_RE.findall((text or "").casefold())
    return {t for t in tokens if len(t) >= 3 and t not in _QUERY_NOISE_WORDS}


def _result_is_relevant(result_text: str, query_tokens: set[str]) -> bool:
    """Whether a single search result's text plausibly discusses the query ligand.

    Tavily is a general web search engine, not a chemistry-aware tool: a
    result can rank for the query while actually being about something else
    entirely, with an unrelated compound name appearing elsewhere on the
    same page. Without this check, any CAS number/abbreviation/quoted string
    found ANYWHERE in that result's text would be extracted and offered to
    the user as if it were a confirmed synonym of the query ligand -- this
    is how an unrelated compound can end up presented as a "candidate
    structure" with a plausible-looking IUPAC name and diagram. Requiring
    most of the query's own distinctive words to actually appear in the
    result text is a coarse but effective guard against that failure mode.

    Substring containment is used rather than exact token matching, so a
    query token like "amino" is still counted as present in text that says
    "diamino" or "monoamino" -- common compound-word variation that exact
    set intersection would otherwise miss.
    """
    if not query_tokens:
        # A query with no distinctive word tokens (e.g. a bare formula or a
        # CAS number) cannot be relevance-checked this way; fall back to the
        # previous permissive behavior rather than silently discarding
        # everything.
        return True
    haystack = (result_text or "").casefold()
    hits = sum(1 for token in query_tokens if token in haystack)
    return hits >= max(1, -(-len(query_tokens) * 6 // 10))  # ceil(60% of query_tokens)


def discover_ligand_identifiers(keyword: str, max_identifiers: int = 8) -> list[str]:
    """Discover alternate names/CAS/abbreviations from scholarly snippets.

    Tavily is never treated as a structure authority. Returned text is fed back
    into OPSIN/PubChem/Cactus and must still pass structural validation.
    """
    query=(keyword or "").strip()
    if not query:
        return []
    try:
        from tavily import TavilyClient
        client=TavilyClient(api_key=_api_key())
        response=client.search(
            query=f'"{query}" ligand MOF linker synonym CAS SMILES chemical name',
            search_depth="advanced", topic="general", max_results=8,
            include_answer=False, include_raw_content=False,
            include_domains=TRUSTED_DOMAINS,
        )
    except Exception:
        return []
    candidates=[]
    query_tokens = _significant_tokens(query)
    for item in response.get("results", []):
        text=" ".join([str(item.get("title") or ""), str(item.get("content") or "")])
        if not _result_is_relevant(text, query_tokens):
            # This result does not actually appear to discuss the queried
            # ligand (see _result_is_relevant docstring): do not trust any
            # identifier extracted from it, however well-formed it looks.
            continue
        candidates.extend(CAS_RE.findall(text))
        candidates.extend(ABBREV_RE.findall(text))
        candidates.extend(QUOTED_RE.findall(text))
    out=[]; seen=set()
    for value in candidates:
        value=" ".join(value.strip().split())
        key=value.casefold()
        if not value or key == query.casefold() or key in seen:
            continue
        # Avoid sentences and obvious non-identifiers.
        if len(value) > 100 or len(value.split()) > 12:
            continue
        seen.add(key); out.append(value)
        if len(out) >= max_identifiers:
            break
    return out
