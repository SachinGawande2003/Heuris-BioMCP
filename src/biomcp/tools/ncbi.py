"""
BioMCP — NCBI Tools
===================
Tools:
  search_pubmed   — Full-text PubMed search with abstract, authors, DOI, MeSH
  get_gene_info   — NCBI Gene: symbol, locus, aliases, RefSeq, summary
  run_blast       — Async NCBI BLAST with polling (blastp/blastn/blastx/tblastn)

APIs:
  NCBI E-utilities  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
  NCBI BLAST        https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi
"""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    _NCBI_SERVICE,
    cached,
    get_http_client,
    ncbi_params,
    rate_limited,
    with_retry,
)

NCBI_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BLAST_BASE = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"


# ─────────────────────────────────────────────────────────────────────────────
# PubMed — search + fetch
# ─────────────────────────────────────────────────────────────────────────────

@cached("pubmed")
@rate_limited(_NCBI_SERVICE)
@with_retry(max_attempts=3)
async def search_pubmed(
    query: str,
    max_results: int = 10,
    sort: str = "relevance",
) -> dict[str, Any]:
    """
    Search PubMed for scientific literature.

    Supports full NCBI query syntax — MeSH terms, Boolean operators,
    field tags ([Gene], [Author], [PDAT]), date ranges, etc.

    Args:
        query:       PubMed query string.
        max_results: Articles to return (1–200). Default 10.
        sort:        'relevance' | 'pub_date'. Default 'relevance'.

    Returns:
        {
          query, total_found, returned,
          articles: [{ pmid, title, authors, journal, year,
                       abstract, doi, url, mesh_terms }]
        }
    """
    max_results = BioValidator.clamp_int(max_results, 1, 200, "max_results")
    if sort not in ("relevance", "pub_date"):
        raise ValueError(f"sort must be 'relevance' or 'pub_date', got '{sort}'")

    client = await get_http_client()

    # ── Step 1: esearch → get ID list ────────────────────────────────────────
    search_resp = await client.get(
        f"{NCBI_BASE}/esearch.fcgi",
        params=ncbi_params({
            "db":         "pubmed",
            "term":       query,
            "retmax":     max_results,
            "sort":       sort,
            "usehistory": "y",
        }),
    )
    search_resp.raise_for_status()
    search_json = search_resp.json()
    esr = search_json.get("esearchresult", {})

    id_list    = esr.get("idlist", [])
    total      = int(esr.get("count", 0))

    if not id_list:
        return {"query": query, "total_found": 0, "returned": 0, "articles": []}

    # ── Step 2: efetch → full XML records ────────────────────────────────────
    fetch_resp = await client.get(
        f"{NCBI_BASE}/efetch.fcgi",
        params=ncbi_params({
            "db":      "pubmed",
            "id":      ",".join(id_list),
            "rettype": "abstract",
            "retmode": "xml",
        }),
    )
    fetch_resp.raise_for_status()

    articles = _parse_pubmed_xml(fetch_resp.text)

    return {
        "query":       query,
        "total_found": total,
        "returned":    len(articles),
        "articles":    articles,
    }


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML into structured dicts."""
    articles: list[dict[str, Any]] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"PubMed XML parse error: {e}")
        return articles

    for pm_art in root.findall(".//PubmedArticle"):
        medline = pm_art.find("MedlineCitation")
        if medline is None:
            continue
        art = medline.find("Article")
        if art is None:
            continue

        # Title (may contain italics / sub elements)
        title_el = art.find("ArticleTitle")
        title    = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract (may be structured with labels)
        abstract_parts: list[str] = []
        for ab in art.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text  = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts) or "No abstract available."

        # Authors (cap at 10 to keep response lean)
        authors: list[str] = []
        for author in art.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())
        authors = authors[:10]

        # Journal + publication year
        journal_el = art.find("Journal")
        journal    = ""
        year       = ""
        if journal_el is not None:
            journal = journal_el.findtext("Title", "")
            year    = (
                journal_el.findtext(".//Year", "")
                or journal_el.findtext(".//MedlineDate", "")
            )

        # PMID
        pmid = medline.findtext("PMID", "")

        # DOI
        doi = ""
        for id_el in pm_art.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = (id_el.text or "").strip()
                break

        # MeSH terms (cap at 15)
        mesh: list[str] = [
            mh.findtext("DescriptorName", "").strip()
            for mh in medline.findall(".//MeshHeading")
            if mh.findtext("DescriptorName", "")
        ][:15]

        articles.append({
            "pmid":       pmid,
            "title":      title,
            "authors":    authors,
            "journal":    journal,
            "year":       year,
            "abstract":   abstract[:2_500],
            "doi":        doi,
            "url":        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "mesh_terms": mesh,
        })

    return articles


# ─────────────────────────────────────────────────────────────────────────────
# NCBI Gene
# ─────────────────────────────────────────────────────────────────────────────

@cached("pubmed")
@rate_limited(_NCBI_SERVICE)
@with_retry(max_attempts=3)
async def get_gene_info(
    gene_symbol: str,
    organism: str = "homo sapiens",
) -> dict[str, Any]:
    """
    Retrieve gene information from NCBI Gene database.

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'TP53', 'BRCA1').
        organism:    Species (default: 'homo sapiens').

    Returns:
        { gene_id, symbol, full_name, organism, chromosome,
          location, aliases, summary, ncbi_url }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    client      = await get_http_client()

    # esearch
    query       = f"{gene_symbol}[Gene Name] AND {organism}[Organism]"
    search_resp = await client.get(
        f"{NCBI_BASE}/esearch.fcgi",
        params=ncbi_params({"db": "gene", "term": query, "retmax": 5}),
    )
    search_resp.raise_for_status()
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return {
            "error": f"No gene found for '{gene_symbol}' in '{organism}'. "
                     "Check the symbol or try a different species."
        }

    # esummary
    summ_resp = await client.get(
        f"{NCBI_BASE}/esummary.fcgi",
        params=ncbi_params({"db": "gene", "id": ids[0]}),
    )
    summ_resp.raise_for_status()
    gd = summ_resp.json().get("result", {}).get(ids[0], {})

    aliases_raw: str = gd.get("otheraliases", "")
    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]

    return {
        "gene_id":      ids[0],
        "symbol":       gd.get("name", gene_symbol),
        "full_name":    gd.get("description", ""),
        "organism":     gd.get("organism", {}).get("scientificname", organism),
        "chromosome":   gd.get("chromosome", ""),
        "location":     gd.get("maplocation", ""),
        "aliases":      aliases,
        "summary":      gd.get("summary", "No summary available.")[:2_000],
        "ncbi_url":     f"https://www.ncbi.nlm.nih.gov/gene/{ids[0]}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NCBI BLAST — async polling
# ─────────────────────────────────────────────────────────────────────────────

_BLAST_PROGRAMS  = frozenset({"blastp", "blastn", "blastx", "tblastn", "tblastx"})
_BLAST_DATABASES = frozenset({"nr", "nt", "swissprot", "pdb", "refseq_protein", "refseq_rna"})


@rate_limited("ncbi")
@with_retry(max_attempts=2)
async def run_blast(
    sequence: str,
    program: str  = "blastp",
    database: str = "nr",
    max_hits: int = 10,
) -> dict[str, Any]:
    """
    Run NCBI BLAST sequence alignment.

    Submits to NCBI, polls every 5 s until ready (max 120 s), then fetches JSON.

    Args:
        sequence:  Raw amino acid or nucleotide sequence.
        program:   'blastp' | 'blastn' | 'blastx' | 'tblastn' | 'tblastx'.
        database:  'nr' | 'nt' | 'swissprot' | 'pdb' | 'refseq_protein'.
        max_hits:  Alignments to return (1–100).

    Returns:
        { rid, program, database, query_length, total_hits, hits: [...], statistics }
    """
    seq_type = "protein" if program in ("blastp", "blastx", "tblastn") else "nucleotide"
    sequence = BioValidator.validate_sequence(sequence, seq_type)
    max_hits = BioValidator.clamp_int(max_hits, 1, 100, "max_hits")

    if program not in _BLAST_PROGRAMS:
        raise ValueError(f"Invalid program '{program}'. Choose from: {_BLAST_PROGRAMS}")
    if database not in _BLAST_DATABASES:
        raise ValueError(f"Invalid database '{database}'. Choose from: {_BLAST_DATABASES}")

    client = await get_http_client()

    # ── Submit ────────────────────────────────────────────────────────────────
    submit = await client.post(
        BLAST_BASE,
        data={
            "CMD":          "Put",
            "PROGRAM":      program,
            "DATABASE":     database,
            "QUERY":        sequence,
            "HITLIST_SIZE": max_hits,
            "FORMAT_TYPE":  "JSON2",
        },
    )
    submit.raise_for_status()

    rid: str | None = None
    for line in submit.text.splitlines():
        if line.strip().startswith("RID = "):
            rid = line.split("=", 1)[1].strip()
            break

    if not rid:
        raise RuntimeError(
            "BLAST submission failed — could not extract RID from response. "
            "NCBI BLAST may be temporarily unavailable."
        )

    logger.info(f"BLAST submitted: RID={rid} program={program} db={database}")

    # ── Poll (max 120 s) ──────────────────────────────────────────────────────
    for attempt in range(24):
        await asyncio.sleep(5)
        poll = await client.get(
            BLAST_BASE,
            params={"CMD": "Get", "FORMAT_OBJECT": "SearchInfo", "RID": rid},
        )
        text = poll.text
        if "Status=READY" in text:
            logger.info(f"BLAST ready after {(attempt + 1) * 5}s")
            break
        if "Status=FAILED" in text or "Status=UNKNOWN" in text:
            raise RuntimeError(f"BLAST job {rid} failed on NCBI servers.")
        logger.debug(f"BLAST polling ({attempt + 1}/24)…")
    else:
        raise TimeoutError(f"BLAST job {rid} timed out after 120 s. Try a shorter sequence.")

    # ── Fetch JSON2 results ───────────────────────────────────────────────────
    result = await client.get(
        BLAST_BASE,
        params={"CMD": "Get", "FORMAT_TYPE": "JSON2", "RID": rid},
    )
    result.raise_for_status()
    return _parse_blast_json2(result.text, rid, program, database)


def _parse_blast_json2(
    raw: str,
    rid: str,
    program: str,
    database: str,
) -> dict[str, Any]:
    """Parse BLAST JSON2 format into a clean summary dict."""
    try:
        data    = json.loads(raw)
        report  = data["BlastOutput2"][0]["report"]
        search  = report["results"]["search"]
        raw_hits = search.get("hits", [])

        hits: list[dict[str, Any]] = []
        for h in raw_hits:
            desc = h.get("description", [{}])[0]
            hsp  = h.get("hsps",        [{}])[0]
            aln  = max(hsp.get("align_len", 1), 1)
            q_len = max(search.get("query_len", 1), 1)

            hits.append({
                "accession":       desc.get("accession", ""),
                "title":           desc.get("title",     "")[:120],
                "taxid":           desc.get("taxid",     ""),
                "sciname":         desc.get("sciname",   ""),
                "identity_pct":    round(hsp.get("identity", 0) / aln * 100, 2),
                "query_cover_pct": round(
                    (hsp.get("query_to", 0) - hsp.get("query_from", 0)) / q_len * 100, 2
                ),
                "evalue":          hsp.get("evalue",    ""),
                "bit_score":       hsp.get("bit_score", ""),
                "gaps":            hsp.get("gaps",      0),
                "positives_pct":   round(hsp.get("positive", 0) / aln * 100, 2),
            })

        return {
            "rid":          rid,
            "program":      program,
            "database":     database,
            "query_length": search.get("query_len", 0),
            "total_hits":   len(raw_hits),
            "hits":         hits,
            "statistics":   search.get("stat", {}),
        }

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Failed to parse BLAST JSON results: {e}. "
            "NCBI may have returned an unexpected format."
        ) from e
