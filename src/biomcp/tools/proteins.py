"""
BioMCP — Protein Tools
======================
Tools:
  get_protein_info        — UniProt Swiss-Prot full record
  search_proteins         — UniProt keyword search with filters
  get_alphafold_structure — AlphaFold DB prediction + pLDDT statistics
  search_pdb_structures   — RCSB PDB experimental structure search

APIs:
  UniProt REST  https://rest.uniprot.org/
  AlphaFold DB  https://alphafold.ebi.ac.uk/api/
  RCSB PDB      https://search.rcsb.org/ + https://data.rcsb.org/
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

UNIPROT_BASE   = "https://rest.uniprot.org"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api"
PDB_SEARCH     = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA       = "https://data.rcsb.org/rest/v1/core/entry"


# ─────────────────────────────────────────────────────────────────────────────
# UniProt — full record
# ─────────────────────────────────────────────────────────────────────────────

@cached("uniprot")
@rate_limited("uniprot")
@with_retry(max_attempts=3)
async def get_protein_info(accession: str) -> dict[str, Any]:
    """
    Retrieve a comprehensive UniProt entry (Swiss-Prot or TrEMBL).

    Args:
        accession: UniProt accession (e.g. 'P04637' for human TP53).

    Returns:
        Full record including function, sequence, domains, PTMs, GO terms,
        disease associations, subcellular location, and cross-references.
    """
    accession = BioValidator.validate_uniprot_accession(accession)
    client    = await get_http_client()

    resp = await client.get(
        f"{UNIPROT_BASE}/uniprotkb/{accession}",
        params={"format": "json"},
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:
        return {"error": f"UniProt accession '{accession}' not found."}
    resp.raise_for_status()
    d = resp.json()

    # ── Names ─────────────────────────────────────────────────────────────────
    prot_desc  = d.get("proteinDescription", {})
    rec_name   = prot_desc.get("recommendedName", {})
    full_name  = rec_name.get("fullName", {}).get("value", "")
    short_names = [s["value"] for s in rec_name.get("shortNames", []) if s.get("value")]

    # ── Genes ──────────────────────────────────────────────────────────────────
    gene_names = [
        gene.get("geneName", {}).get("value", "")
        for gene in d.get("genes", [])
        if gene.get("geneName", {}).get("value")
    ]

    # ── Comments (function, location, disease, PTM, …) ───────────────────────
    comments   = d.get("comments", [])
    _txt       = lambda c: (c.get("texts") or [{}])[0].get("value", "")

    functions  = [_txt(c) for c in comments if c.get("commentType") == "FUNCTION"]
    ptms       = [_txt(c) for c in comments if c.get("commentType") == "PTM"]
    locations  = [
        loc.get("location", {}).get("value", "")
        for c in comments if c.get("commentType") == "SUBCELLULAR LOCATION"
        for loc in c.get("subcellularLocations", [])
    ]
    diseases = [
        {
            "name":        c.get("disease", {}).get("diseaseName", {}).get("value", ""),
            "description": _txt(c),
        }
        for c in comments if c.get("commentType") == "DISEASE"
    ]

    # ── GO terms ──────────────────────────────────────────────────────────────
    go_terms: list[dict[str, str]] = []
    for xref in d.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        props = {p["key"]: p["value"] for p in xref.get("properties", [])}
        go_terms.append({
            "id":       xref.get("id", ""),
            "term":     props.get("GoTerm", ""),
            "evidence": props.get("GoEvidenceType", ""),
        })

    # ── Features (domains, binding sites, active sites) ──────────────────────
    features = [
        {
            "type":        f.get("type", ""),
            "description": f.get("description", ""),
            "start":       f.get("location", {}).get("start", {}).get("value"),
            "end":         f.get("location", {}).get("end",   {}).get("value"),
        }
        for f in d.get("features", [])[:25]
    ]

    # ── Sequence ──────────────────────────────────────────────────────────────
    seq = d.get("sequence", {})

    return {
        "accession":           accession,
        "entry_type":          d.get("entryType", ""),
        "reviewed":            "Swiss-Prot" in d.get("entryType", ""),
        "full_name":           full_name,
        "short_names":         short_names,
        "gene_names":          gene_names,
        "organism":            d.get("organism", {}).get("scientificName", ""),
        "taxon_id":            d.get("organism", {}).get("taxonId", ""),
        "sequence_length":     seq.get("length", 0),
        "molecular_weight_da": seq.get("molWeight", 0),
        "sequence":            seq.get("value", ""),
        "function":            " ".join(functions)[:2_000],
        "subcellular_location":locations,
        "diseases":            diseases,
        "ptm":                 ptms,
        "go_terms":            go_terms[:25],
        "features":            features,
        "uniprot_url":         f"https://www.uniprot.org/uniprotkb/{accession}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# UniProt — keyword search
# ─────────────────────────────────────────────────────────────────────────────

@cached("uniprot")
@rate_limited("uniprot")
@with_retry(max_attempts=3)
async def search_proteins(
    query: str,
    organism: str = "homo sapiens",
    max_results: int = 10,
    reviewed_only: bool = True,
) -> dict[str, Any]:
    """
    Search UniProt for proteins matching a free-text query.

    Args:
        query:         Search terms (gene name, function keyword, disease, etc.).
        organism:      Species filter (default: 'homo sapiens').
        max_results:   Results to return (1–100).
        reviewed_only: Return only Swiss-Prot manually reviewed entries.

    Returns:
        { query, total_results, proteins: [{ accession, name, genes, organism,
                                              length, reviewed, url }] }
    """
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    q_parts = [f"({query})"]
    if organism:
        q_parts.append(f"(organism_name:{organism})")
    if reviewed_only:
        q_parts.append("(reviewed:true)")

    resp = await client.get(
        f"{UNIPROT_BASE}/uniprotkb/search",
        params={
            "query":  " AND ".join(q_parts),
            "format": "json",
            "size":   max_results,
            "fields": "accession,protein_name,gene_names,organism_name,length,reviewed",
        },
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[dict[str, Any]] = []
    for entry in data.get("results", []):
        acc      = entry.get("primaryAccession", "")
        rec_name = entry.get("proteinDescription", {}).get("recommendedName", {})
        name     = rec_name.get("fullName", {}).get("value", "Unknown")
        genes    = [
            g.get("geneName", {}).get("value", "")
            for g in entry.get("genes", []) if g.get("geneName")
        ]
        results.append({
            "accession": acc,
            "name":      name,
            "genes":     genes,
            "organism":  entry.get("organism", {}).get("scientificName", ""),
            "length":    entry.get("sequence", {}).get("length", 0),
            "reviewed":  "Swiss-Prot" in entry.get("entryType", ""),
            "url":       f"https://www.uniprot.org/uniprotkb/{acc}",
        })

    return {
        "query":         query,
        "organism":      organism,
        "reviewed_only": reviewed_only,
        "total_results": data.get("totalResults", len(results)),
        "proteins":      results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AlphaFold DB — predicted structure + pLDDT analysis
# ─────────────────────────────────────────────────────────────────────────────

@cached("alphafold")
@rate_limited("alphafold")
@with_retry(max_attempts=3)
async def get_alphafold_structure(
    uniprot_accession: str,
    model_version: str = "v4",
) -> dict[str, Any]:
    """
    Retrieve AlphaFold structure prediction metadata.

    Args:
        uniprot_accession: UniProt accession (e.g. 'P04637').
        model_version:     'v1' | 'v2' | 'v3' | 'v4'. Default 'v4'.

    Returns:
        { accession, entry_id, model_date, plddt_summary,
          confidence_bands, download_urls, viewer_url }

    pLDDT interpretation:
        ≥ 90 Very high — backbone + side chains reliable
        70–90 Confident — backbone generally reliable
        50–70 Low — treat with caution
        < 50  Very low — likely intrinsically disordered
    """
    accession = BioValidator.validate_uniprot_accession(uniprot_accession)
    client    = await get_http_client()

    resp = await client.get(
        f"{ALPHAFOLD_BASE}/prediction/{accession}",
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:
        return {
            "error": (
                f"No AlphaFold structure for '{accession}'. "
                "The protein may be too long, fragmentary, or not yet predicted."
            ),
            "suggestion": "Try searching UniProt for alternative accessions for this protein.",
        }
    resp.raise_for_status()

    predictions = resp.json()
    if not predictions:
        return {"error": f"Empty prediction list for '{accession}'."}

    pred = predictions[-1]  # latest model

    return {
        "accession":               accession,
        "entry_id":                pred.get("entryId", ""),
        "model_created_date":      pred.get("modelCreatedDate", ""),
        "latest_version":          pred.get("latestVersion", ""),
        "organism_scientific":     pred.get("organismScientificName", ""),
        "uniprot_sequence_length": pred.get("uniprotSequenceLength", 0),
        "plddt_summary":           _summarise_plddt(pred.get("plddt", [])),
        "confidence_guide": {
            "very_high_90_100": "Backbone + side chains reliable",
            "confident_70_90":  "Backbone generally accurate",
            "low_50_70":        "Use with caution",
            "very_low_0_50":    "Likely intrinsically disordered",
        },
        "download_urls": {
            "pdb":       pred.get("pdbUrl",      ""),
            "mmcif":     pred.get("cifUrl",      ""),
            "bcif":      pred.get("bcifUrl",     ""),
            "pae_image": pred.get("paeImageUrl", ""),
            "pae_json":  pred.get("paeDocUrl",   ""),
        },
        "viewer_url":      f"https://alphafold.ebi.ac.uk/entry/{accession}",
        "alphafold_db_url":f"https://alphafold.ebi.ac.uk/entry/{accession}",
    }


def _summarise_plddt(plddt: list[float]) -> dict[str, Any]:
    """Compute descriptive statistics and confidence-band percentages."""
    if not plddt:
        return {"note": "No pLDDT scores available in this prediction."}

    arr = np.array(plddt, dtype=float)
    n   = len(arr)

    bands = {
        "very_high_90_100": int((arr >= 90).sum()),
        "confident_70_90":  int(((arr >= 70) & (arr < 90)).sum()),
        "low_50_70":        int(((arr >= 50) & (arr < 70)).sum()),
        "very_low_0_50":    int((arr < 50).sum()),
    }

    return {
        "mean":              round(float(arr.mean()),   2),
        "median":            round(float(np.median(arr)), 2),
        "std":               round(float(arr.std()),    2),
        "min":               round(float(arr.min()),    2),
        "max":               round(float(arr.max()),    2),
        "total_residues":    n,
        "band_counts":       bands,
        "band_percentages":  {k: round(v / n * 100, 1) for k, v in bands.items()},
        "overall_confidence":(
            "Very high" if arr.mean() >= 90 else
            "Confident" if arr.mean() >= 70 else
            "Low"       if arr.mean() >= 50 else
            "Very low"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RCSB PDB — experimental structure search
# ─────────────────────────────────────────────────────────────────────────────

@cached("uniprot")
@rate_limited("pdb")
@with_retry(max_attempts=3)
async def search_pdb_structures(
    query: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search RCSB PDB for experimental 3-D protein structures.

    Args:
        query:       Protein name, gene, organism, method, or PDB keyword.
        max_results: Results to return (1–50).

    Returns:
        { query, total_found, structures: [{ pdb_id, title, method,
          resolution_Å, deposition_date, organism, download urls }] }
    """
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    # ── Full-text search ──────────────────────────────────────────────────────
    payload = {
        "query": {
            "type":    "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort":     [{"sort_by": "score", "direction": "desc"}],
        },
    }
    search_resp = await client.post(
        PDB_SEARCH,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    if search_resp.status_code == 204:
        return {"query": query, "total_found": 0, "structures": []}
    search_resp.raise_for_status()

    data    = search_resp.json()
    pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
    total   = data.get("total_count", len(pdb_ids))

    # ── Fetch details for each hit (paralleled) ───────────────────────────────
    async def _fetch_entry(pdb_id: str) -> dict[str, Any] | None:
        try:
            r = await client.get(
                f"{PDB_DATA}/{pdb_id}",
                headers={"Accept": "application/json"},
            )
            if r.status_code != 200:
                return None
            d      = r.json()
            struct = d.get("struct", {})
            exptl  = (d.get("exptl") or [{}])[0]
            refine = (d.get("refine") or [{}])[0]
            info   = d.get("rcsb_accession_info", {})
            entry  = d.get("rcsb_entry_info",     {})
            return {
                "pdb_id":           pdb_id,
                "title":            struct.get("title", ""),
                "method":           exptl.get("method", ""),
                "resolution_A":     refine.get("ls_d_res_high"),
                "deposition_date":  info.get("deposit_date", ""),
                "release_date":     info.get("initial_release_date", ""),
                "organism":         entry.get("organism_name"),
                "chain_count":      entry.get("deposited_polymer_entity_instance_count"),
                "rcsb_url":         f"https://www.rcsb.org/structure/{pdb_id}",
                "download_pdb":     f"https://files.rcsb.org/download/{pdb_id}.pdb",
                "download_cif":     f"https://files.rcsb.org/download/{pdb_id}.cif",
            }
        except Exception as exc:
            logger.warning(f"PDB detail fetch failed for {pdb_id}: {exc}")
            return None

    import asyncio
    results_raw = await asyncio.gather(*[_fetch_entry(pid) for pid in pdb_ids])
    structures  = [r for r in results_raw if r is not None]

    return {
        "query":       query,
        "total_found": total,
        "returned":    len(structures),
        "structures":  structures,
    }
