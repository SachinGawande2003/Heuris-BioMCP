"""
BioMCP — Tier 2 Extended Databases  [FIXED v2.2]
=============================================
Fixes applied:
  - Bug #8: search_metabolomics now uses asyncio.gather with a semaphore
    instead of a sequential for-loop (was up to 200 sequential HTTP calls).
    Concurrent requests capped at 10 with asyncio.Semaphore.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

_BIOGRID_BASE = "https://webservice.thebiogrid.org"
_ORPHANET_BASE= "https://api.orphacode.org/EN/ClinicalEntity"
_GDC_BASE     = "https://api.gdc.cancer.gov"
_ENCODE_BASE  = "https://www.encodeproject.org"
_METABOLIGHTS = "https://www.ebi.ac.uk/metabolights/ws"
_UCSC_BASE    = "https://api.genome.ucsc.edu"


# ─────────────────────────────────────────────────────────────────────────────
# BioGRID
# ─────────────────────────────────────────────────────────────────────────────

@cached("biogrid")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_biogrid_interactions(
    gene_symbol:      str,
    interaction_type: str  = "physical",
    min_publications: int  = 1,
    max_results:      int  = 25,
    include_genetic:  bool = False,
) -> dict[str, Any]:
    """Retrieve curated protein-protein interactions from BioGRID."""
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    biogrid_key = os.getenv("BIOGRID_API_KEY", "").strip()
    if not biogrid_key:
        return {
            "gene": gene_symbol,
            "error": "BIOGRID_API_KEY is not configured. Set BIOGRID_API_KEY to query BioGRID.",
            "interactions": [],
        }
    client      = await get_http_client()

    params: dict[str, Any] = {
        "searchNames": True, "geneList": gene_symbol,
        "organism": 9606, "includeInteractors": True,
        "includeEvidence": True, "includePubmedId": True,
        "includeOfficialSymbol": True, "taxId": 9606,
        "max": max_results, "format": "json",
        "accesskey": biogrid_key,
    }

    resp = await client.get(f"{_BIOGRID_BASE}/interactions/", params=params,
                            headers={"Accept": "application/json"})
    if resp.status_code == 403:
        return {"gene": gene_symbol,
                "error": "BioGRID requires a free API key. Register at https://webservice.thebiogrid.org/",
                "interactions": []}
    if resp.status_code == 404:
        return {"gene": gene_symbol, "interactions": [], "interaction_count": 0}
    resp.raise_for_status()
    data = resp.json()

    interactions: list[dict[str, Any]] = []
    physical_count = genetic_count = 0
    for iid, record in list(data.items())[:max_results]:
        if not isinstance(record, dict):
            continue
        gene_a = record.get("OFFICIAL_SYMBOL_A", "")
        gene_b = record.get("OFFICIAL_SYMBOL_B", "")
        partner = gene_b if gene_a.upper() == gene_symbol else gene_a
        exp_type = record.get("EXPERIMENTAL_SYSTEM_TYPE", "physical")
        pubmed_ids = [str(p) for p in (record.get("PUBMED_ID", "") or "").split("|") if p]
        if exp_type == "physical":
            physical_count += 1
        else:
            genetic_count += 1
            if not include_genetic and interaction_type != "all":
                continue

        interactions.append({
            "partner_gene":             partner,
            "experimental_system":      record.get("EXPERIMENTAL_SYSTEM", ""),
            "experimental_system_type": exp_type,
            "publication_count":        len(pubmed_ids),
            "pubmed_ids":               pubmed_ids[:5],
            "biogrid_interaction_id":   iid,
            "biogrid_url":              f"https://thebiogrid.org/interaction/{iid}",
        })

    return {
        "gene":              gene_symbol,
        "interaction_count": len(data),
        "returned":          len(interactions),
        "interactions":      interactions,
        "network_stats": {
            "physical_interactions": physical_count,
            "genetic_interactions":  genetic_count,
            "hub_score":             round(min(1.0, len(interactions) / 100), 2),
        },
        "biogrid_gene_url": f"https://thebiogrid.org/search.php?search={gene_symbol}&organism=9606",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orphanet
# ─────────────────────────────────────────────────────────────────────────────

@cached("orphanet")
@rate_limited("default")
@with_retry(max_attempts=3)
async def search_orphan_diseases(
    gene_symbol:  str = "",
    disease_name: str = "",
    max_results:  int = 15,
) -> dict[str, Any]:
    """Search Orphanet for rare diseases associated with a gene or disease name."""
    if not gene_symbol and not disease_name:
        raise ValueError("Provide either gene_symbol or disease_name.")
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()
    query       = gene_symbol or disease_name

    if gene_symbol:
        gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)

    diseases: list[dict[str, Any]] = []
    resp = await client.get(
        f"https://api.orphacode.org/EN/ClinicalEntity/ApproximateName/{query}",
        headers={"accept": "application/json"},
    )
    if resp.status_code == 200:
        data = resp.json()
        for item in (data if isinstance(data, list) else [])[:max_results]:
            orpha_code = item.get("OrphaCode", "")
            name       = item.get("Preferred term", item.get("Name", ""))
            if orpha_code and name:
                diseases.append({
                    "orpha_code":   str(orpha_code),
                    "disease_name": name,
                    "disease_type": item.get("DisorderType", {}).get("Name", ""),
                    "orphanet_url": f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Expert={orpha_code}",
                })

    return {
        "query":        query,
        "total_found":  len(diseases),
        "diseases":     diseases[:max_results],
        "orphanet_search_url": (
            f"https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN"
            f"&Disease_Disease_Search_diseaseGroup={query}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TCGA via GDC
# ─────────────────────────────────────────────────────────────────────────────

@cached("tcga")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_tcga_expression(
    gene_symbol: str,
    cancer_type: str = "",
    max_cases:   int = 10,
) -> dict[str, Any]:
    """Retrieve gene expression data from TCGA tumor samples via GDC API."""
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_cases   = BioValidator.clamp_int(max_cases, 1, 50, "max_cases")
    client      = await get_http_client()

    file_filters: dict[str, Any] = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        ]
    }
    if cancer_type:
        file_filters["content"].append({
            "op": "=",
            "content": {"field": "cases.project.project_id", "value": cancer_type.upper()}
        })

    files_resp = await client.post(
        f"{_GDC_BASE}/files",
        json={
            "filters": file_filters,
            "fields":  "file_id,file_name,cases.project.project_id,cases.case_id,cases.samples.sample_type,file_size",
            "format":  "json", "size": max_cases,
        },
        headers={"Content-Type": "application/json"},
    )
    files_resp.raise_for_status()
    file_hits = files_resp.json().get("data", {}).get("hits", [])

    expression_files: list[dict[str, Any]] = []
    for hit in file_hits[:max_cases]:
        cases  = hit.get("cases", [{}])
        case   = cases[0] if cases else {}
        samples= case.get("samples", [{}])
        sample = samples[0] if samples else {}
        expression_files.append({
            "file_id":     hit.get("file_id", ""),
            "case_id":     case.get("case_id", ""),
            "project":     case.get("project", {}).get("project_id", ""),
            "sample_type": sample.get("sample_type", ""),
            "file_size_mb":round(hit.get("file_size", 0) / 1_000_000, 2),
            "download_url":f"https://api.gdc.cancer.gov/data/{hit.get('file_id', '')}",
        })

    return {
        "gene":              gene_symbol,
        "cancer_type_filter":cancer_type or "pan-cancer",
        "total_files_found": files_resp.json().get("data", {}).get("pagination", {}).get("total", 0),
        "expression_files":  expression_files,
        "gdc_url":           f"https://portal.gdc.cancer.gov/exploration?filters={{\"genes.symbol\":\"{gene_symbol}\"}}",
        "download_note":     "Use GDC Data Transfer Tool: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CellMarker
# ─────────────────────────────────────────────────────────────────────────────

@cached("cellmarker")
@rate_limited("default")
@with_retry(max_attempts=3)
async def search_cellmarker(
    gene_symbol: str = "",
    tissue:      str = "",
    cell_type:   str = "",
    species:     str = "Human",
    max_results: int = 20,
) -> dict[str, Any]:
    """Search CellMarker 2.0 for validated cell type markers."""
    if not any([gene_symbol, tissue, cell_type]):
        raise ValueError("Provide at least one of: gene_symbol, tissue, or cell_type.")
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    try:
        resp = await client.get(
            "http://xteam.xbio.top/CellMarker/download/all_cell_markers.txt",
            timeout=20.0,
        )
        if resp.status_code != 200:
            raise ValueError("CellMarker unavailable")
        lines   = resp.text.strip().split("\n")
        headers = lines[0].split("\t") if lines else []
        results: list[dict[str, Any]] = []

        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < len(headers):
                continue
            row = dict(zip(headers, cols, strict=False))
            if species.lower() not in row.get("speciesType", "Human").lower():
                continue
            if tissue and tissue.lower() not in row.get("tissueType", "").lower():
                continue
            if cell_type and cell_type.lower() not in row.get("cellName", "").lower():
                continue
            if gene_symbol and gene_symbol.upper() not in row.get("cellMarker", "").upper():
                continue

            marker_list = [m.strip() for m in row.get("cellMarker", "").split(",") if m.strip()]
            results.append({
                "cell_name":        row.get("cellName", ""),
                "tissue_type":      row.get("tissueType", ""),
                "cell_marker_list": marker_list,
                "marker_count":     len(marker_list),
                "pmid":             row.get("PMID", ""),
            })
            if len(results) >= max_results:
                break

    except Exception as exc:
        return {
            "query": {"gene_symbol": gene_symbol, "tissue": tissue, "cell_type": cell_type},
            "error": f"CellMarker unavailable: {exc}",
            "direct_url": "http://xteam.xbio.top/CellMarker/",
        }

    return {
        "query": {"gene_symbol": gene_symbol, "tissue": tissue, "cell_type": cell_type},
        "total_found":       len(results),
        "markers":           results[:max_results],
        "gene_expressed_in": list({r["cell_name"] for r in results if gene_symbol})[:20],
        "cellmarker_url":    "http://xteam.xbio.top/CellMarker/",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────────────────────────────────────────

@cached("encode")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_encode_regulatory(
    gene_symbol:  str,
    element_type: str = "all",
    biosample:    str = "",
    max_results:  int = 15,
) -> dict[str, Any]:
    """Search ENCODE for regulatory elements associated with a gene."""
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    params: dict[str, Any] = {
        "searchTerm": gene_symbol, "type": "Experiment",
        "status": "released",
        "replicates.library.biosample.donor.organism.scientific_name": "Homo sapiens",
        "format": "json", "limit": max_results, "frame": "object",
    }
    assay_map = {
        "promoter":       "CAGE",
        "enhancer":       "ChIP-seq",
        "CTCF":           "ChIP-seq",
        "TF_binding":     "ChIP-seq",
        "open_chromatin": "ATAC-seq",
    }
    if element_type != "all" and element_type in assay_map:
        params["assay_term_name"] = assay_map[element_type]
    if biosample:
        params["biosample_ontology.term_name"] = biosample

    resp = await client.get(
        f"{_ENCODE_BASE}/search/", params=params,
        headers={"Accept": "application/json", "User-Agent": "BioMCP/2.2"},
    )
    if resp.status_code != 200:
        return {
            "gene": gene_symbol, "error": f"ENCODE returned {resp.status_code}",
            "regulatory_elements": [],
            "encode_search_url": f"https://www.encodeproject.org/search/?searchTerm={gene_symbol}",
        }

    data  = resp.json()
    graph = data.get("@graph", [])
    elements: list[dict[str, Any]] = []
    assay_counts: dict[str, int] = {}

    for exp in graph[:max_results]:
        assay = exp.get("assay_term_name", "")
        target_obj = exp.get("target", {})
        target = target_obj.get("label", "") if isinstance(target_obj, dict) else ""
        bs_ont = exp.get("biosample_ontology", {})
        biosample_name = bs_ont.get("term_name", "") if isinstance(bs_ont, dict) else ""
        accession = exp.get("accession", "")
        assay_counts[assay] = assay_counts.get(assay, 0) + 1

        elements.append({
            "accession":    accession,
            "assay":        assay,
            "target":       target,
            "biosample":    biosample_name,
            "encode_url":   f"https://www.encodeproject.org/experiments/{accession}/",
        })

    return {
        "gene":                gene_symbol,
        "element_type_filter": element_type,
        "total_experiments":   data.get("total", len(elements)),
        "regulatory_elements": elements,
        "assay_summary":       assay_counts,
        "encode_gene_url":     f"https://www.encodeproject.org/search/?searchTerm={gene_symbol}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MetaboLights — FIX #8: sequential loop → concurrent with semaphore
# ─────────────────────────────────────────────────────────────────────────────

@cached("metabolights")
@rate_limited("default")
@with_retry(max_attempts=3)
async def search_metabolomics(
    gene_symbol: str = "",
    metabolite:  str = "",
    disease:     str = "",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search MetaboLights for metabolomics studies.

    FIX #8: Replaced the sequential for-loop (up to 200 individual HTTP calls)
    with concurrent asyncio.gather capped at 10 simultaneous requests via
    asyncio.Semaphore.  Worst-case latency drops from ~20s → ~2s.
    """
    if not any([gene_symbol, metabolite, disease]):
        raise ValueError("Provide at least one of: gene_symbol, metabolite, or disease.")
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    query = " ".join(filter(None, [gene_symbol, metabolite, disease]))

    # Fetch the full study list
    resp = await client.get(f"{_METABOLIGHTS}/study/list",
                             headers={"Accept": "application/json"})
    studies: list[dict[str, Any]] = []

    if resp.status_code == 200:
        study_list = resp.json().get("content", [])
        # Limit to first 200 candidates for scan
        candidates = study_list[:200]

        # ── FIX #8: concurrent fetch with semaphore ───────────────────────
        sem = asyncio.Semaphore(10)   # max 10 simultaneous requests

        async def _fetch_study_title(study_id: str) -> tuple[str, str] | None:
            async with sem:
                try:
                    r = await client.get(
                        f"{_METABOLIGHTS}/study/{study_id}/title",
                        headers={"Accept": "application/json"},
                        timeout=5.0,
                    )
                    if r.status_code != 200:
                        return None
                    title = r.json().get("content", "")
                    return (study_id, title) if title else None
                except Exception:
                    return None

        raw_titles = await asyncio.gather(
            *[_fetch_study_title(sid) for sid in candidates],
            return_exceptions=True,
        )

        for item in raw_titles:
            if len(studies) >= max_results:
                break
            if not item or isinstance(item, Exception):
                continue
            study_id, title = item
            if query and not any(
                q.lower() in title.lower()
                for q in query.split() if len(q) > 3
            ):
                continue
            studies.append({
                "study_id":         study_id,
                "title":            title[:120],
                "metabolights_url": f"https://www.ebi.ac.uk/metabolights/study/{study_id}",
                "download_url":     f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/",
            })

    return {
        "query": {"gene_symbol": gene_symbol, "metabolite": metabolite, "disease": disease},
        "total_found":     len(studies),
        "studies":         studies[:max_results],
        "metabolights_url":"https://www.ebi.ac.uk/metabolights/",
        "analysis_tools": [
            "MetaboAnalyst (web) — comprehensive metabolomics analysis",
            "XCMS — LC-MS data processing",
            "mzMine — mass spectrometry data analysis",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# UCSC Genome Browser
# ─────────────────────────────────────────────────────────────────────────────

@cached("ucsc")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_ucsc_splice_variants(
    gene_symbol: str,
    genome:      str  = "hg38",
    include_alt: bool = True,
) -> dict[str, Any]:
    """Retrieve alternative splicing isoforms and UTR annotations from UCSC."""
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    client      = await get_http_client()
    isoforms: list[dict[str, Any]] = []

    resp = await client.get(
        f"{_UCSC_BASE}/search",
        params={"search": gene_symbol, "genome": genome},
        headers={"Accept": "application/json"},
    )

    if resp.status_code == 200:
        for match in resp.json().get("results", [])[:3]:
            chrom = match.get("chrom", "")
            start = match.get("chromStart", 0)
            end   = match.get("chromEnd",   0)
            if not (chrom and start and end):
                continue
            tx_resp = await client.get(
                f"{_UCSC_BASE}/getData/track",
                params={"genome": genome, "track": "knownGene",
                        "chrom": chrom, "start": start, "end": end},
                headers={"Accept": "application/json"},
            )
            if tx_resp.status_code != 200:
                continue
            for tx in tx_resp.json().get("knownGene", [])[:20]:
                if gene_symbol.upper() not in tx.get("name2", "").upper():
                    continue
                exon_starts = tx.get("exonStarts", [])
                tx_start = tx.get("txStart",  start)
                tx_end   = tx.get("txEnd",    end)
                cds_start= tx.get("cdsStart", tx_start)
                cds_end  = tx.get("cdsEnd",   tx_end)
                isoforms.append({
                    "transcript_id":   tx.get("name", ""),
                    "chromosome":      chrom,
                    "strand":          tx.get("strand", "+"),
                    "tx_start":        tx_start, "tx_end": tx_end,
                    "cds_start":       cds_start, "cds_end": cds_end,
                    "exon_count":      len(exon_starts),
                    "total_length_bp": tx_end - tx_start,
                    "cds_length_bp":   cds_end - cds_start,
                    "utr5_length_bp":  abs(cds_start - tx_start),
                    "utr3_length_bp":  abs(tx_end - cds_end),
                    "is_coding":       cds_start < cds_end,
                    "ucsc_url":        f"https://genome.ucsc.edu/cgi-bin/hgGene?hgg_gene={tx.get('name','')}&db={genome}",
                })
            break

    canonical = max(isoforms, key=lambda x: x.get("cds_length_bp", 0)) if isoforms else {}
    exon_counts = [iso.get("exon_count", 0) for iso in isoforms]

    return {
        "gene":             gene_symbol,
        "genome_assembly":  genome,
        "canonical_isoform":canonical,
        "total_isoforms":   len(isoforms),
        "isoforms":         isoforms[:20],
        "splicing_summary": {
            "total_isoforms":   len(isoforms),
            "coding_isoforms":  sum(1 for i in isoforms if i.get("is_coding")),
            "min_exon_count":   min(exon_counts) if exon_counts else 0,
            "max_exon_count":   max(exon_counts) if exon_counts else 0,
            "complexity":       (
                "Highly alternatively spliced" if len(isoforms) > 10 else
                "Moderately spliced"           if len(isoforms) > 3  else
                "Simple gene structure"
            ),
        },
        "ucsc_gene_url": f"https://genome.ucsc.edu/cgi-bin/hgGene?hgg_gene={gene_symbol}&db={genome}",
    }
