"""
BioMCP — Advanced Tools  [FIXED v2.2]
========================
Fixes applied:
  - Bug #6: Added @cached("multi_omics") to multi_omics_gene_report.
    Without it every repeated call for the same gene re-fired all 7 parallel
    queries, wasting API quota and adding multi-second latency.
  - Updated HCA endpoint comment (Bug #12 note — current endpoint same domain)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from biomcp.utils import (
    _NCBI_SERVICE,
    BioValidator,
    cached,
    get_http_client,
    ncbi_params,
    rate_limited,
    strip_cache_metadata,
    with_retry,
)

_MultiOmicsProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

CLINTRIALS_BASE = "https://clinicaltrials.gov/api/v2"
ENSEMBL_BASE    = "https://rest.ensembl.org"
GEO_BASE        = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HCA_BASE        = "https://service.azul.data.humancellatlas.org"
ANTHROPIC_MESSAGES_API = "https://api.anthropic.com/v1/messages"

_CT_HEADERS: dict[str, str] = {
    "Accept":          "application/json",
    "User-Agent":      (
        "Mozilla/5.0 (compatible; Heuris-BioMCP/2.2; "
        "+https://github.com/SachinGawande2003/Heuris-BioMCP)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_VALID_STATUSES = frozenset({
    "RECRUITING", "COMPLETED", "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING", "TERMINATED", "ALL",
})
_VALID_PHASES = frozenset({"PHASE1", "PHASE2", "PHASE3", "PHASE4"})
_CT_403_RETRY_ATTEMPTS = int(os.getenv("BIOMCP_CT_403_RETRY_ATTEMPTS", "3"))
_CT_403_RETRY_BASE_DELAY_SECONDS = float(os.getenv("BIOMCP_CT_403_RETRY_BASE_DELAY", "2"))
_CT_403_RETRY_MAX_DELAY_SECONDS = float(os.getenv("BIOMCP_CT_403_RETRY_MAX_DELAY", "8"))
_CLAUDE_SYNTHESIS_ENABLED = os.getenv("BIOMCP_ENABLE_CLAUDE_SYNTHESIS", "1").strip().lower()
_CLAUDE_SYNTHESIS_MODEL = os.getenv("BIOMCP_CLAUDE_SYNTHESIS_MODEL", "claude-sonnet-4-20250514")
_CLAUDE_SYNTHESIS_MAX_TOKENS = int(os.getenv("BIOMCP_CLAUDE_SYNTHESIS_MAX_TOKENS", "700"))


async def _clinical_trials_get_with_403_retry(
    client: Any,
    path: str,
    *,
    params: dict[str, Any],
) -> Any:
    response = None
    for attempt in range(_CT_403_RETRY_ATTEMPTS):
        resp = await client.get(f"{CLINTRIALS_BASE}{path}", params=params, headers=_CT_HEADERS)
        response = resp
        if resp.status_code != 403 or attempt == _CT_403_RETRY_ATTEMPTS - 1:
            return resp
        delay_s = min(
            _CT_403_RETRY_BASE_DELAY_SECONDS * (2 ** attempt),
            _CT_403_RETRY_MAX_DELAY_SECONDS,
        )
        logger.warning(
            f"[ClinicalTrials] 403 rate limit on {path}; retrying in {delay_s:.1f}s "
            f"(attempt {attempt + 1}/{_CT_403_RETRY_ATTEMPTS})"
        )
        await asyncio.sleep(delay_s)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov
# ─────────────────────────────────────────────────────────────────────────────

@cached("clinical_trials")
@rate_limited("clinical_trials")
@with_retry(max_attempts=3)
async def search_clinical_trials(
    query:       str,
    status:      str       = "RECRUITING",
    phase:       str | None = None,
    max_results: int        = 10,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov for clinical studies."""
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}, got '{status}'")
    if phase and phase not in _VALID_PHASES:
        raise ValueError(f"phase must be one of {_VALID_PHASES}, got '{phase}'")

    client = await get_http_client()
    params: dict[str, Any] = {
        "query.term": query, "pageSize": max_results,
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,StudyType,"
            "EnrollmentCount,StartDate,PrimaryCompletionDate,"
            "Condition,InterventionName,InterventionType,"
            "BriefSummary,EligibilityCriteria,"
            "LocationCity,LocationCountry,LeadSponsorName"
        ),
    }
    if status != "ALL":
        params["filter.overallStatus"] = status
    if phase:
        params["filter.phase"] = phase

    resp = await _clinical_trials_get_with_403_retry(client, "/studies", params=params)
    if resp.status_code == 403:
        return {
            "error": (
                "ClinicalTrials.gov returned repeated 403 responses after exponential backoff. "
                "Retry shortly."
            ),
            "query": query, "studies": [], "total_found": 0,
        }
    resp.raise_for_status()
    data = resp.json()

    studies: list[dict[str, Any]] = []
    for study in data.get("studies", []):
        proto      = study.get("protocolSection", {})
        id_mod     = proto.get("identificationModule",    {})
        status_mod = proto.get("statusModule",            {})
        desc_mod   = proto.get("descriptionModule",       {})
        design_mod = proto.get("designModule",            {})
        cond_mod   = proto.get("conditionsModule",        {})
        interv_mod = proto.get("armsInterventionsModule", {})
        loc_mod    = proto.get("contactsLocationsModule", {})
        spon_mod   = proto.get("sponsorCollaboratorsModule", {})

        nct_id    = id_mod.get("nctId", "")
        locations = list({
            f"{location.get('city', '')}, {location.get('country', '')}".strip(", ")
            for location in (loc_mod.get("locations") or [])[:8]
            if location.get("city") or location.get("country")
        })
        interventions = [
            {"name": i.get("interventionName", ""), "type": i.get("interventionType", "")}
            for i in (interv_mod.get("interventions") or [])[:6]
        ]
        studies.append({
            "nct_id":              nct_id,
            "title":               id_mod.get("briefTitle", ""),
            "status":              status_mod.get("overallStatus", ""),
            "phase":               design_mod.get("phases", []),
            "conditions":          cond_mod.get("conditions", []),
            "interventions":       interventions,
            "enrollment":          design_mod.get("enrollmentInfo", {}).get("count", ""),
            "start_date":          status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date":     status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
            "summary":             (desc_mod.get("briefSummary") or "")[:600],
            "sponsor":             spon_mod.get("leadSponsor", {}).get("name", ""),
            "locations":           locations,
            "clinicaltrials_url":  f"https://clinicaltrials.gov/study/{nct_id}",
        })

    return {
        "query":         query,
        "status_filter": status,
        "total_found":   data.get("totalCount", len(studies)),
        "returned":      len(studies),
        "studies":       studies,
    }


@cached("clinical_trials")
@rate_limited("clinical_trials")
@with_retry(max_attempts=3)
async def get_trial_details(nct_id: str) -> dict[str, Any]:
    """Retrieve full protocol details for one clinical trial."""
    nct_id = BioValidator.validate_nct_id(nct_id)
    client = await get_http_client()

    resp = await _clinical_trials_get_with_403_retry(
        client,
        f"/studies/{nct_id}",
        params={"format": "json"},
    )
    if resp.status_code == 403:
        return {
            "error": (
                "ClinicalTrials.gov returned repeated 403 responses after exponential backoff. "
                "Retry shortly."
            )
        }
    if resp.status_code == 404:
        return {"error": f"Trial '{nct_id}' not found."}
    resp.raise_for_status()

    proto   = resp.json().get("protocolSection", {})
    out_mod = proto.get("outcomesModule",          {})
    arms_mod= proto.get("armsInterventionsModule", {})
    elig_mod= proto.get("eligibilityModule",       {})

    return {
        "nct_id":   nct_id,
        "primary_outcomes": [
            {"measure": o.get("measure", ""), "time_frame": o.get("timeFrame", "")}
            for o in (out_mod.get("primaryOutcomes") or [])
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure", ""), "time_frame": o.get("timeFrame", "")}
            for o in (out_mod.get("secondaryOutcomes") or [])[:6]
        ],
        "arms": [
            {"label": a.get("armGroupLabel", ""), "type": a.get("armGroupType", ""),
             "description": (a.get("description") or "")[:300]}
            for a in (arms_mod.get("armGroups") or [])
        ],
        "eligibility": {
            "criteria": (elig_mod.get("eligibilityCriteria") or "")[:1_500],
            "min_age":  elig_mod.get("minimumAge", ""),
            "max_age":  elig_mod.get("maximumAge", ""),
            "sex":      elig_mod.get("sex", ""),
        },
        "clinicaltrials_url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensembl gene variants
# ─────────────────────────────────────────────────────────────────────────────

@cached("ensembl")
@rate_limited("ensembl")
@with_retry(max_attempts=3)
async def get_gene_variants(
    gene_symbol:      str,
    consequence_type: str = "missense_variant",
    max_results:      int = 20,
) -> dict[str, Any]:
    """Retrieve genetic variants in a gene from Ensembl."""
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    xref_resp = await client.get(
        f"{ENSEMBL_BASE}/xrefs/symbol/homo_sapiens/{gene_symbol}",
        headers={"Accept": "application/json"},
    )
    xref_resp.raise_for_status()
    gene_ids = [e["id"] for e in xref_resp.json() if e.get("type") == "gene"]
    if not gene_ids:
        return {"gene": gene_symbol, "variants": [],
                "error": f"'{gene_symbol}' not found in Ensembl."}

    gene_id  = gene_ids[0]
    lookup = await client.get(
        f"{ENSEMBL_BASE}/lookup/id/{gene_id}",
        headers={"Accept": "application/json"},
    )
    lookup.raise_for_status()
    info     = lookup.json()
    chrom    = info.get("seq_region_name", "")
    start    = info.get("start", 0)
    end      = min(info.get("end", 0), start + 200_000)

    var_resp = await client.get(
        f"{ENSEMBL_BASE}/overlap/region/human/{chrom}:{start}-{end}/variation",
        headers={"Accept": "application/json"},
        params={"feature": "variation"},
    )
    if var_resp.status_code == 404:
        return {
            "gene": gene_symbol,
            "ensembl_gene_id": gene_id,
            "chromosome": chrom,
            "total_variants": 0,
            "returned": 0,
            "variants": [],
            "note": f"No Ensembl overlap variants found for '{gene_symbol}'.",
        }
    var_resp.raise_for_status()
    all_vars = var_resp.json()

    variants: list[dict[str, Any]] = []
    for v in all_vars[:max_results]:
        vid = v.get("id", "")
        variants.append({
            "id":                   vid,
            "chromosome":           chrom,
            "start":                v.get("start", ""),
            "alleles":              v.get("alleles", []),
            "consequence_types":    v.get("consequence_type", []),
            "clinical_significance":v.get("clinical_significance", []),
            "ensembl_url": (
                f"https://www.ensembl.org/Homo_sapiens/Variation/Summary?v={vid}"
                if vid.startswith("rs") else ""
            ),
        })

    return {
        "gene":            gene_symbol,
        "ensembl_gene_id": gene_id,
        "chromosome":      chrom,
        "total_variants":  len(all_vars),
        "returned":        len(variants),
        "variants":        variants,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NCBI GEO
# ─────────────────────────────────────────────────────────────────────────────

@cached("expression")
@rate_limited(_NCBI_SERVICE)
@with_retry(max_attempts=3)
async def search_gene_expression(
    gene_symbol:  str,
    condition:    str = "",
    max_datasets: int = 10,
) -> dict[str, Any]:
    """Search NCBI GEO for gene expression datasets."""
    gene_symbol  = BioValidator.validate_gene_symbol(gene_symbol)
    max_datasets = BioValidator.clamp_int(max_datasets, 1, 50, "max_datasets")
    client       = await get_http_client()

    q = gene_symbol + (f" AND {condition}" if condition else "")
    search = await client.get(
        f"{GEO_BASE}/esearch.fcgi",
        params=ncbi_params({"db": "gds", "term": q, "retmax": max_datasets}),
    )
    search.raise_for_status()
    result = search.json().get("esearchresult", {})
    ids    = result.get("idlist", [])
    total  = int(result.get("count", 0))

    if not ids:
        return {"gene": gene_symbol, "condition": condition, "total_found": 0, "datasets": []}

    summ = await client.get(
        f"{GEO_BASE}/esummary.fcgi",
        params=ncbi_params({"db": "gds", "id": ",".join(ids)}),
    )
    summ.raise_for_status()
    summaries = summ.json().get("result", {})

    datasets: list[dict[str, Any]] = []
    for uid in ids:
        d = summaries.get(uid, {})
        if not d:
            continue
        acc = d.get("accession", "")
        datasets.append({
            "geo_accession": acc,
            "title":         d.get("title",    ""),
            "summary":       (d.get("summary") or "")[:400],
            "organism":      d.get("organism") or d.get("taxon") or d.get("orgname", ""),
            "platform":      d.get("gpl",       ""),
            "n_samples":     d.get("n_samples", 0),
            "pubmed_ids":    d.get("pubmedids", []),
            "geo_url":       f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
        })

    return {
        "gene":        gene_symbol,
        "condition":   condition,
        "total_found": total,
        "returned":    len(datasets),
        "datasets":    datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human Cell Atlas
# ─────────────────────────────────────────────────────────────────────────────

@cached("expression")
@rate_limited("hca")
@with_retry(max_attempts=2)
async def search_scrna_datasets(
    tissue:      str,
    species:     str = "Homo sapiens",
    max_results: int = 10,
) -> dict[str, Any]:
    """Search Human Cell Atlas for single-cell RNA-seq datasets."""
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    client      = await get_http_client()

    resp = await client.get(
        f"{HCA_BASE}/index/projects",
        params={
            "catalog":      "dcp2",
            "size":         max_results,
            "organ":        tissue,
            "genusSpecies": species,
            "sort":         "projectTitle",
            "order":        "asc",
        },
        headers={"Accept": "application/json"},
    )

    if resp.status_code not in (200, 206):
        return {
            "tissue": tissue, "species": species,
            "message": (
                "HCA may be temporarily unavailable. "
                "Visit https://data.humancellatlas.org/ directly."
            ),
            "datasets": [],
        }

    data = resp.json()
    hits = data.get("hits", [])
    datasets: list[dict[str, Any]] = []

    for hit in hits:
        proj  = (hit.get("projects") or [{}])[0]
        cells = (hit.get("cellSuspensions") or [{}])[0]
        protos = hit.get("protocols", [])
        techs = list({
            (p.get("libraryConstructionApproach") or [None])[0]
            for p in protos
            if (p.get("libraryConstructionApproach") or [None])[0]
        })
        pid = proj.get("projectId", "")
        datasets.append({
            "project_id":              pid,
            "title":                   proj.get("projectTitle", ""),
            "cell_count":              cells.get("totalCells", 0),
            "donor_count":             len(hit.get("donorOrganisms", [])),
            "sequencing_technologies": techs,
            "hca_url": f"https://data.humancellatlas.org/explore/projects/{pid}",
        })

    return {
        "tissue":      tissue,
        "species":     species,
        "total_found": data.get("pagination", {}).get("total", len(datasets)),
        "returned":    len(datasets),
        "datasets":    datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Omics Gene Report — FIX #6: Added @cached("multi_omics")
# ─────────────────────────────────────────────────────────────────────────────

def _compact_multi_omics_literature(literature: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_publications": literature.get("total_found", 0),
        "recent_papers": [
            {
                "pmid": article["pmid"],
                "title": article["title"],
                "year": article["year"],
                "journal": article["journal"],
            }
            for article in literature.get("articles", [])
        ],
    }


def _compact_multi_omics_layer(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    if label == "literature":
        return _compact_multi_omics_literature(payload)
    if label == "genomics":
        aliases = payload.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        summary = str(payload.get("summary", "") or "")
        return {
            "symbol": payload.get("symbol", ""),
            "name": payload.get("name") or payload.get("description", ""),
            "organism": payload.get("organism", ""),
            "chromosome": payload.get("chromosome", ""),
            "location": payload.get("map_location") or payload.get("maplocation", ""),
            "summary": summary[:280],
            "aliases": aliases[:5],
        }
    if label == "reactome":
        pathways = payload.get("pathways", [])
        if not isinstance(pathways, list):
            pathways = []
        return {
            "total_pathways": payload.get("total_pathways", len(pathways)),
            "top_pathways": [
                {
                    "name": pathway.get("name", ""),
                    "species": pathway.get("species", ""),
                    "found_entities": pathway.get("found_entities", 0),
                    "fdr": pathway.get("fdr"),
                }
                for pathway in pathways[:5]
                if isinstance(pathway, dict)
            ],
        }
    if label == "drug_targets":
        drugs = payload.get("drugs", [])
        if not isinstance(drugs, list):
            drugs = []
        return {
            "total_drugs": payload.get("total_drugs", len(drugs)),
            "top_drugs": [
                {
                    "name": drug.get("name", ""),
                    "mechanism": drug.get("mechanism", ""),
                    "phase": drug.get("max_phase"),
                }
                for drug in drugs[:5]
                if isinstance(drug, dict)
            ],
        }
    if label == "disease_associations":
        associations = payload.get("associations", [])
        if not isinstance(associations, list):
            associations = []
        return {
            "total_associations": payload.get("total_associations", len(associations)),
            "top_diseases": [
                {
                    "disease_name": association.get("disease_name", ""),
                    "association_score": association.get("association_score")
                    or association.get("score")
                    or association.get("gda_score"),
                }
                for association in associations[:5]
                if isinstance(association, dict)
            ],
        }
    if label == "expression":
        datasets = payload.get("datasets", [])
        if not isinstance(datasets, list):
            datasets = []
        return {
            "total_datasets": payload.get("total_found", len(datasets)),
            "top_datasets": [
                {
                    "gse_id": dataset.get("gse_id") or dataset.get("accession", ""),
                    "title": dataset.get("title", ""),
                    "platform": dataset.get("platform", ""),
                }
                for dataset in datasets[:5]
                if isinstance(dataset, dict)
            ],
        }
    if label == "clinical_trials":
        studies = payload.get("studies", [])
        if not isinstance(studies, list):
            studies = []
        return {
            "total_trials": payload.get("total_found", len(studies)),
            "top_trials": [
                {
                    "nct_id": study.get("nct_id", ""),
                    "brief_title": study.get("brief_title", ""),
                    "status": study.get("status", ""),
                    "phase": study.get("phase", ""),
                }
                for study in studies[:5]
                if isinstance(study, dict)
            ],
        }
    return payload


def _shape_multi_omics_layer(
    label: str,
    payload: dict[str, Any],
    detail_level: str,
) -> dict[str, Any]:
    if detail_level == "full":
        return payload
    if detail_level == "compact":
        return _compact_multi_omics_layer(label, payload)
    if label == "literature":
        return _compact_multi_omics_literature(payload)
    return payload


def _claude_synthesis_enabled() -> bool:
    return _CLAUDE_SYNTHESIS_ENABLED not in {"0", "false", "off", "no"}


def _anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def _build_multi_omics_synthesis_prompt(
    gene_symbol: str,
    layers: dict[str, dict[str, Any]],
) -> str:
    compact_layers = {
        label: _compact_multi_omics_layer(label, strip_cache_metadata(payload))
        for label, payload in layers.items()
        if isinstance(payload, dict)
    }
    return (
        "You are writing a synthesis for a bioinformatics MCP tool.\n"
        "Use only the provided JSON facts.\n"
        f"Gene: {gene_symbol}\n"
        "Return strict JSON with keys:\n"
        "narrative_paragraphs: array of exactly 3 strings\n"
        "clinical_implications: array of 2 to 5 concise strings\n"
        "synthesis_confidence: number from 0.0 to 1.0\n"
        "Do not include markdown fences or extra text.\n"
        f"Facts JSON:\n{json.dumps(compact_layers, sort_keys=True, default=str)}"
    )


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content", [])
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


async def _generate_multi_omics_synthesis(
    gene_symbol: str,
    layers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    api_key = _anthropic_api_key()
    if not _claude_synthesis_enabled():
        return {
            "status": "disabled",
            "reason": "Claude synthesis is disabled by configuration.",
        }
    if not api_key:
        return {
            "status": "disabled",
            "reason": "ANTHROPIC_API_KEY is not configured.",
        }

    client = await get_http_client()
    resp = await client.post(
        ANTHROPIC_MESSAGES_API,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": _CLAUDE_SYNTHESIS_MODEL,
            "max_tokens": _CLAUDE_SYNTHESIS_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": _build_multi_omics_synthesis_prompt(gene_symbol, layers),
                }
            ],
        },
    )
    if resp.status_code in {401, 403}:
        return {
            "status": "failed",
            "error": "Anthropic API rejected the synthesis request. Check ANTHROPIC_API_KEY.",
            "status_code": resp.status_code,
        }
    resp.raise_for_status()

    response_payload = resp.json()
    text = _extract_anthropic_text(response_payload)
    if not text:
        return {
            "status": "failed",
            "error": "Anthropic API returned no text blocks for synthesis.",
            "model": _CLAUDE_SYNTHESIS_MODEL,
        }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        return {
            "status": "generated",
            "model": _CLAUDE_SYNTHESIS_MODEL,
            "generated_at": int(time.time()),
            "narrative_paragraphs": paragraphs[:3],
            "clinical_implications": [],
            "raw_text": text,
        }

    paragraphs = parsed.get("narrative_paragraphs", [])
    implications = parsed.get("clinical_implications", [])
    return {
        "status": "generated",
        "model": _CLAUDE_SYNTHESIS_MODEL,
        "generated_at": int(time.time()),
        "narrative_paragraphs": [str(item).strip() for item in paragraphs[:3] if str(item).strip()],
        "clinical_implications": [str(item).strip() for item in implications[:5] if str(item).strip()],
        "synthesis_confidence": parsed.get("synthesis_confidence"),
    }


async def _multi_omics_gene_report_impl(
    gene_symbol: str,
    detail_level: str = "standard",
    include_synthesis: bool = True,
    progress_callback: _MultiOmicsProgressCallback | None = None,
) -> dict[str, Any]:
    from biomcp.tools.ncbi import get_gene_info, search_pubmed
    from biomcp.tools.pathways import (
        get_drug_targets,
        get_gene_disease_associations,
        get_reactome_pathways,
    )

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    if detail_level not in {"compact", "standard", "full"}:
        raise ValueError("detail_level must be one of compact, standard, or full.")
    logger.info(f"[Multi-Omics] Generating report for {gene_symbol}")

    layer_calls = [
        ("genomics", get_gene_info(gene_symbol)),
        ("literature", search_pubmed(f"{gene_symbol}[Gene] function mechanism review", max_results=5)),
        ("reactome", get_reactome_pathways(gene_symbol)),
        ("drug_targets", get_drug_targets(gene_symbol, max_results=10)),
        ("disease_associations", get_gene_disease_associations(gene_symbol, max_results=10)),
        ("expression", search_gene_expression(gene_symbol, max_datasets=5)),
        ("clinical_trials", search_clinical_trials(gene_symbol, max_results=5)),
    ]

    async def _run_layer(label: str, awaitable: Awaitable[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        try:
            result = strip_cache_metadata(await awaitable)
            if isinstance(result, dict):
                result = _shape_multi_omics_layer(label, result, detail_level)
            return label, result
        except Exception as exc:
            logger.warning(f"[Multi-Omics] {label} failed: {exc}")
            return label, {"error": str(exc), "status": "failed"}

    pending = [
        asyncio.create_task(_run_layer(label, awaitable))
        for label, awaitable in layer_calls
    ]

    completed_layers: dict[str, Any] = {}
    for pending_task in asyncio.as_completed(pending):
        label, layer_result = await pending_task
        completed_layers[label] = layer_result
        if progress_callback is not None:
            try:
                await progress_callback(label, layer_result)
            except Exception as exc:
                logger.debug(f"[Multi-Omics] Progress callback failed for {label}: {exc}")

    ordered_layers = {
        label: completed_layers[label]
        for label, _ in layer_calls
        if label in completed_layers
    }
    synthesis = (
        await _generate_multi_omics_synthesis(gene_symbol, ordered_layers)
        if include_synthesis
        else {"status": "disabled", "reason": "Synthesis explicitly disabled for this call."}
    )

    return {
        "gene":          gene_symbol,
        "report_type":   "multi_omics_integrated",
        "detail_level":  detail_level,
        "layers":        ordered_layers,
        "synthesis":     synthesis,
        "data_sources": [
            "NCBI Gene", "PubMed", "Reactome", "ChEMBL",
            "Open Targets", "NCBI GEO", "ClinicalTrials.gov",
        ],
        "note": (
            "Queries run in parallel. Result cached for 1 hour. "
            "Layers with 'status: failed' can be retried individually."
        ),
    }

@cached("multi_omics")       # FIX #6: was missing — every call re-fired all 7 queries
@rate_limited("default")
async def multi_omics_gene_report(
    gene_symbol: str,
    detail_level: str = "compact",
    include_synthesis: bool = True,
) -> dict[str, Any]:
    """
    Generate a comprehensive multi-omics report for a gene.

    Queries 7 databases simultaneously and returns an integrated report.
    """
    return await _multi_omics_gene_report_impl(
        gene_symbol,
        detail_level=detail_level,
        include_synthesis=include_synthesis,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Neuroimaging datasets
# ─────────────────────────────────────────────────────────────────────────────

_OPENNEURO_GQL = """
query SearchDatasets($q: String!, $first: Int!) {
  datasets(filterBy: {search: $q} first: $first orderBy: {created: descending}) {
    edges {
      node {
        id name created
        metadata { datasetUrl species modalities sampleSize studyDesign }
      }
    }
  }
}
"""


async def query_neuroimaging_datasets(
    brain_region: str,
    modality:     str = "fMRI",
    condition:    str = "",
    max_results:  int = 10,
) -> dict[str, Any]:
    """Search OpenNeuro + NeuroVault for neuroimaging datasets."""
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    _VALID = {"fMRI", "EEG", "MEG", "DTI", "MRI", "PET"}
    if modality not in _VALID:
        raise ValueError(f"modality must be one of {_VALID}, got '{modality}'")

    client   = await get_http_client()
    search_q = f"{brain_region} {condition} {modality}".strip()
    datasets: list[dict[str, Any]] = []

    async def _openneuro() -> None:
        try:
            r = await client.post(
                "https://openneuro.org/crn/graphql",
                json={"query": _OPENNEURO_GQL,
                      "variables": {"q": search_q, "first": max_results}},
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                return
            for edge in r.json().get("data", {}).get("datasets", {}).get("edges", []):
                node = edge.get("node", {})
                meta = node.get("metadata") or {}
                pid  = node.get("id", "")
                datasets.append({
                    "source": "OpenNeuro", "dataset_id": pid,
                    "title": node.get("name", ""),
                    "modalities": meta.get("modalities", [modality]),
                    "n_subjects": meta.get("sampleSize", "N/A"),
                    "url": meta.get("datasetUrl") or f"https://openneuro.org/datasets/{pid}",
                })
        except Exception as exc:
            logger.warning(f"[Neuroimaging] OpenNeuro failed: {exc}")

    async def _neurovault() -> None:
        try:
            r = await client.get(
                "https://neurovault.org/api/collections/",
                params={"format": "json", "search": search_q,
                        "limit": min(max_results, 10)},
            )
            if r.status_code != 200:
                return
            for col in r.json().get("results", []):
                datasets.append({
                    "source": "NeuroVault", "collection_id": col.get("id", ""),
                    "title": col.get("name", ""),
                    "n_subjects": col.get("number_of_subjects"),
                    "url": col.get("url", ""), "doi": col.get("doi", ""),
                })
        except Exception as exc:
            logger.warning(f"[Neuroimaging] NeuroVault failed: {exc}")

    await asyncio.gather(_openneuro(), _neurovault())

    return {
        "brain_region": brain_region,
        "modality":     modality,
        "condition":    condition,
        "total_found":  len(datasets),
        "datasets":     datasets[:max_results],
        "recommended_tools": {
            "preprocessing": ["fMRIPrep", "HCP Pipelines", "FSL"],
            "analysis":      ["nilearn", "MNE-Python", "AFNI"],
        },
    }
