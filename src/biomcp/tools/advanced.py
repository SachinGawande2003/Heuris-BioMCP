"""
BioMCP — Advanced Tools
========================
Tools:
  search_clinical_trials   — ClinicalTrials.gov v2 API
  get_trial_details        — Full protocol: arms, outcomes, eligibility
  get_gene_variants        — Ensembl REST: SNPs, indels, VEP consequences
  search_gene_expression   — NCBI GEO datasets
  search_scrna_datasets    — Human Cell Atlas single-cell RNA-seq
  multi_omics_gene_report  — 7-database parallel integration flagship tool
  query_neuroimaging       — OpenNeuro + NeuroVault datasets

APIs:
  ClinicalTrials.gov v2   https://clinicaltrials.gov/api/v2/
  Ensembl REST            https://rest.ensembl.org/
  NCBI GEO / E-utilities  https://eutils.ncbi.nlm.nih.gov/
  Human Cell Atlas        https://service.azul.data.humancellatlas.org/
  OpenNeuro               https://openneuro.org/crn/graphql
  NeuroVault              https://neurovault.org/api/
"""

from __future__ import annotations

import asyncio
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

CLINTRIALS_BASE = "https://clinicaltrials.gov/api/v2"
ENSEMBL_BASE    = "https://rest.ensembl.org"
GEO_BASE        = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HCA_BASE        = "https://service.azul.data.humancellatlas.org"


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov — search
# ─────────────────────────────────────────────────────────────────────────────

_VALID_STATUSES = frozenset({
    "RECRUITING", "COMPLETED", "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING", "TERMINATED", "ALL",
})
_VALID_PHASES = frozenset({"PHASE1", "PHASE2", "PHASE3", "PHASE4"})


@cached("clinical_trials")
@rate_limited("clinical_trials")
@with_retry(max_attempts=3)
async def search_clinical_trials(
    query: str,
    status: str = "RECRUITING",
    phase: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search ClinicalTrials.gov for clinical studies.

    Args:
        query:       Disease, drug, gene, or condition.
        status:      RECRUITING | COMPLETED | NOT_YET_RECRUITING |
                     ACTIVE_NOT_RECRUITING | ALL.
        phase:       PHASE1 | PHASE2 | PHASE3 | PHASE4 | None (all phases).
        max_results: Results (1–100). Default 10.

    Returns:
        { query, total_found, studies: [{ nct_id, title, status, phase,
          conditions, interventions, enrollment, dates, sponsor,
          summary, eligibility_snippet, locations }] }
    """
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}, got '{status}'")
    if phase and phase not in _VALID_PHASES:
        raise ValueError(f"phase must be one of {_VALID_PHASES}, got '{phase}'")

    client = await get_http_client()

    params: dict[str, Any] = {
        "query.term": query,
        "pageSize":   max_results,
        "format":     "json",
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

    resp = await client.get(
        f"{CLINTRIALS_BASE}/studies",
        params=params,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()

    studies: list[dict[str, Any]] = []
    for study in data.get("studies", []):
        proto      = study.get("protocolSection", {})
        id_mod     = proto.get("identificationModule",  {})
        status_mod = proto.get("statusModule",          {})
        desc_mod   = proto.get("descriptionModule",     {})
        design_mod = proto.get("designModule",          {})
        cond_mod   = proto.get("conditionsModule",      {})
        interv_mod = proto.get("armsInterventionsModule", {})
        elig_mod   = proto.get("eligibilityModule",     {})
        loc_mod    = proto.get("contactsLocationsModule", {})
        spon_mod   = proto.get("sponsorCollaboratorsModule", {})

        nct_id      = id_mod.get("nctId", "")
        locations   = list({
            f"{l.get('city','')}, {l.get('country','')}".strip(", ")
            for l in (loc_mod.get("locations") or [])[:8]
            if l.get("city") or l.get("country")
        })
        interventions = [
            {"name": i.get("interventionName",""), "type": i.get("interventionType","")}
            for i in (interv_mod.get("interventions") or [])[:6]
        ]

        studies.append({
            "nct_id":               nct_id,
            "title":                id_mod.get("briefTitle", ""),
            "status":               status_mod.get("overallStatus", ""),
            "phase":                design_mod.get("phases", []),
            "study_type":           design_mod.get("studyType", ""),
            "conditions":           cond_mod.get("conditions", []),
            "interventions":        interventions,
            "enrollment":           design_mod.get("enrollmentInfo", {}).get("count", ""),
            "start_date":           status_mod.get("startDateStruct",             {}).get("date",""),
            "completion_date":      status_mod.get("primaryCompletionDateStruct", {}).get("date",""),
            "summary":              (desc_mod.get("briefSummary") or "")[:600],
            "eligibility_snippet":  (elig_mod.get("eligibilityCriteria") or "")[:500],
            "sponsor":              spon_mod.get("leadSponsor", {}).get("name",""),
            "locations":            locations,
            "clinicaltrials_url":   f"https://clinicaltrials.gov/study/{nct_id}",
        })

    return {
        "query":        query,
        "status_filter":status,
        "phase_filter": phase,
        "total_found":  data.get("totalCount", len(studies)),
        "returned":     len(studies),
        "studies":      studies,
    }


@cached("clinical_trials")
@rate_limited("clinical_trials")
@with_retry(max_attempts=3)
async def get_trial_details(nct_id: str) -> dict[str, Any]:
    """
    Retrieve full protocol details for one clinical trial.

    Args:
        nct_id: NCT identifier (e.g. 'NCT04280705').

    Returns:
        Arms, primary/secondary outcomes, eligibility criteria, contacts.
    """
    nct_id = BioValidator.validate_nct_id(nct_id)
    client = await get_http_client()

    resp = await client.get(
        f"{CLINTRIALS_BASE}/studies/{nct_id}",
        params={"format": "json"},
        headers={"Accept": "application/json"},
    )
    if resp.status_code == 404:
        return {"error": f"Trial '{nct_id}' not found in ClinicalTrials.gov."}
    resp.raise_for_status()

    proto      = resp.json().get("protocolSection", {})
    out_mod    = proto.get("outcomesModule",            {})
    arms_mod   = proto.get("armsInterventionsModule",   {})
    elig_mod   = proto.get("eligibilityModule",         {})
    cont_mod   = proto.get("contactsLocationsModule",   {})

    return {
        "nct_id":  nct_id,
        "primary_outcomes": [
            {"measure": o.get("measure",""), "time_frame": o.get("timeFrame","")}
            for o in (out_mod.get("primaryOutcomes")   or [])
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure",""), "time_frame": o.get("timeFrame","")}
            for o in (out_mod.get("secondaryOutcomes") or [])[:6]
        ],
        "arms": [
            {
                "label":       a.get("armGroupLabel",  ""),
                "type":        a.get("armGroupType",   ""),
                "description": (a.get("description")  or "")[:300],
            }
            for a in (arms_mod.get("armGroups") or [])
        ],
        "eligibility": {
            "criteria":           (elig_mod.get("eligibilityCriteria") or "")[:1_500],
            "min_age":            elig_mod.get("minimumAge", ""),
            "max_age":            elig_mod.get("maximumAge", ""),
            "sex":                elig_mod.get("sex",        ""),
            "healthy_volunteers": elig_mod.get("healthyVolunteers", ""),
        },
        "central_contacts": [
            {
                "name":  c.get("centralContactName",  ""),
                "email": c.get("centralContactEMail", ""),
                "phone": c.get("centralContactPhone", ""),
            }
            for c in (cont_mod.get("centralContacts") or [])[:3]
        ],
        "clinicaltrials_url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensembl — gene variants
# ─────────────────────────────────────────────────────────────────────────────

@cached("ensembl")
@rate_limited("ensembl")
@with_retry(max_attempts=3)
async def get_gene_variants(
    gene_symbol: str,
    consequence_type: str = "missense_variant",
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Retrieve genetic variants in a gene from Ensembl.

    Args:
        gene_symbol:      HGNC gene symbol (e.g. 'BRCA2').
        consequence_type: VEP consequence type (or 'all' for unfiltered).
        max_results:      Variants to return (1–100).

    Returns:
        { gene, ensembl_gene_id, chromosome, total_variants,
          variants: [{ id, position, alleles, consequences, clinical_significance }] }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    max_results = BioValidator.clamp_int(max_results, 1, 100, "max_results")
    client      = await get_http_client()

    # Resolve to Ensembl gene ID
    xref_resp = await client.get(
        f"{ENSEMBL_BASE}/xrefs/symbol/homo_sapiens/{gene_symbol}",
        headers={"Accept": "application/json"},
    )
    xref_resp.raise_for_status()
    gene_ids = [e["id"] for e in xref_resp.json() if e.get("type") == "gene"]

    if not gene_ids:
        return {"gene": gene_symbol, "variants": [],
                "error": f"'{gene_symbol}' not found in Ensembl."}

    gene_id = gene_ids[0]

    # Get genomic coordinates
    lookup = await client.get(
        f"{ENSEMBL_BASE}/lookup/id/{gene_id}",
        headers={"Accept": "application/json"},
    )
    lookup.raise_for_status()
    info  = lookup.json()
    chrom = info.get("seq_region_name", "")
    start = info.get("start", 0)
    end   = min(info.get("end", 0), start + 200_000)   # cap window at 200 kb

    # Overlap variation endpoint
    var_resp = await client.get(
        f"{ENSEMBL_BASE}/overlap/region/human/{chrom}:{start}-{end}/variation",
        headers={"Accept": "application/json"},
        params={"feature": "variation"},
    )
    var_resp.raise_for_status()
    all_vars = var_resp.json()

    variants: list[dict[str, Any]] = []
    for v in all_vars[:max_results]:
        vid = v.get("id", "")
        variants.append({
            "id":                   vid,
            "chromosome":           chrom,
            "start":                v.get("start", ""),
            "end":                  v.get("end",   ""),
            "alleles":              v.get("alleles", []),
            "strand":               v.get("strand", ""),
            "consequence_types":    v.get("consequence_type", []),
            "clinical_significance":v.get("clinical_significance", []),
            "ensembl_url": (
                f"https://www.ensembl.org/Homo_sapiens/Variation/Summary?v={vid}"
                if vid.startswith("rs") else ""
            ),
        })

    return {
        "gene":             gene_symbol,
        "ensembl_gene_id":  gene_id,
        "chromosome":       chrom,
        "gene_start":       start,
        "gene_end":         info.get("end", 0),
        "total_variants":   len(all_vars),
        "returned":         len(variants),
        "variants":         variants,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NCBI GEO — gene expression datasets
# ─────────────────────────────────────────────────────────────────────────────

@cached("expression")
@rate_limited(_NCBI_SERVICE)
@with_retry(max_attempts=3)
async def search_gene_expression(
    gene_symbol: str,
    condition: str = "",
    max_datasets: int = 10,
) -> dict[str, Any]:
    """
    Search NCBI GEO for gene expression datasets.

    Args:
        gene_symbol:  HGNC gene symbol.
        condition:    Disease / tissue / cell-type filter (optional).
        max_datasets: Datasets to return (1–50).

    Returns:
        { gene, condition, total_found, datasets: [{ geo_accession, title,
          summary, organism, platform, n_samples, pubmed_ids, geo_url }] }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
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
        return {"gene": gene_symbol, "condition": condition,
                "total_found": 0, "datasets": []}

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
            "geo_accession":   acc,
            "title":           d.get("title",       ""),
            "summary":         (d.get("summary")    or "")[:400],
            "organism":        d.get("organism",    ""),
            "platform":        d.get("gpl",         ""),
            "n_samples":       d.get("n_samples",   0),
            "entry_type":      d.get("entrytype",   ""),
            "pubmed_ids":      d.get("pubmedids",   []),
            "submission_date": d.get("pdat",        ""),
            "geo_url":         f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
        })

    return {
        "gene":        gene_symbol,
        "condition":   condition,
        "total_found": total,
        "returned":    len(datasets),
        "datasets":    datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human Cell Atlas — single-cell RNA-seq datasets
# ─────────────────────────────────────────────────────────────────────────────

@cached("expression")
@rate_limited("hca")
@with_retry(max_attempts=2)
async def search_scrna_datasets(
    tissue: str,
    species: str = "Homo sapiens",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search Human Cell Atlas for single-cell RNA-seq datasets.

    Args:
        tissue:      Tissue or organ (e.g. 'brain', 'lung', 'heart', 'liver').
        species:     'Homo sapiens' | 'Mus musculus'. Default Homo sapiens.
        max_results: Datasets (1–50).

    Returns:
        { tissue, species, total_found, datasets: [{ project_id, title,
          cell_count, donor_count, sequencing_technologies, hca_url }] }
    """
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
            "tissue":  tissue,
            "species": species,
            "message": (
                "Human Cell Atlas may be temporarily unavailable. "
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
            "title":                   proj.get("projectTitle",    ""),
            "short_name":              proj.get("projectShortname",""),
            "cell_count":              cells.get("totalCells",     0),
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
# Multi-Omics Gene Report — FLAGSHIP TOOL
# ─────────────────────────────────────────────────────────────────────────────

async def multi_omics_gene_report(gene_symbol: str) -> dict[str, Any]:
    """
    Generate a comprehensive multi-omics report for a gene.

    Queries 7 databases SIMULTANEOUSLY (asyncio.gather):
      1. NCBI Gene     — genomic info
      2. PubMed        — recent literature
      3. Reactome      — pathway associations
      4. ChEMBL        — drug targets
      5. Open Targets  — disease associations
      6. NCBI GEO      — expression datasets
      7. ClinicalTrials— active trials

    Args:
        gene_symbol: HGNC gene symbol (e.g. 'EGFR', 'TP53', 'BRCA1').

    Returns:
        Integrated report across all 7 data layers with provenance metadata.
    """
    from biomcp.tools.ncbi import get_gene_info, search_pubmed
    from biomcp.tools.pathways import (
        get_reactome_pathways,
        get_drug_targets,
        get_gene_disease_associations,
    )

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    logger.info(f"[Multi-Omics] Generating report for {gene_symbol}")

    # Fire all 7 queries in parallel — fastest possible aggregation
    results = await asyncio.gather(
        get_gene_info(gene_symbol),
        search_pubmed(f"{gene_symbol}[Gene] function mechanism review", max_results=5),
        get_reactome_pathways(gene_symbol),
        get_drug_targets(gene_symbol, max_results=10),
        get_gene_disease_associations(gene_symbol, max_results=10),
        search_gene_expression(gene_symbol, max_datasets=5),
        search_clinical_trials(gene_symbol, max_results=5),
        return_exceptions=True,
    )

    labels = ["genomics", "literature", "reactome", "drug_targets",
              "disease_associations", "expression", "clinical_trials"]

    layers: dict[str, Any] = {}
    for label, res in zip(labels, results):
        if isinstance(res, Exception):
            logger.warning(f"[Multi-Omics] {label} failed: {res}")
            layers[label] = {"error": str(res), "status": "failed"}
        else:
            layers[label] = res

    # Summarise literature layer
    if "articles" in layers.get("literature", {}):
        lit = layers.pop("literature")
        layers["literature"] = {
            "total_publications": lit.get("total_found", 0),
            "recent_papers": [
                {"pmid": a["pmid"], "title": a["title"],
                 "year": a["year"],  "journal": a["journal"]}
                for a in lit.get("articles", [])
            ],
        }

    return {
        "gene":          gene_symbol,
        "report_type":   "multi_omics_integrated",
        "layers":        layers,
        "data_sources": [
            "NCBI Gene", "PubMed", "Reactome", "ChEMBL",
            "Open Targets", "NCBI GEO", "ClinicalTrials.gov",
        ],
        "note": (
            "Queries run in parallel — total latency ≈ slowest single query. "
            "Layers with 'status: failed' encountered a transient API error; "
            "retry individually for full data."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Neuroimaging datasets — OpenNeuro + NeuroVault
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
    modality: str = "fMRI",
    condition: str = "",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search for neuroimaging datasets across open repositories.

    Queries OpenNeuro (GraphQL) and NeuroVault (REST) in parallel.

    Args:
        brain_region: Brain region (e.g. 'hippocampus', 'prefrontal cortex').
        modality:     fMRI | EEG | MEG | DTI | MRI | PET. Default fMRI.
        condition:    Neurological condition (e.g. 'Alzheimer', 'depression').
        max_results:  Datasets (1–50).

    Returns:
        { brain_region, modality, condition, datasets: [...],
          recommended_tools: { ... } }
    """
    max_results = BioValidator.clamp_int(max_results, 1, 50, "max_results")
    _VALID_MODALITIES = {"fMRI","EEG","MEG","DTI","MRI","PET"}
    if modality not in _VALID_MODALITIES:
        raise ValueError(f"modality must be one of {_VALID_MODALITIES}, got '{modality}'")

    client    = await get_http_client()
    search_q  = f"{brain_region} {condition} {modality}".strip()
    datasets: list[dict[str, Any]] = []

    # ── OpenNeuro (GraphQL) ───────────────────────────────────────────────────
    async def _openneuro() -> None:
        try:
            r = await client.post(
                "https://openneuro.org/crn/graphql",
                json={"query": _OPENNEURO_GQL,
                      "variables": {"q": search_q, "first": max_results}},
                headers={"Content-Type": "application/json",
                         "Accept":       "application/json"},
            )
            if r.status_code != 200:
                return
            for edge in r.json().get("data", {}).get("datasets", {}).get("edges", []):
                node = edge.get("node", {})
                meta = node.get("metadata") or {}
                datasets.append({
                    "source":         "OpenNeuro",
                    "dataset_id":     node.get("id",    ""),
                    "title":          node.get("name",  ""),
                    "modalities":     meta.get("modalities", [modality]),
                    "species":        meta.get("species",    "Human"),
                    "n_subjects":     meta.get("sampleSize", "N/A"),
                    "study_design":   meta.get("studyDesign",""),
                    "deposited":      node.get("created",""),
                    "url": (
                        meta.get("datasetUrl")
                        or f"https://openneuro.org/datasets/{node.get('id','')}"
                    ),
                })
        except Exception as exc:
            logger.warning(f"[Neuroimaging] OpenNeuro failed: {exc}")

    # ── NeuroVault (REST) ─────────────────────────────────────────────────────
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
                    "source":        "NeuroVault",
                    "collection_id": col.get("id",           ""),
                    "title":         col.get("name",         ""),
                    "description":   (col.get("description") or "")[:300],
                    "n_subjects":    col.get("number_of_subjects"),
                    "scanner":       col.get("scanner_make", ""),
                    "url":           col.get("url",          ""),
                    "doi":           col.get("doi",          ""),
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
            "preprocessing":  ["fMRIPrep", "HCP Pipelines", "SPM12", "FSL"],
            "analysis":       ["nilearn", "MNE-Python", "AFNI", "FreeSurfer"],
            "visualization":  ["Niivue", "FSLeyes", "BrainPainter", "mrview"],
            "connectivity":   ["CONN Toolbox", "The Virtual Brain", "GraphVar"],
        },
    }
