"""
Heuris-BioMCP strategic MCP server.

This server exposes the curated public MCP surface while retaining lower-level
legacy handlers internally for planner and composition workflows.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from biomcp import __version__
from biomcp.utils import close_http_client, format_error, format_success

# FIX #1: Removed duplicate import line that was here

SERVER_NAME = "heuris-biomcp"
SERVER_DISPLAY_NAME = "Heuris-BioMCP"
STREAMABLE_HTTP_PATH = "/mcp"
SSE_PATH = "/sse"
MESSAGE_PATH = "/messages/"
DEFAULT_SERVER_WEBSITE_URL = "https://heuris-biomcp.onrender.com"
LOGO_ROUTE_PATH = "/logo.jpeg"

_DISPATCH_TABLE: dict[str, Callable[..., Any]] | None = None
_TOOL_MODULES: dict[str, Any] | None = None

CAPABILITY_CONFIG: dict[str, dict[str, Any]] = {
    "core_server": {
        "env_any": [],
        "healthy_detail": "Core MCP server, tool registry, and shared HTTP infrastructure are available.",
        "degraded_detail": "Core MCP server is unavailable.",
        "tools": [],
        "required_env": [],
    },
    "ncbi_enhanced": {
        "env_any": ["NCBI_API_KEY"],
        "healthy_detail": "NCBI API key configured for elevated rate limits.",
        "degraded_detail": "NCBI tools work without a key, but rate limits are reduced to 3 requests/second.",
        "tools": [
            "search_pubmed",
            "get_gene_info",
            "run_blast",
            "search_gene_expression",
            "get_omim_gene_diseases",
        ],
        "required_env": ["NCBI_API_KEY"],
    },
    "nvidia_boltz2": {
        "env_any": ["NVIDIA_BOLTZ2_API_KEY", "NVIDIA_NIM_API_KEY"],
        "healthy_detail": "Boltz-2 credentials configured.",
        "degraded_detail": (
            "Boltz-2 tools are unavailable until NVIDIA_BOLTZ2_API_KEY or NVIDIA_NIM_API_KEY is set."
        ),
        "tools": ["predict_structure_boltz2"],
        "required_env": ["NVIDIA_BOLTZ2_API_KEY", "NVIDIA_NIM_API_KEY"],
    },
    "nvidia_evo2": {
        "env_any": ["NVIDIA_EVO2_API_KEY", "NVIDIA_NIM_API_KEY"],
        "healthy_detail": "Evo2 credentials configured.",
        "degraded_detail": (
            "Evo2 tools are unavailable until NVIDIA_EVO2_API_KEY or NVIDIA_NIM_API_KEY is set."
        ),
        "tools": ["generate_dna_evo2"],
        "required_env": ["NVIDIA_EVO2_API_KEY", "NVIDIA_NIM_API_KEY"],
    },
}


def _server_website_url() -> str:
    return os.getenv("BIOMCP_WEBSITE_URL", DEFAULT_SERVER_WEBSITE_URL)


def _server_icon_url() -> str:
    return os.getenv("BIOMCP_ICON_URL", f"{_server_website_url().rstrip('/')}{LOGO_ROUTE_PATH}")


def _resolve_logo_path() -> str | None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(root_dir, "LOGO.jpeg"),
        os.path.join(os.path.dirname(__file__), "LOGO.jpeg"),
        os.path.abspath("LOGO.jpeg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schema Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tool(name: str, description: str, properties: dict, required: list[str]) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _int_prop(desc: str, default: int, min_: int, max_: int) -> dict:
    return {
        "type": "integer",
        "description": f"{desc} Default {default}.",
        "default": default,
        "minimum": min_,
        "maximum": max_,
    }


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _bool_prop(desc: str, default: bool) -> dict:
    return {"type": "boolean", "description": desc, "default": default}


def _float_prop(desc: str, default: float) -> dict:
    return {"type": "number", "description": desc, "default": default}


def _enum_prop(desc: str, values: list[str], default: str | None = None) -> dict:
    p: dict[str, Any] = {"type": "string", "description": desc, "enum": values}
    if default is not None:
        p["default"] = default
    return p


def _array_prop(desc: str, item_type: str = "string") -> dict:
    return {"type": "array", "description": desc, "items": {"type": item_type}}


def _obj_prop(desc: str) -> dict:
    return {
        "type": "object",
        "description": desc,
        "additionalProperties": {"type": "number"},
    }


def _tool_names() -> list[str]:
    return [tool.name for tool in TOOLS]


def _build_capability_status() -> dict[str, dict[str, Any]]:
    tool_names = set(_tool_names())
    capabilities: dict[str, dict[str, Any]] = {}
    for name, config in CAPABILITY_CONFIG.items():
        env_any = config["env_any"]
        is_healthy = True if not env_any else any(os.getenv(var) for var in env_any)
        capabilities[name] = {
            "status": "healthy" if is_healthy else "degraded",
            "detail": config["healthy_detail"] if is_healthy else config["degraded_detail"],
            "required_env": config["required_env"],
            "tools": [tool for tool in config["tools"] if tool in tool_names],
        }
    return capabilities


def _build_health_report(transport_mode: str | None = None) -> dict[str, Any]:
    mode = transport_mode or os.getenv("BIOMCP_TRANSPORT", "stdio")
    capabilities = _build_capability_status()
    degraded = [name for name, cfg in capabilities.items() if cfg["status"] != "healthy"]
    return {
        "service": SERVER_NAME,
        "display_name": SERVER_DISPLAY_NAME,
        "version": __version__,
        "status": "degraded" if degraded else "healthy",
        "tool_count": len(TOOLS),
        "transport": {
            "mode": mode,
            "streamable_http_path": STREAMABLE_HTTP_PATH if mode == "http" else None,
            "sse_path": SSE_PATH if mode == "http" else None,
            "message_path": MESSAGE_PATH if mode == "http" else None,
        },
        "capabilities": capabilities,
        "degraded_capabilities": degraded,
    }


def _build_readiness_report(transport_mode: str | None = None) -> dict[str, Any]:
    mode = transport_mode or os.getenv("BIOMCP_TRANSPORT", "stdio")
    return {
        "service": SERVER_NAME,
        "version": __version__,
        "ready": bool(TOOLS),
        "transport_mode": mode,
        "tool_count": len(TOOLS),
    }


def _build_root_report(transport_mode: str | None = None) -> dict[str, Any]:
    mode = transport_mode or os.getenv("BIOMCP_TRANSPORT", "stdio")
    base_url = _server_website_url().rstrip("/")
    return {
        "service": SERVER_NAME,
        "display_name": SERVER_DISPLAY_NAME,
        "version": __version__,
        "status": "ok",
        "transport_mode": mode,
        "recommended_remote_url": f"{base_url}{STREAMABLE_HTTP_PATH}" if mode == "http" else None,
        "legacy_sse_url": f"{base_url}{SSE_PATH}" if mode == "http" else None,
        "health_url": f"{base_url}/healthz" if mode == "http" else None,
        "ready_url": f"{base_url}/readyz" if mode == "http" else None,
        "tool_health_url": f"{base_url}/tool-health" if mode == "http" else None,
    }


def _build_tool_health_report() -> dict[str, Any]:
    tool_names = _tool_names()
    capabilities = _build_capability_status()
    gated_capabilities = {
        name: cfg
        for name, cfg in capabilities.items()
        if cfg["tools"] and cfg["status"] != "healthy"
    }
    return {
        "service": SERVER_NAME,
        "version": __version__,
        "registered_tools": tool_names,
        "registered_tool_count": len(tool_names),
        "gated_capabilities": gated_capabilities,
        "ungated_tool_count": len(tool_names)
        - sum(len(cfg["tools"]) for cfg in gated_capabilities.values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Complete Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list[Tool] = [
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 1: Literature & NCBI (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "search_pubmed",
        "Search PubMed for scientific literature. Supports full NCBI query syntax "
        "(MeSH terms, Boolean operators, field tags, date ranges). Returns articles "
        "with title, authors, abstract, DOI, PMID, journal, year, and MeSH terms. "
        "Results auto-indexed into session knowledge graph.",
        {
            "query": _str_prop("PubMed query. E.g. 'BRCA1[Gene] AND breast cancer AND Review[pt]'"),
            "max_results": _int_prop("Articles to return", 10, 1, 200),
            "sort": _enum_prop("Sort order.", ["relevance", "pub_date"], "relevance"),
        },
        ["query"],
    ),
    _tool(
        "get_gene_info",
        "Retrieve gene information from NCBI Gene — symbol, full name, chromosomal "
        "location, aliases, RefSeq IDs, and functional summary. "
        "Auto-indexes gene entity into session knowledge graph.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'BRCA1', 'EGFR')."),
            "organism": _str_prop("Species. Default: 'homo sapiens'."),
        },
        ["gene_symbol"],
    ),
    _tool(
        "run_blast",
        "Run NCBI BLAST sequence alignment (blastp/blastn/blastx/tblastn). "
        "Async polling — waits up to 120s for results.",
        {
            "sequence": _str_prop("Amino acid or nucleotide sequence (raw or FASTA)."),
            "program": _enum_prop(
                "BLAST program.", ["blastp", "blastn", "blastx", "tblastn"], "blastp"
            ),
            "database": _enum_prop(
                "Target database.", ["nr", "nt", "swissprot", "pdb", "refseq_protein"], "nr"
            ),
            "max_hits": _int_prop("Alignments to return", 10, 1, 100),
        },
        ["sequence"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 2: Proteins & Structures (4 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_protein_info",
        "Full UniProt Swiss-Prot entry: function, domains, PTMs, GO terms, disease "
        "links, sequence. Prefer reviewed accessions (P/Q/O prefix). "
        "Auto-indexes protein + disease edges into session knowledge graph.",
        {"accession": _str_prop("UniProt accession (e.g. 'P04637' for human TP53).")},
        ["accession"],
    ),
    _tool(
        "search_proteins",
        "Search UniProt for proteins matching a query with species/review filter.",
        {
            "query": _str_prop("Gene name, function, disease, etc."),
            "organism": _str_prop("Species filter. Default: 'homo sapiens'."),
            "max_results": _int_prop("Results", 10, 1, 100),
            "reviewed_only": _bool_prop("Swiss-Prot only.", True),
        },
        ["query"],
    ),
    _tool(
        "get_alphafold_structure",
        "AlphaFold DB predicted structure: per-residue pLDDT confidence stats, "
        "PDB/mmCIF download URLs. pLDDT ≥90=very high, 70–90=confident, <50=disordered.",
        {
            "uniprot_accession": _str_prop("UniProt accession (e.g. 'P04637')."),
            "model_version": _str_prop("AlphaFold model version. Default: v4."),
        },
        ["uniprot_accession"],
    ),
    _tool(
        "search_pdb_structures",
        "Search RCSB PDB for experimental protein structures with method, resolution, "
        "deposition date, and download links.",
        {
            "query": _str_prop("Protein name, gene, organism, or keywords."),
            "max_results": _int_prop("Results", 10, 1, 50),
        },
        ["query"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 3: Pathways (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "search_pathways",
        "Search KEGG for biological pathways with viewer URLs and diagram images.",
        {
            "query": _str_prop("Keyword — pathway name, gene, disease."),
            "organism": _str_prop("KEGG organism code. Default: 'hsa' (human)."),
        },
        ["query"],
    ),
    _tool(
        "get_pathway_genes",
        "List all genes in a KEGG pathway with IDs and descriptions.",
        {"pathway_id": _str_prop("KEGG pathway ID (e.g. 'hsa05200').")},
        ["pathway_id"],
    ),
    _tool(
        "get_reactome_pathways",
        "Get Reactome pathways for a gene with hierarchy and diagram URLs. "
        "Auto-indexes gene→pathway edges into session knowledge graph.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "species": _str_prop("NCBI taxonomy ID. Default: '9606' (Homo sapiens)."),
        },
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 4: Drug Discovery (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_drug_targets",
        "ChEMBL drug-target activities: IC50, Ki, Kd values, assay types, "
        "approval status. Auto-indexes drug→gene edges into knowledge graph.",
        {
            "gene_symbol": _str_prop("Target gene symbol (e.g. 'EGFR', 'BRAF', 'KRAS')."),
            "max_results": _int_prop("Drug entries", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_compound_info",
        "ChEMBL compound details: SMILES, ADMET properties, Lipinski Ro5, QED score, "
        "clinical phase, therapeutic indications.",
        {"chembl_id": _str_prop("ChEMBL compound ID (e.g. 'CHEMBL25' for aspirin).")},
        ["chembl_id"],
    ),
    _tool(
        "get_gene_disease_associations",
        "Open Targets gene-disease evidence across 6 datatypes: genetic_association, "
        "somatic_mutation, known_drug, animal_model, affected_pathway, literature. "
        "Auto-indexes gene→disease edges into knowledge graph.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "max_results": _int_prop("Associations", 15, 1, 50),
        },
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 5: Genomics & Expression (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_gene_variants",
        "Ensembl variants: SNPs, indels, VEP consequence types, clinical significance. "
        "Auto-indexes gene→variant edges into knowledge graph.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "consequence_type": _str_prop("VEP consequence filter. Default: 'missense_variant'."),
            "max_results": _int_prop("Variants", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_gene_expression",
        "NCBI GEO expression datasets with organism, platform, sample counts, and PubMed refs.",
        {
            "gene_symbol": _str_prop("Gene symbol to search for."),
            "condition": _str_prop("Disease/tissue filter (optional)."),
            "max_datasets": _int_prop("Datasets", 10, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_scrna_datasets",
        "Human Cell Atlas single-cell RNA-seq datasets by tissue with cell counts and tech.",
        {
            "tissue": _str_prop("Tissue/organ (e.g. 'brain', 'lung', 'liver')."),
            "species": _str_prop("Species. Default: 'Homo sapiens'."),
            "max_results": _int_prop("Datasets", 10, 1, 50),
        },
        ["tissue"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 6: Clinical (2 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "search_clinical_trials",
        "ClinicalTrials.gov v2: trial status, phase, interventions, enrollment, "
        "eligibility. Auto-indexes drug→disease treatment edges from trials.",
        {
            "query": _str_prop("Disease, drug, gene, or condition."),
            "status": _enum_prop(
                "Trial status.",
                ["RECRUITING", "COMPLETED", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ALL"],
                "RECRUITING",
            ),
            "phase": _enum_prop(
                "Phase filter (optional).", ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
            ),
            "max_results": _int_prop("Results", 10, 1, 100),
        },
        ["query"],
    ),
    _tool(
        "get_trial_details",
        "Full protocol for one trial: arms, primary/secondary outcomes, eligibility, contacts.",
        {"nct_id": _str_prop("NCT identifier (e.g. 'NCT04280705').")},
        ["nct_id"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 7: AI-Powered — NVIDIA NIM (4 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "predict_structure_boltz2",
        "MIT Boltz-2 via NVIDIA NIM: protein/DNA/RNA/ligand structure prediction + "
        "binding affinity (FEP accuracy, 1000x faster). Requires NVIDIA_BOLTZ2_API_KEY.",
        {
            "protein_sequences": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Protein AA sequences.",
            },
            "ligand_smiles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ligand SMILES strings.",
            },
            "predict_affinity": _bool_prop("Compute binding affinity.", False),
            "recycling_steps": _int_prop("Recycling steps", 3, 1, 10),
            "sampling_steps": _int_prop("Diffusion steps", 200, 50, 500),
        },
        ["protein_sequences"],
    ),
    _tool(
        "generate_dna_evo2",
        "Arc Evo2-40B via NVIDIA NIM: Generate DNA sequences with 40B parameter genomic "
        "foundation model. Requires NVIDIA_EVO2_API_KEY.",
        {
            "sequence": _str_prop("Seed DNA sequence (ACGT). Evo2 continues from this."),
            "num_tokens": _int_prop("New DNA bases to generate", 200, 1, 1200),
            "temperature": _float_prop("0.0=deterministic, 1.0=diverse.", 1.0),
        },
        ["sequence"],
    ),
    _tool(
        "score_sequence_evo2",
        "Evo2-40B variant effect prediction: compare wildtype vs variant DNA log-likelihoods.",
        {
            "wildtype_sequence": _str_prop("Reference wildtype DNA sequence."),
            "variant_sequence": _str_prop("Mutant DNA sequence (same length)."),
        },
        ["wildtype_sequence", "variant_sequence"],
    ),
    _tool(
        "design_protein_ligand",
        "Full drug-discovery pipeline: UniProt fetch → Boltz-2 structure + affinity in one call.",
        {
            "uniprot_accession": _str_prop("Target protein UniProt ID."),
            "ligand_smiles": _str_prop("Drug SMILES string."),
            "predict_affinity": _bool_prop("Compute binding affinity. Default True.", True),
        },
        ["uniprot_accession", "ligand_smiles"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 8: Integrated & Advanced (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "multi_omics_gene_report",
        "FLAGSHIP: 7-database parallel integration — NCBI Gene, PubMed, Reactome, "
        "ChEMBL, Open Targets, GEO, ClinicalTrials.gov. One call, complete overview.",
        {"gene_symbol": _str_prop("HGNC gene symbol (e.g. 'EGFR', 'TP53', 'BRCA1', 'KRAS').")},
        ["gene_symbol"],
    ),
    _tool(
        "query_neuroimaging_datasets",
        "OpenNeuro + NeuroVault neuroimaging datasets with acquisition metadata.",
        {
            "brain_region": _str_prop("Brain region (e.g. 'hippocampus', 'prefrontal cortex')."),
            "modality": _enum_prop(
                "Imaging modality.", ["fMRI", "EEG", "MEG", "DTI", "MRI", "PET"], "fMRI"
            ),
            "condition": _str_prop("Neurological condition filter."),
            "max_results": _int_prop("Datasets", 10, 1, 50),
        },
        ["brain_region"],
    ),
    _tool(
        "generate_research_hypothesis",
        "Literature mining → data-driven testable hypotheses with supporting evidence.",
        {
            "topic": _str_prop("Research topic."),
            "max_hypotheses": _int_prop("Hypotheses to generate", 3, 1, 10),
        },
        ["topic"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 9: Extended Databases (7 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_omim_gene_diseases",
        "OMIM genetic disease-gene relationships with inheritance patterns.",
        {"gene_symbol": _str_prop("HGNC gene symbol.")},
        ["gene_symbol"],
    ),
    _tool(
        "get_string_interactions",
        "STRING protein-protein interaction network with confidence scores.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "min_score": _int_prop("Minimum score (400=medium, 700=high)", 400, 0, 1000),
            "max_results": _int_prop("Interaction partners", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_gtex_expression",
        "GTEx tissue-specific gene expression in healthy humans across 54 tissues.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "top_tissues": _int_prop("Top tissues by median TPM", 10, 1, 54),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_cbio_mutations",
        "cBioPortal cancer mutation frequencies across TCGA cohorts.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "cancer_type": _str_prop("TCGA cancer type (e.g. 'luad'). Empty=pan-cancer."),
            "max_studies": _int_prop("Studies to query", 10, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_gwas_catalog",
        "NHGRI-EBI GWAS Catalog: genome-wide significant associations for a gene.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "max_results": _int_prop("Associations", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_disgenet_associations",
        "DisGeNET comprehensive gene-disease associations with GDA scores.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "min_score": _float_prop("Minimum GDA score (0–1). Default 0.1.", 0.1),
            "max_results": _int_prop("Associations", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_pharmgkb_variants",
        "PharmGKB pharmacogenomics: genetic variants affecting drug response.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'CYP2D6', 'TPMT')."),
            "max_results": _int_prop("Annotations", 15, 1, 50),
        },
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 10: Verification & Conflict Detection (2 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "verify_biological_claim",
        "Verify a biological claim against 3–5 databases with graded evidence.",
        {
            "claim": _str_prop("Natural language biological claim to verify."),
            "context_gene": _str_prop("Optional gene symbol to focus evidence gathering."),
        },
        ["claim"],
    ),
    _tool(
        "detect_database_conflicts",
        "Scan for conflicting biological information about a gene across databases.",
        {"gene_symbol": _str_prop("HGNC gene symbol to scan for cross-database conflicts.")},
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 11: Experimental Design (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "generate_experimental_protocol",
        "Generate a complete experimental protocol from a biological hypothesis.",
        {
            "hypothesis": _str_prop("Research hypothesis."),
            "gene_symbol": _str_prop("Primary gene of interest."),
            "cancer_type": _str_prop("Cancer type."),
            "assay_type": _enum_prop(
                "Assay type.",
                ["auto", "crispr_knockout", "sirna_knockdown", "drug_sensitivity",
                 "apoptosis_flow", "protein_interaction"],
                "auto",
            ),
        },
        ["hypothesis"],
    ),
    _tool(
        "suggest_cell_lines",
        "Recommend validated cell lines for a research context.",
        {
            "cancer_type": _str_prop("Cancer type."),
            "gene_symbol": _str_prop("Gene of interest for mutation-aware filtering."),
            "molecular_feature": _str_prop("Required molecular feature."),
            "max_results": _int_prop("Cell lines to return", 5, 1, 15),
        },
        ["cancer_type"],
    ),
    _tool(
        "estimate_statistical_power",
        "Calculate required sample size for adequate statistical power.",
        {
            "expected_effect_size": _float_prop("Cohen's d (0.2=small, 0.5=medium, 0.8=large).", 0.5),
            "alpha": _float_prop("Significance threshold. Default 0.05.", 0.05),
            "power": _float_prop("Desired power. Default 0.8.", 0.8),
            "n_groups": _int_prop("Number of comparison groups", 2, 2, 10),
        },
        [],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 12: Session Intelligence (5 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "resolve_entity",
        "Resolve any biological identifier to canonical cross-database form.",
        {
            "query": _str_prop("Any biological identifier."),
            "hint_type": _enum_prop("Entity type hint.", ["gene", "protein", "drug", "disease"], "gene"),
        },
        ["query"],
    ),
    _tool(
        "get_session_knowledge_graph",
        "Return the live Session Knowledge Graph built from all tool calls this session.",
        {},
        [],
    ),
    _tool(
        "find_biological_connections",
        "Discover multi-hop connections between biological entities in the session graph.",
        {"min_path_length": _int_prop("Minimum path hops", 2, 2, 4)},
        [],
    ),
    _tool(
        "export_research_session",
        "Export full research session with provenance, BibTeX citations, and reproducibility script.",
        {},
        [],
    ),
    _tool(
        "plan_and_execute_research",
        "Build and execute an optimized DAG-based research plan from a natural language goal.",
        {
            "goal": _str_prop("Natural language research objective."),
            "depth": _enum_prop("Research depth.", ["quick", "standard", "deep"], "standard"),
            "gene": _str_prop("Primary gene symbol (auto-extracted if not provided)."),
        },
        ["goal"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 13: Intelligence Layer (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "validate_reasoning_chain",
        "Verify a multi-step biological reasoning chain against primary databases.",
        {
            "reasoning_chain": _str_prop("Arrow notation: 'KRAS → RAF → MEK → ERK → proliferation'"),
            "verify_depth": _enum_prop("Verification depth.", ["quick", "standard", "deep"], "standard"),
        },
        ["reasoning_chain"],
    ),
    _tool(
        "find_repurposing_candidates",
        "Drug repurposing engine: surface approved drugs with activity against a target/disease.",
        {
            "disease": _str_prop("Target disease."),
            "gene_target": _str_prop("Primary gene target. Optional."),
            "max_candidates": _int_prop("Maximum repurposing candidates", 15, 1, 50),
        },
        ["disease"],
    ),
    _tool(
        "find_research_gaps",
        "Map what IS and ISN'T known for a topic; surface high-impact unanswered questions.",
        {
            "topic": _str_prop("Research topic."),
            "max_gaps": _int_prop("Maximum gaps to report", 10, 1, 25),
        },
        ["topic"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 14: Tier 2 Extended Databases (7 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_biogrid_interactions",
        "BioGRID 2M+ manually curated protein-protein interactions from primary literature.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "interaction_type": _enum_prop("Interaction type.", ["physical", "genetic", "all"], "physical"),
            "max_results": _int_prop("Interactions to return", 25, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_orphan_diseases",
        "Orphanet 6,000+ rare diseases with gene associations and prevalence.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol to find associated rare diseases."),
            "disease_name": _str_prop("Disease name or keyword."),
        },
        [],
    ),
    _tool(
        "get_tcga_expression",
        "TCGA tumor RNA-seq from actual patient samples via GDC API.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "cancer_type": _str_prop("TCGA project code (e.g. 'TCGA-LUAD'). Empty=pan-cancer."),
            "max_cases": _int_prop("Cases to sample", 10, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_cellmarker",
        "CellMarker 2.0 validated cell type markers for scRNA-seq annotation.",
        {
            "gene_symbol": _str_prop("Gene to find which cell types it marks."),
            "tissue": _str_prop("Tissue filter."),
            "cell_type": _str_prop("Cell type filter."),
        },
        [],
    ),
    _tool(
        "get_encode_regulatory",
        "ENCODE regulatory elements: promoters, enhancers, CTCF, TF binding.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "element_type": _enum_prop(
                "Regulatory element type.",
                ["all", "promoter", "enhancer", "CTCF", "TF_binding", "open_chromatin"], "all",
            ),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_metabolomics",
        "MetaboLights metabolomics studies connecting metabolites to genes and diseases.",
        {
            "gene_symbol": _str_prop("Gene to find related metabolic studies."),
            "metabolite": _str_prop("Metabolite name (e.g. 'glucose', 'lactate')."),
            "disease": _str_prop("Disease context."),
        },
        [],
    ),
    _tool(
        "get_ucsc_splice_variants",
        "UCSC Genome Browser alternative splicing isoforms and UTR annotations.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "genome": _enum_prop("Reference genome.", ["hg38", "hg19"], "hg38"),
        },
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 15: CRISPR Design Suite (5 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "design_crispr_guides",
        "Design CRISPR sgRNA guides with Doench 2016 efficiency scoring.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "cas_variant": _enum_prop("Cas nuclease.", ["SpCas9", "SaCas9", "Cas12a", "CjCas9"], "SpCas9"),
            "n_guides": _int_prop("Top guides to return", 5, 1, 20),
        },
        ["gene_symbol"],
    ),
    _tool(
        "score_guide_efficiency",
        "Score an sgRNA using Doench 2016 RS2-inspired multi-feature model.",
        {
            "guide_sequence": _str_prop("17–24nt guide RNA sequence."),
            "cas_variant": _enum_prop("Cas variant.", ["SpCas9", "SaCas9", "Cas12a"], "SpCas9"),
        },
        ["guide_sequence"],
    ),
    _tool(
        "predict_off_target_sites",
        "Predict CRISPR off-target risk using seed-region analysis and optional BLAST.",
        {
            "guide_sequence": _str_prop("20nt sgRNA sequence."),
            "use_blast": _bool_prop("Submit seed to NCBI BLAST (~30s extra).", True),
        },
        ["guide_sequence"],
    ),
    _tool(
        "design_base_editor_guides",
        "Design guides for CBE/ABE base editing to introduce specific mutations.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "target_mutation": _str_prop("Target mutation (e.g. 'G12D', 'c.524G>A')."),
            "editor_type": _enum_prop("Base editor type.", ["CBE", "ABE", "auto"], "auto"),
        },
        ["gene_symbol", "target_mutation"],
    ),
    _tool(
        "get_crispr_repair_outcomes",
        "Predict CRISPR-Cas9 NHEJ/HDR repair outcome distribution.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "guide_sequence": _str_prop("20nt sgRNA sequence."),
            "cell_line": _enum_prop("Cell line.", ["generic", "HEK293", "HeLa", "primary"], "generic"),
        },
        ["gene_symbol", "guide_sequence"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 16: FDA Drug Safety Intelligence (4 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "query_adverse_events",
        "Query FDA FAERS adverse event reports for drug safety signals.",
        {
            "drug_name": _str_prop("Drug name (generic or brand)."),
            "event_type": _enum_prop(
                "Event category.",
                ["all", "cardiac", "hepatic", "hematologic", "neurological",
                 "renal", "hypersensitivity", "respiratory", "oncology"], "all",
            ),
            "serious_only": _bool_prop("Only serious adverse events.", False),
        },
        ["drug_name"],
    ),
    _tool(
        "analyze_safety_signals",
        "Pharmacovigilance disproportionality analysis: PRR, ROR, IC on FAERS.",
        {
            "drug_name": _str_prop("Drug of interest."),
            "event_terms": {
                "type": "array", "items": {"type": "string"},
                "description": "MedDRA event terms to analyze.",
            },
        },
        ["drug_name"],
    ),
    _tool(
        "get_drug_label_warnings",
        "Retrieve FDA-approved drug label: black box warnings, contraindications, ADRs.",
        {"drug_name": _str_prop("Generic or brand drug name.")},
        ["drug_name"],
    ),
    _tool(
        "compare_drug_safety",
        "Head-to-head safety comparison between 2–5 drugs using FDA FAERS.",
        {
            "drugs": {
                "type": "array", "items": {"type": "string"},
                "description": "List of 2–5 drug names.",
            },
        },
        ["drugs"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 17: Variant Interpreter (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "classify_variant",
        "Classify a genetic variant using ACMG/AMP 2015 guidelines (5-tier output).",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "variant": _str_prop("Variant notation: protein, cDNA, rsID, or HGVS."),
            "inheritance": _enum_prop("Inheritance pattern.", ["AD", "AR", "XL", "unknown"], "unknown"),
        },
        ["gene_symbol", "variant"],
    ),
    _tool(
        "get_population_frequency",
        "Query gnomAD v4 for population-specific allele frequencies.",
        {
            "variant_id": _str_prop("Variant in rsID or gnomAD format."),
            "dataset": _enum_prop("gnomAD dataset.", ["gnomad_r4", "gnomad_r2_1"], "gnomad_r4"),
        },
        ["variant_id"],
    ),
    _tool(
        "lookup_clinvar_variant",
        "Search ClinVar for clinical significance, star rating, and submissions.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "variant": _str_prop("Variant notation."),
            "clinvar_id": _str_prop("Direct ClinVar variation ID."),
            "max_results": _int_prop("Maximum results", 5, 1, 20),
        },
        [],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 18: Innovations — NEW (7 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "bulk_gene_analysis",
        "Analyze multiple genes simultaneously in parallel and return a cross-gene "
        "comparison matrix. Queries NCBI Gene, ChEMBL, Open Targets, and Reactome "
        "for each gene, then synthesizes a comparative summary. Ideal for gene panel "
        "analysis, gene family studies, and multi-target drug discovery.",
        {
            "gene_symbols": {
                "type": "array", "items": {"type": "string"},
                "description": "List of 2–10 HGNC gene symbols to analyze in parallel.",
            },
            "comparison_axes": {
                "type": "array", "items": {"type": "string"},
                "description": "Aspects to compare: 'drugs', 'diseases', 'pathways', 'expression'. "
                               "Default: all four.",
            },
        },
        ["gene_symbols"],
    ),
    _tool(
        "compute_pathway_enrichment",
        "Fisher exact test pathway enrichment analysis for a gene list against KEGG/Reactome. "
        "Given a list of differentially expressed or mutated genes, identifies which pathways "
        "are statistically over-represented. Returns enriched pathways with p-values, "
        "FDR correction, and gene overlap lists — essential for omics data interpretation.",
        {
            "gene_list": {
                "type": "array", "items": {"type": "string"},
                "description": "List of HGNC gene symbols (e.g. from DE analysis or CRISPR screen).",
            },
            "background_size": _int_prop(
                "Total gene universe size for enrichment denominator. Default 20000.", 20000, 100, 30000
            ),
            "database": _enum_prop("Pathway database.", ["KEGG", "Reactome", "both"], "both"),
            "min_genes": _int_prop("Minimum gene overlap for pathway inclusion", 2, 1, 10),
            "fdr_threshold": _float_prop("FDR significance threshold. Default 0.05.", 0.05),
        },
        ["gene_list"],
    ),
    _tool(
        "search_biorxiv",
        "Search bioRxiv and medRxiv for recent preprints — access unpublished research "
        "up to 6 months before formal publication. Critical for staying current in "
        "fast-moving fields. Returns abstracts, author lists, posting date, "
        "DOI, and category tags. Can detect if a preprint has since been published.",
        {
            "query": _str_prop("Search query (e.g. 'KRAS G12C inhibitor 2025')."),
            "server": _enum_prop(
                "Preprint server.", ["biorxiv", "medrxiv", "both"], "both"
            ),
            "max_results": _int_prop("Results to return", 10, 1, 50),
            "days_back": _int_prop(
                "How many days back to search (max 365). Default 90.", 90, 1, 365
            ),
        },
        ["query"],
    ),
    _tool(
        "get_protein_domain_structure",
        "Retrieve protein domain architecture from InterPro — integrates PFam, SMART, "
        "PROSITE, CDD, and SUPERFAMILY domain annotations. Returns domain boundaries, "
        "domain family descriptions, 3D structure representatives, and known active/binding "
        "sites. Essential for understanding protein function from sequence alone.",
        {
            "uniprot_accession": _str_prop("UniProt accession (e.g. 'P04637')."),
            "include_disordered": _bool_prop(
                "Include predicted intrinsically disordered regions (MobiDB).", False
            ),
        },
        ["uniprot_accession"],
    ),
    _tool(
        "analyze_coexpression",
        "Compute pairwise co-expression correlation between two genes using TCGA "
        "RNA-seq data across cancer types, plus GTEx for normal tissue comparison. "
        "Returns Pearson and Spearman correlations, cancer-type breakdown, and "
        "literature support for the co-expression relationship. Identifies genes "
        "likely to be in the same pathway or regulatory module.",
        {
            "gene_a": _str_prop("First HGNC gene symbol."),
            "gene_b": _str_prop("Second HGNC gene symbol."),
            "cancer_types": {
                "type": "array", "items": {"type": "string"},
                "description": "TCGA cancer types (e.g. ['TCGA-LUAD', 'TCGA-BRCA']). "
                               "Empty = pan-cancer.",
            },
        },
        ["gene_a", "gene_b"],
    ),
    _tool(
        "get_cancer_hotspots",
        "Identify mutation hotspots for a gene using COSMIC Census + cBioPortal data. "
        "Returns positional distribution of somatic mutations, activating vs loss-of-function "
        "hotspot classification, affected protein domains, and cancer-type enrichment. "
        "Critical for understanding which mutations drive oncogenesis vs. passengers.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'KRAS', 'PIK3CA')."),
            "cancer_type": _str_prop("Specific cancer type to focus on. Empty = pan-cancer."),
            "min_samples": _int_prop("Minimum samples with mutation to call a hotspot", 5, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "predict_splice_impact",
        "Predict the functional impact of a variant on RNA splicing using SpliceAI-inspired "
        "rules and Ensembl VEP splice annotations. Detects: exon skipping, intron retention, "
        "cryptic splice site activation, and branch point disruption. Returns delta scores "
        "for acceptor/donor gain/loss, predicted new splice site position, and estimated "
        "fraction of transcripts affected. Critical for clinical variant interpretation.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "variant": _str_prop("Variant in cDNA notation (e.g. 'c.524+1G>A') or rsID."),
            "distance": _int_prop(
                "Distance from nearest splice site to analyze (bp). Default 50.", 50, 1, 200
            ),
        },
        ["gene_symbol", "variant"],
    ),
]

_LEGACY_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _legacy_tool(name: str) -> Tool:
    return _LEGACY_TOOLS_BY_NAME[name]


TOOLS = [
    _legacy_tool("search_pubmed"),
    _legacy_tool("get_gene_info"),
    _legacy_tool("run_blast"),
    _legacy_tool("get_protein_info"),
    _tool(
        "find_protein",
        "Unified protein discovery across UniProt and PDB. Use accession for a direct record lookup or query to search reviewed proteins and experimental structures.",
        {
            "query": _str_prop("Protein, gene, family, or free-text search query."),
            "source": _enum_prop("Which source to search.", ["auto", "both", "uniprot", "pdb"], "auto"),
            "accession": _str_prop("Optional UniProt accession for direct lookup."),
            "organism": _str_prop("Organism filter for UniProt search. Default 'homo sapiens'."),
            "reviewed_only": _bool_prop("Limit UniProt search to reviewed Swiss-Prot entries.", True),
            "max_results": _int_prop("Maximum results per source.", 10, 1, 25),
        },
        [],
    ),
    _legacy_tool("get_alphafold_structure"),
    _tool(
        "pathway_analysis",
        "Merged pathway workflow. Search KEGG, retrieve genes for a pathway, or assemble Reactome plus KEGG context for a gene.",
        {
            "action": _enum_prop("Workflow step.", ["auto", "search", "gene_context", "genes"], "auto"),
            "db": _enum_prop("Preferred pathway database.", ["auto", "kegg", "reactome"], "auto"),
            "query": _str_prop("Free-text pathway search term."),
            "gene_symbol": _str_prop("Gene symbol for Reactome or KEGG context."),
            "pathway_id": _str_prop("KEGG pathway identifier such as 'hsa05200'."),
            "organism": _str_prop("KEGG organism code. Default 'hsa'."),
        },
        [],
    ),
    _legacy_tool("get_drug_targets"),
    _legacy_tool("get_gene_disease_associations"),
    _legacy_tool("search_clinical_trials"),
    _legacy_tool("multi_omics_gene_report"),
    _tool(
        "predict_structure_boltz2",
        "Boltz-2 structure workflow. Use mode='structure' for direct multimolecular structure prediction or mode='protein_ligand' for the integrated UniProt-to-docking workflow.",
        {
            "mode": _enum_prop("Boltz-2 workflow mode.", ["structure", "protein_ligand"], "structure"),
            "protein_sequences": _array_prop("Protein sequences for direct Boltz-2 prediction."),
            "ligand_smiles": _array_prop("Ligand SMILES strings."),
            "dna_sequences": _array_prop("Optional DNA sequences for complex prediction."),
            "rna_sequences": _array_prop("Optional RNA sequences for complex prediction."),
            "uniprot_accession": _str_prop("UniProt accession used when mode='protein_ligand'."),
            "predict_affinity": _bool_prop("Predict ligand binding affinity when ligands are present.", False),
            "method_conditioning": _enum_prop("Optional structure-style conditioning.", ["x-ray", "nmr", "md"], "x-ray"),
            "pocket_residues": {
                "type": "array",
                "description": "Optional binding-pocket residue constraints.",
                "items": {"type": "object"},
            },
            "recycling_steps": _int_prop("Boltz-2 recycling iterations.", 3, 1, 10),
            "sampling_steps": _int_prop("Diffusion sampling steps.", 200, 50, 500),
            "diffusion_samples": _int_prop("Number of structural hypotheses.", 1, 1, 5),
        },
        [],
    ),
    _tool(
        "generate_dna_evo2",
        "Evo2 DNA workflow. Use mode='generate' for sequence generation or mode='score' for wildtype-versus-variant scoring.",
        {
            "mode": _enum_prop("Evo2 workflow mode.", ["generate", "score"], "generate"),
            "sequence": _str_prop("Seed DNA sequence for generation."),
            "num_tokens": _int_prop("Number of DNA tokens to generate.", 200, 1, 1200),
            "temperature": _float_prop("Sampling temperature.", 1.0),
            "top_k": _int_prop("Top-K sampling parameter.", 4, 0, 6),
            "top_p": _float_prop("Top-P sampling parameter.", 1.0),
            "enable_logits": _bool_prop("Return per-token logits for generation.", False),
            "num_generations": _int_prop("Independent generation runs.", 1, 1, 5),
            "wildtype_sequence": _str_prop("Reference DNA sequence used when mode='score'."),
            "variant_sequence": _str_prop("Variant DNA sequence used when mode='score'."),
        },
        [],
    ),
    _tool(
        "crispr_analysis",
        "Merged CRISPR workflow covering guide design, guide scoring, off-target review, base editing, and repair outcome estimation.",
        {
            "action": _enum_prop("CRISPR workflow step.", ["design", "score", "off_target", "base_edit", "repair"], "design"),
            "gene_symbol": _str_prop("Target gene symbol."),
            "guide_sequence": _str_prop("Guide sequence for scoring, off-target review, or repair analysis."),
            "target_mutation": _str_prop("Desired mutation for base-edit design."),
            "repair_template": _str_prop("Optional HDR repair template."),
            "cas_variant": _enum_prop("CRISPR nuclease.", ["SpCas9", "SaCas9", "Cas12a", "CjCas9"], "SpCas9"),
            "target_region": _str_prop("Guide search region, such as 'early_exons' or 'all_coding'."),
            "n_guides": _int_prop("Number of guides to return.", 5, 1, 20),
            "min_score": _float_prop("Minimum guide score for design.", 40.0),
            "mismatches": _int_prop("Maximum mismatches for off-target review.", 3, 1, 5),
            "genome": _enum_prop("Genome assembly used for off-target review.", ["hg38", "hg19"], "hg38"),
            "cell_line": _str_prop("Cell-line context for repair estimates."),
            "use_blast": _bool_prop("Use BLAST to supplement off-target review.", False),
        },
        ["action"],
    ),
    _tool(
        "drug_safety",
        "Merged FDA drug-safety workflow for adverse-event search, signal detection, label review, and head-to-head comparison.",
        {
            "action": _enum_prop("Drug-safety workflow step.", ["events", "signals", "label", "compare"], "events"),
            "drug_name": _str_prop("Generic or brand drug name."),
            "comparator_drug": _str_prop("Comparator drug used when action='compare'."),
            "event_type": _enum_prop("Safety category filter.", ["all", "cardiac", "hepatic", "hematologic", "neurological", "renal", "hypersensitivity", "respiratory", "oncology"], "all"),
            "serious_only": _bool_prop("Restrict adverse-event search to serious reports.", False),
            "event_terms": _array_prop("Optional adverse-event terms for signal detection."),
            "max_results": _int_prop("Maximum reports to summarize.", 50, 1, 100),
            "patient_sex": _enum_prop("Patient sex filter.", ["", "male", "female"], ""),
            "age_group": _enum_prop("Age filter.", ["", "pediatric", "adult", "elderly"], ""),
        },
        ["action", "drug_name"],
    ),
    _tool(
        "variant_analysis",
        "Merged variant-interpretation workflow for ACMG classification, population frequency, ClinVar review, splice review, and full integrated reporting.",
        {
            "action": _enum_prop("Variant workflow step.", ["classify", "population_frequency", "clinvar", "splice", "full_report"], "full_report"),
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "variant": _str_prop("Variant notation, rsID, HGVS, or protein-change string."),
            "inheritance": _enum_prop("Inheritance mode for ACMG scoring.", ["AD", "AR", "XL", "unknown"], "unknown"),
            "consequence": _str_prop("Optional VEP consequence if already known."),
            "proband_phenotype": _str_prop("Clinical phenotype context."),
            "populations": _array_prop("Population IDs to report for gnomAD frequencies."),
        },
        ["action"],
    ),
    _legacy_tool("find_repurposing_candidates"),
    _legacy_tool("verify_biological_claim"),
    _legacy_tool("search_cbio_mutations"),
    _legacy_tool("search_gwas_catalog"),
    _tool(
        "session",
        "Merged research-session workflow for entity resolution, knowledge-graph inspection, connection discovery, provenance export, and automated planning.",
        {
            "action": _enum_prop("Session workflow step.", ["resolve_entity", "knowledge_graph", "connections", "export", "plan"], "resolve_entity"),
            "query": _str_prop("Entity query or planning goal."),
            "hint_type": _str_prop("Entity type hint used for resolution."),
            "goal": _str_prop("Explicit research goal for planning."),
            "depth": _enum_prop("Planning depth.", ["quick", "standard", "deep"], "standard"),
            "min_path_length": _int_prop("Minimum path length for unexpected connections.", 2, 1, 6),
        },
        ["action"],
    ),
    _tool(
        "drug_interaction_checker",
        "Check FDA label interaction context between a primary drug and a list of co-medications.",
        {
            "drug_name": _str_prop("Primary drug name."),
            "co_medications": _array_prop("Co-medications to screen against the primary label."),
        },
        ["drug_name"],
    ),
    _tool(
        "protein_binding_pocket",
        "Summarize candidate binding sites from UniProt functional-site annotations plus AlphaFold confidence context.",
        {
            "accession": _str_prop("UniProt accession."),
            "query": _str_prop("Protein or gene query used to resolve a reviewed accession."),
            "max_sites": _int_prop("Maximum candidate sites to return.", 10, 1, 20),
        },
        [],
    ),
    _tool(
        "biomarker_panel_design",
        "Draft a disease-focused biomarker panel using Open Targets evidence with a literature fallback.",
        {
            "disease": _str_prop("Disease, indication, or phenotype of interest."),
            "panel_size": _int_prop("Number of biomarkers to include.", 10, 3, 25),
            "context": _str_prop("Context such as oncology, inflammation, or rare disease."),
        },
        ["disease"],
    ),
    _tool(
        "pharmacogenomics_report",
        "Summarize CPIC-style pharmacogenomic genes and supporting PGx evidence for a drug.",
        {
            "drug_name": _str_prop("Drug of interest."),
            "gene_symbol": _str_prop("Optional gene to force into the report."),
            "max_annotations": _int_prop("Maximum supporting annotations per gene.", 10, 1, 25),
        },
        ["drug_name"],
    ),
    _tool(
        "protein_family_analysis",
        "Summarize protein family and domain context from curated UniProt annotations with direct Pfam and InterPro links.",
        {
            "accession": _str_prop("UniProt accession."),
            "query": _str_prop("Protein or gene query used to resolve an accession."),
        },
        [],
    ),
    _tool(
        "network_enrichment",
        "Summarize recurrent Reactome pathways and STRING network hubs across a gene set.",
        {
            "gene_list": _array_prop("Input genes for enrichment analysis."),
            "min_string_score": _int_prop("Minimum STRING confidence score.", 700, 0, 1000),
            "max_results": _int_prop("Maximum pathways and hubs to return.", 10, 3, 25),
        },
        ["gene_list"],
    ),
    _tool(
        "rnaseq_deconvolution",
        "Marker-based heuristic deconvolution of a bulk RNA-seq profile into likely cell-type fractions.",
        {
            "expression_profile": _obj_prop("Expression profile keyed by gene symbol with numeric abundance values."),
            "ranked_genes": _array_prop("Optional ranked marker genes when numeric expression is unavailable."),
            "max_cell_types": _int_prop("Maximum cell types to return.", 5, 2, 10),
        },
        [],
    ),
    _tool(
        "structural_similarity",
        "PubChem-backed structural similarity search from a compound name or SMILES string.",
        {
            "query": _str_prop("Compound name or identifier."),
            "smiles": _str_prop("Canonical or query SMILES string."),
            "threshold": _int_prop("PubChem 2D similarity threshold.", 90, 70, 99),
            "max_results": _int_prop("Maximum similar compounds to return.", 10, 1, 25),
        },
        [],
    ),
    _tool(
        "rare_disease_diagnosis",
        "Normalize phenotype terms and rank OMIM differentials for a candidate gene.",
        {
            "phenotype_terms": _array_prop("Phenotype terms or symptoms to normalize."),
            "gene_symbol": _str_prop("Candidate gene symbol."),
            "max_results": _int_prop("Maximum ranked differentials to return.", 10, 1, 20),
        },
        [],
    ),
    _tool(
        "genome_browser_snapshot",
        "Generate genome-browser links and locus context for a gene or explicit genomic interval.",
        {
            "gene_symbol": _str_prop("Gene symbol to resolve to a locus."),
            "region": _str_prop("Explicit region such as 'chr17:43044295-43170245'."),
            "flank_bp": _int_prop("Flanking sequence to include around the locus.", 25000, 1000, 500000),
            "assembly": _enum_prop("Genome assembly.", ["GRCh38", "GRCh37"], "GRCh38"),
        },
        [],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Handler
# ─────────────────────────────────────────────────────────────────────────────


_PUBLIC_TOOL_EXAMPLES: dict[str, list[dict[str, Any]]] = {
    "search_pubmed": [
        {"query": "KRAS G12C inhibitor lung cancer", "max_results": 5, "sort": "relevance"}
    ],
    "get_gene_info": [{"gene_symbol": "TP53", "organism": "homo sapiens"}],
    "run_blast": [
        {
            "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPEN",
            "program": "blastp",
            "database": "swissprot",
            "max_hits": 5,
        }
    ],
    "get_protein_info": [{"accession": "P04637"}],
    "find_protein": [
        {
            "query": "EGFR kinase domain",
            "source": "both",
            "organism": "homo sapiens",
            "reviewed_only": True,
            "max_results": 5,
        }
    ],
    "get_alphafold_structure": [{"uniprot_accession": "P04637", "model_version": "v4"}],
    "pathway_analysis": [
        {
            "action": "gene_context",
            "db": "reactome",
            "gene_symbol": "KRAS",
            "organism": "hsa",
        }
    ],
    "get_drug_targets": [{"gene_symbol": "EGFR", "max_results": 10}],
    "get_gene_disease_associations": [{"gene_symbol": "BRCA1", "max_results": 10}],
    "search_clinical_trials": [
        {
            "query": "EGFR non-small cell lung cancer",
            "status": "RECRUITING",
            "phase": "PHASE2",
            "max_results": 5,
        }
    ],
    "multi_omics_gene_report": [{"gene_symbol": "MYC"}],
    "predict_structure_boltz2": [
        {
            "mode": "structure",
            "protein_sequences": ["MKTAYIAKQRQISFVKSHFSRQ"],
            "diffusion_samples": 1,
        }
    ],
    "generate_dna_evo2": [
        {
            "mode": "generate",
            "sequence": "ATGCGTATGCGT",
            "num_tokens": 80,
            "temperature": 0.8,
            "top_k": 4,
            "top_p": 0.95,
            "num_generations": 1,
        }
    ],
    "crispr_analysis": [
        {
            "action": "design",
            "gene_symbol": "PCSK9",
            "cas_variant": "SpCas9",
            "n_guides": 4,
            "genome": "hg38",
        }
    ],
    "drug_safety": [
        {
            "action": "events",
            "drug_name": "warfarin",
            "event_type": "cardiac",
            "serious_only": True,
            "max_results": 5,
        }
    ],
    "variant_analysis": [
        {
            "action": "full_report",
            "gene_symbol": "BRCA1",
            "variant": "c.68_69delAG",
            "inheritance": "AD",
        }
    ],
    "find_repurposing_candidates": [
        {"disease": "idiopathic pulmonary fibrosis", "max_candidates": 10}
    ],
    "verify_biological_claim": [
        {
            "claim": "KRAS G12C inhibitors improve progression-free survival in NSCLC",
            "context_gene": "KRAS",
        }
    ],
    "search_cbio_mutations": [
        {"gene_symbol": "TP53", "cancer_type": "breast cancer", "max_studies": 5}
    ],
    "search_gwas_catalog": [{"gene_symbol": "APOE", "max_results": 10}],
    "session": [{"action": "resolve_entity", "query": "EGFR", "hint_type": "gene"}],
    "drug_interaction_checker": [
        {"drug_name": "warfarin", "co_medications": ["amiodarone", "fluconazole"]}
    ],
    "protein_binding_pocket": [{"accession": "P00533", "max_sites": 5}],
    "biomarker_panel_design": [
        {
            "disease": "triple negative breast cancer",
            "panel_size": 8,
            "context": "tissue biopsy",
        }
    ],
    "pharmacogenomics_report": [
        {"drug_name": "clopidogrel", "gene_symbol": "CYP2C19", "max_annotations": 8}
    ],
    "protein_family_analysis": [{"query": "EGFR family kinases"}],
    "network_enrichment": [
        {"gene_list": ["TP53", "MDM2", "ATM"], "min_string_score": 700, "max_results": 10}
    ],
    "rnaseq_deconvolution": [
        {"ranked_genes": ["EPCAM", "KRT18", "COL1A1", "PTPRC"], "max_cell_types": 5}
    ],
    "structural_similarity": [{"query": "EGFR inhibitors", "threshold": 85, "max_results": 8}],
    "rare_disease_diagnosis": [
        {"phenotype_terms": ["developmental delay", "ataxia", "seizures"], "max_results": 5}
    ],
    "genome_browser_snapshot": [{"gene_symbol": "BRCA1", "assembly": "GRCh38", "flank_bp": 25000}],
}

_RESOURCE_CAPABILITIES_URI = "biomcp://server/capabilities"
_RESOURCE_TOOL_CATALOG_URI = "biomcp://tools/catalog"
_RESOURCE_METADATA: dict[str, dict[str, str]] = {
    _RESOURCE_CAPABILITIES_URI: {
        "name": "server-capabilities",
        "title": "BioMCP Server Capabilities",
        "description": "Transport, health, capability, and deployment-facing metadata.",
    },
    _RESOURCE_TOOL_CATALOG_URI: {
        "name": "tool-catalog",
        "title": "BioMCP Tool Catalog",
        "description": "Public tool inventory with required arguments and example payloads.",
    },
}


def _apply_public_tool_examples() -> None:
    for tool in TOOLS:
        examples = _PUBLIC_TOOL_EXAMPLES.get(tool.name)
        if examples:
            tool.inputSchema["examples"] = examples


def _tool_catalog_entries() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "required": tool.inputSchema.get("required", []),
            "properties": list(tool.inputSchema.get("properties", {}).keys()),
            "examples": tool.inputSchema.get("examples", []),
        }
        for tool in TOOLS
    ]


def _resource_payload(uri: str) -> dict[str, Any]:
    if uri == _RESOURCE_CAPABILITIES_URI:
        return {
            "service": SERVER_NAME,
            "display_name": SERVER_DISPLAY_NAME,
            "version": __version__,
            "transport_modes": ["stdio", "http"],
            "transport_endpoints": {
                "streamable_http": STREAMABLE_HTTP_PATH,
                "sse": SSE_PATH,
                "messages": MESSAGE_PATH,
            },
            "health_endpoints": ["/health", "/healthz", "/readyz", "/tool-health"],
            "public_tool_count": len(TOOLS),
            "resource_uris": sorted(_RESOURCE_METADATA),
            "capabilities": _build_capability_status(),
        }
    if uri == _RESOURCE_TOOL_CATALOG_URI:
        return {
            "service": SERVER_NAME,
            "version": __version__,
            "tool_count": len(TOOLS),
            "tools": _tool_catalog_entries(),
        }
    raise ValueError(f"Unknown resource '{uri}'")


def _list_resource_definitions() -> list[Resource]:
    return [
        Resource(
            name=meta["name"],
            title=meta["title"],
            uri=uri,
            description=meta["description"],
            mimeType="application/json",
        )
        for uri, meta in _RESOURCE_METADATA.items()
    ]


def _read_resource_contents(uri: Any) -> list[ReadResourceContents]:
    payload = json.dumps(_resource_payload(str(uri)), indent=2, sort_keys=True)
    return [ReadResourceContents(content=payload, mime_type="application/json")]


_apply_public_tool_examples()


async def _generate_research_hypothesis(
    topic: str,
    context_genes: list[str] | None = None,
    max_hypotheses: int = 3,
) -> dict[str, Any]:
    from biomcp.tools.ncbi import search_pubmed
    from biomcp.utils import BioValidator

    genes = context_genes or []
    max_hyp = BioValidator.clamp_int(max_hypotheses, 1, 10, "max_hypotheses")
    gene_clause = " OR ".join(genes[:5]) if genes else ""
    query = f"({topic})" + (f" AND ({gene_clause})" if gene_clause else "") + " AND Review[pt]"

    papers = await search_pubmed(query, max_results=20)
    articles = papers.get("articles", [])

    mesh_coverage: dict[str, int] = {}
    for art in articles:
        for mesh in art.get("mesh_terms", []):
            mesh_coverage[mesh] = mesh_coverage.get(mesh, 0) + 1

    top_mesh = sorted(mesh_coverage, key=mesh_coverage.__getitem__, reverse=True)[:8]

    hypotheses = []
    for i in range(min(max_hyp, 5)):
        target = (
            genes[i]
            if i < len(genes)
            else (top_mesh[i] if i < len(top_mesh) else "key pathway nodes")
        )
        hypotheses.append({
            "id": i + 1,
            "title": f"Hypothesis {i + 1}: Role of {target} in {topic}",
            "rationale": (
                f"Based on {len(articles)} review articles, {target} appears as a "
                f"recurring theme in '{topic}'. Mechanistic validation is lacking."
            ),
            "supporting_paper_count": max(0, len(articles) - i * 2),
            "key_mesh_context": top_mesh[:5],
            "suggested_experiments": [
                f"CRISPR knockdown of {target} in relevant cell line",
                "RNA-seq differential expression under perturbed conditions",
                "Protein interaction network analysis (STRING/BioGRID)",
            ],
        })

    return {
        "topic": topic, "context_genes": genes,
        "literature_base": {
            "query": query,
            "total_papers": papers.get("total_found", 0),
            "top_mesh_terms": top_mesh,
        },
        "hypotheses": hypotheses,
        "disclaimer": "AI-generated hypotheses from literature patterns. Validate with domain expertise.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session Intelligence Handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _resolve_entity(query: str, hint_type: str = "gene") -> dict[str, Any]:
    from biomcp.core.entity_resolver import get_resolver
    resolver = await get_resolver()
    entity = await resolver.resolve(query, hint_type=hint_type)
    return entity.to_dict()


async def _get_session_knowledge_graph() -> dict[str, Any]:
    from biomcp.core.knowledge_graph import get_skg
    skg = await get_skg()
    return skg.snapshot()


async def _find_biological_connections(min_path_length: int = 2) -> dict[str, Any]:
    from biomcp.core.knowledge_graph import get_skg
    skg = await get_skg()
    connections = skg.find_unexpected_connections(min_path_length=min_path_length)
    stats = skg.stats()
    return {
        "connections_found": len(connections),
        "connections": connections,
        "graph_stats": stats,
    }


async def _export_research_session() -> dict[str, Any]:
    from biomcp.core.knowledge_graph import get_skg
    skg = await get_skg()
    return skg.export_provenance()


async def _plan_and_execute_research(
    goal: str,
    depth: str = "standard",
    gene: str = "",
    uniprot: str = "",
    timeout_per_tool: int = 60,
) -> dict[str, Any]:
    from biomcp.core.query_planner import AdaptiveQueryPlanner
    entities: dict[str, str] = {}
    if gene:
        entities["gene"] = gene.upper()
    if uniprot:
        entities["uniprot"] = uniprot
    planner = AdaptiveQueryPlanner(dispatcher=_raw_dispatch)
    return await planner.plan_and_execute(
        goal=goal, depth=depth,
        entities=entities or None,
        timeout_per_tool=float(timeout_per_tool),
    )


async def _session_workflow(
    action: str,
    query: str = "",
    hint_type: str = "gene",
    goal: str = "",
    depth: str = "standard",
    min_path_length: int = 2,
) -> dict[str, Any]:
    action = action.lower()
    if action == "resolve_entity":
        if not query:
            raise ValueError("query is required when action='resolve_entity'.")
        return await _resolve_entity(query=query, hint_type=hint_type)
    if action == "knowledge_graph":
        return await _get_session_knowledge_graph()
    if action == "connections":
        return await _find_biological_connections(min_path_length=min_path_length)
    if action == "export":
        return await _export_research_session()
    if action == "plan":
        plan_goal = goal or query
        if not plan_goal:
            raise ValueError("goal or query is required when action='plan'.")
        return await _plan_and_execute_research(goal=plan_goal, depth=depth)
    raise ValueError("Unsupported session action.")


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — FIX #2: removed all duplicate import blocks
# ─────────────────────────────────────────────────────────────────────────────


async def _raw_dispatch_legacy(name: str, args: dict[str, Any]) -> Any:
    """Raw dispatcher returning Python objects. Used by the query planner."""
    # ── Core tool imports ──────────────────────────────────────────────────────
    from biomcp.tools.advanced import (
        get_gene_variants,
        get_trial_details,
        multi_omics_gene_report,
        query_neuroimaging_datasets,
        search_clinical_trials,
        search_gene_expression,
        search_scrna_datasets,
    )
    from biomcp.tools.crispr_tools import (
        design_base_editor_guides,
        design_crispr_guides,
        get_crispr_repair_outcomes,
        predict_off_target_sites,
        score_guide_efficiency,
    )
    from biomcp.tools.databases import (
        get_disgenet_associations,
        get_gtex_expression,
        get_omim_gene_diseases,
        get_pharmgkb_variants,
        get_string_interactions,
        search_cbio_mutations,
        search_gwas_catalog,
    )
    from biomcp.tools.drug_safety import (
        analyze_safety_signals,
        compare_drug_safety,
        get_drug_label_warnings,
        query_adverse_events,
    )

    # FIX #2: Single import of extended_databases (removed duplicate)
    from biomcp.tools.extended_databases import (
        get_biogrid_interactions,
        get_encode_regulatory,
        get_tcga_expression,
        get_ucsc_splice_variants,
        search_cellmarker,
        search_metabolomics,
        search_orphan_diseases,
    )

    # New innovation tools
    from biomcp.tools.innovations import (
        analyze_coexpression,
        bulk_gene_analysis,
        compute_pathway_enrichment,
        get_cancer_hotspots,
        get_protein_domain_structure,
        predict_splice_impact,
        search_biorxiv,
    )
    from biomcp.tools.intelligence import (
        find_repurposing_candidates,
        find_research_gaps,
        validate_reasoning_chain,
    )
    from biomcp.tools.ncbi import get_gene_info, run_blast, search_pubmed
    from biomcp.tools.nvidia_nim import design_protein_ligand, score_sequence_evo2
    from biomcp.tools.pathways import (
        get_compound_info,
        get_drug_targets,
        get_gene_disease_associations,
        get_pathway_genes,
        get_reactome_pathways,
        search_pathways,
    )
    from biomcp.tools.proteins import (
        get_alphafold_structure,
        get_protein_info,
        search_pdb_structures,
        search_proteins,
    )
    from biomcp.tools.protocol_generator import (
        estimate_statistical_power,
        generate_experimental_protocol,
        suggest_cell_lines,
    )
    from biomcp.tools.strategy_surface import (
        biomarker_panel_design,
        boltz2_workflow,
        crispr_analysis,
        drug_interaction_checker,
        evo2_workflow,
        find_protein,
        genome_browser_snapshot,
        network_enrichment,
        pathway_analysis,
        pharmacogenomics_report,
        protein_binding_pocket,
        protein_family_analysis,
        rare_disease_diagnosis,
        rnaseq_deconvolution,
        structural_similarity,
        variant_analysis,
    )
    from biomcp.tools.strategy_surface import (
        drug_safety as drug_safety_workflow,
    )
    from biomcp.tools.variant_interpreter import (
        classify_variant,
        get_population_frequency,
        lookup_clinvar_variant,
    )

    # FIX #2: Single import of verify and protocol tools (removed duplicates)
    from biomcp.tools.verify import detect_database_conflicts, verify_biological_claim

    DISPATCH: dict[str, Any] = {
        # Literature
        "search_pubmed": search_pubmed,
        "get_gene_info": get_gene_info,
        "run_blast": run_blast,
        # Proteins
        "get_protein_info": get_protein_info,
        "search_proteins": search_proteins,
        "get_alphafold_structure": get_alphafold_structure,
        "search_pdb_structures": search_pdb_structures,
        "find_protein": find_protein,
        # Pathways
        "search_pathways": search_pathways,
        "get_pathway_genes": get_pathway_genes,
        "get_reactome_pathways": get_reactome_pathways,
        "pathway_analysis": pathway_analysis,
        # Drug Discovery
        "get_drug_targets": get_drug_targets,
        "get_compound_info": get_compound_info,
        "get_gene_disease_associations": get_gene_disease_associations,
        # Genomics
        "get_gene_variants": get_gene_variants,
        "search_gene_expression": search_gene_expression,
        "search_scrna_datasets": search_scrna_datasets,
        # Clinical
        "search_clinical_trials": search_clinical_trials,
        "get_trial_details": get_trial_details,
        # Advanced
        "multi_omics_gene_report": multi_omics_gene_report,
        "query_neuroimaging_datasets": query_neuroimaging_datasets,
        "generate_research_hypothesis": _module_dispatch(
            sys.modules[__name__], "_generate_research_hypothesis"
        ),
        # NVIDIA NIM
        "predict_structure_boltz2": boltz2_workflow,
        "generate_dna_evo2": evo2_workflow,
        "score_sequence_evo2": score_sequence_evo2,
        "design_protein_ligand": design_protein_ligand,
        # Extended Databases
        "get_omim_gene_diseases": get_omim_gene_diseases,
        "get_string_interactions": get_string_interactions,
        "get_gtex_expression": get_gtex_expression,
        "search_cbio_mutations": search_cbio_mutations,
        "search_gwas_catalog": search_gwas_catalog,
        "get_disgenet_associations": get_disgenet_associations,
        "get_pharmgkb_variants": get_pharmgkb_variants,
        # Verification
        "verify_biological_claim": verify_biological_claim,
        "detect_database_conflicts": detect_database_conflicts,
        # Experimental Design
        "generate_experimental_protocol": generate_experimental_protocol,
        "suggest_cell_lines": suggest_cell_lines,
        "estimate_statistical_power": estimate_statistical_power,
        # Session Intelligence
        "session": _module_dispatch(sys.modules[__name__], "_session_workflow"),
        "resolve_entity": _module_dispatch(sys.modules[__name__], "_resolve_entity"),
        "get_session_knowledge_graph": _module_dispatch(
            sys.modules[__name__], "_get_session_knowledge_graph"
        ),
        "find_biological_connections": _module_dispatch(
            sys.modules[__name__], "_find_biological_connections"
        ),
        "export_research_session": _module_dispatch(
            sys.modules[__name__], "_export_research_session"
        ),
        "plan_and_execute_research": _module_dispatch(
            sys.modules[__name__], "_plan_and_execute_research"
        ),
        # Intelligence Layer
        "validate_reasoning_chain": validate_reasoning_chain,
        "find_repurposing_candidates": find_repurposing_candidates,
        "find_research_gaps": find_research_gaps,
        # Tier 2 Extended
        "get_biogrid_interactions": get_biogrid_interactions,
        "search_orphan_diseases": search_orphan_diseases,
        "get_tcga_expression": get_tcga_expression,
        "search_cellmarker": search_cellmarker,
        "get_encode_regulatory": get_encode_regulatory,
        "search_metabolomics": search_metabolomics,
        "get_ucsc_splice_variants": get_ucsc_splice_variants,
        # CRISPR
        "crispr_analysis": crispr_analysis,
        "design_crispr_guides": design_crispr_guides,
        "score_guide_efficiency": score_guide_efficiency,
        "predict_off_target_sites": predict_off_target_sites,
        "design_base_editor_guides": design_base_editor_guides,
        "get_crispr_repair_outcomes": get_crispr_repair_outcomes,
        # FDA Drug Safety
        "drug_safety": drug_safety_workflow,
        "query_adverse_events": query_adverse_events,
        "analyze_safety_signals": analyze_safety_signals,
        "get_drug_label_warnings": get_drug_label_warnings,
        "compare_drug_safety": compare_drug_safety,
        # Variant Interpreter
        "variant_analysis": variant_analysis,
        "classify_variant": classify_variant,
        "get_population_frequency": get_population_frequency,
        "lookup_clinvar_variant": lookup_clinvar_variant,
        # Innovations
        "bulk_gene_analysis": bulk_gene_analysis,
        "compute_pathway_enrichment": compute_pathway_enrichment,
        "search_biorxiv": search_biorxiv,
        "get_protein_domain_structure": get_protein_domain_structure,
        "analyze_coexpression": analyze_coexpression,
        "get_cancer_hotspots": get_cancer_hotspots,
        "predict_splice_impact": predict_splice_impact,
        # Strategy additions
        "drug_interaction_checker": drug_interaction_checker,
        "protein_binding_pocket": protein_binding_pocket,
        "biomarker_panel_design": biomarker_panel_design,
        "pharmacogenomics_report": pharmacogenomics_report,
        "protein_family_analysis": protein_family_analysis,
        "network_enrichment": network_enrichment,
        "rnaseq_deconvolution": rnaseq_deconvolution,
        "structural_similarity": structural_similarity,
        "rare_disease_diagnosis": rare_disease_diagnosis,
        "genome_browser_snapshot": genome_browser_snapshot,
    }

    if name not in DISPATCH:
        raise ValueError(f"Unknown tool '{name}'")
    return await DISPATCH[name](**args)


def _module_dispatch(module: Any, attr_name: str) -> Callable[..., Any]:
    async def _call(**kwargs: Any) -> Any:
        return await getattr(module, attr_name)(**kwargs)

    return _call


def _build_tool_modules() -> dict[str, Any]:
    return {
        "advanced": importlib.import_module("biomcp.tools.advanced"),
        "crispr": importlib.import_module("biomcp.tools.crispr_tools"),
        "databases": importlib.import_module("biomcp.tools.databases"),
        "drug_safety": importlib.import_module("biomcp.tools.drug_safety"),
        "extended_databases": importlib.import_module("biomcp.tools.extended_databases"),
        "innovations": importlib.import_module("biomcp.tools.innovations"),
        "intelligence": importlib.import_module("biomcp.tools.intelligence"),
        "ncbi": importlib.import_module("biomcp.tools.ncbi"),
        "nvidia_nim": importlib.import_module("biomcp.tools.nvidia_nim"),
        "pathways": importlib.import_module("biomcp.tools.pathways"),
        "proteins": importlib.import_module("biomcp.tools.proteins"),
        "protocol_generator": importlib.import_module("biomcp.tools.protocol_generator"),
        "strategy_surface": importlib.import_module("biomcp.tools.strategy_surface"),
        "variant_interpreter": importlib.import_module("biomcp.tools.variant_interpreter"),
        "verify": importlib.import_module("biomcp.tools.verify"),
    }


def _get_tool_modules() -> dict[str, Any]:
    global _TOOL_MODULES
    if _TOOL_MODULES is None:
        _TOOL_MODULES = _build_tool_modules()
        logger.debug(f"Tool modules initialized with {len(_TOOL_MODULES)} modules")
    return _TOOL_MODULES


def _build_dispatch_table() -> dict[str, Callable[..., Any]]:
    tool_modules = _get_tool_modules()
    advanced_tools = tool_modules["advanced"]
    crispr_tools = tool_modules["crispr"]
    database_tools = tool_modules["databases"]
    drug_safety_tools = tool_modules["drug_safety"]
    extended_database_tools = tool_modules["extended_databases"]
    innovation_tools = tool_modules["innovations"]
    intelligence_tools = tool_modules["intelligence"]
    ncbi_tools = tool_modules["ncbi"]
    nvidia_nim_tools = tool_modules["nvidia_nim"]
    pathway_tools = tool_modules["pathways"]
    protein_tools = tool_modules["proteins"]
    protocol_tools = tool_modules["protocol_generator"]
    strategy_tools = tool_modules["strategy_surface"]
    variant_tools = tool_modules["variant_interpreter"]
    verification_tools = tool_modules["verify"]

    return {
        "search_pubmed": _module_dispatch(ncbi_tools, "search_pubmed"),
        "get_gene_info": _module_dispatch(ncbi_tools, "get_gene_info"),
        "run_blast": _module_dispatch(ncbi_tools, "run_blast"),
        "get_protein_info": _module_dispatch(protein_tools, "get_protein_info"),
        "search_proteins": _module_dispatch(protein_tools, "search_proteins"),
        "get_alphafold_structure": _module_dispatch(protein_tools, "get_alphafold_structure"),
        "search_pdb_structures": _module_dispatch(protein_tools, "search_pdb_structures"),
        "find_protein": _module_dispatch(strategy_tools, "find_protein"),
        "search_pathways": _module_dispatch(pathway_tools, "search_pathways"),
        "get_pathway_genes": _module_dispatch(pathway_tools, "get_pathway_genes"),
        "get_reactome_pathways": _module_dispatch(pathway_tools, "get_reactome_pathways"),
        "pathway_analysis": _module_dispatch(strategy_tools, "pathway_analysis"),
        "get_drug_targets": _module_dispatch(pathway_tools, "get_drug_targets"),
        "get_compound_info": _module_dispatch(pathway_tools, "get_compound_info"),
        "get_gene_disease_associations": _module_dispatch(pathway_tools, "get_gene_disease_associations"),
        "get_gene_variants": _module_dispatch(advanced_tools, "get_gene_variants"),
        "search_gene_expression": _module_dispatch(advanced_tools, "search_gene_expression"),
        "search_scrna_datasets": _module_dispatch(advanced_tools, "search_scrna_datasets"),
        "search_clinical_trials": _module_dispatch(advanced_tools, "search_clinical_trials"),
        "get_trial_details": _module_dispatch(advanced_tools, "get_trial_details"),
        "multi_omics_gene_report": _module_dispatch(advanced_tools, "multi_omics_gene_report"),
        "query_neuroimaging_datasets": _module_dispatch(advanced_tools, "query_neuroimaging_datasets"),
        "generate_research_hypothesis": _module_dispatch(
            sys.modules[__name__], "_generate_research_hypothesis"
        ),
        "predict_structure_boltz2": _module_dispatch(strategy_tools, "boltz2_workflow"),
        "generate_dna_evo2": _module_dispatch(strategy_tools, "evo2_workflow"),
        "score_sequence_evo2": _module_dispatch(nvidia_nim_tools, "score_sequence_evo2"),
        "design_protein_ligand": _module_dispatch(nvidia_nim_tools, "design_protein_ligand"),
        "get_omim_gene_diseases": _module_dispatch(database_tools, "get_omim_gene_diseases"),
        "get_string_interactions": _module_dispatch(database_tools, "get_string_interactions"),
        "get_gtex_expression": _module_dispatch(database_tools, "get_gtex_expression"),
        "search_cbio_mutations": _module_dispatch(database_tools, "search_cbio_mutations"),
        "search_gwas_catalog": _module_dispatch(database_tools, "search_gwas_catalog"),
        "get_disgenet_associations": _module_dispatch(database_tools, "get_disgenet_associations"),
        "get_pharmgkb_variants": _module_dispatch(database_tools, "get_pharmgkb_variants"),
        "verify_biological_claim": _module_dispatch(verification_tools, "verify_biological_claim"),
        "detect_database_conflicts": _module_dispatch(verification_tools, "detect_database_conflicts"),
        "generate_experimental_protocol": _module_dispatch(protocol_tools, "generate_experimental_protocol"),
        "suggest_cell_lines": _module_dispatch(protocol_tools, "suggest_cell_lines"),
        "estimate_statistical_power": _module_dispatch(protocol_tools, "estimate_statistical_power"),
        "session": _module_dispatch(sys.modules[__name__], "_session_workflow"),
        "resolve_entity": _module_dispatch(sys.modules[__name__], "_resolve_entity"),
        "get_session_knowledge_graph": _module_dispatch(
            sys.modules[__name__], "_get_session_knowledge_graph"
        ),
        "find_biological_connections": _module_dispatch(
            sys.modules[__name__], "_find_biological_connections"
        ),
        "export_research_session": _module_dispatch(
            sys.modules[__name__], "_export_research_session"
        ),
        "plan_and_execute_research": _module_dispatch(
            sys.modules[__name__], "_plan_and_execute_research"
        ),
        "validate_reasoning_chain": _module_dispatch(intelligence_tools, "validate_reasoning_chain"),
        "find_repurposing_candidates": _module_dispatch(intelligence_tools, "find_repurposing_candidates"),
        "find_research_gaps": _module_dispatch(intelligence_tools, "find_research_gaps"),
        "get_biogrid_interactions": _module_dispatch(extended_database_tools, "get_biogrid_interactions"),
        "search_orphan_diseases": _module_dispatch(extended_database_tools, "search_orphan_diseases"),
        "get_tcga_expression": _module_dispatch(extended_database_tools, "get_tcga_expression"),
        "search_cellmarker": _module_dispatch(extended_database_tools, "search_cellmarker"),
        "get_encode_regulatory": _module_dispatch(extended_database_tools, "get_encode_regulatory"),
        "search_metabolomics": _module_dispatch(extended_database_tools, "search_metabolomics"),
        "get_ucsc_splice_variants": _module_dispatch(extended_database_tools, "get_ucsc_splice_variants"),
        "crispr_analysis": _module_dispatch(strategy_tools, "crispr_analysis"),
        "design_crispr_guides": _module_dispatch(crispr_tools, "design_crispr_guides"),
        "score_guide_efficiency": _module_dispatch(crispr_tools, "score_guide_efficiency"),
        "predict_off_target_sites": _module_dispatch(crispr_tools, "predict_off_target_sites"),
        "design_base_editor_guides": _module_dispatch(crispr_tools, "design_base_editor_guides"),
        "get_crispr_repair_outcomes": _module_dispatch(crispr_tools, "get_crispr_repair_outcomes"),
        "drug_safety": _module_dispatch(strategy_tools, "drug_safety"),
        "query_adverse_events": _module_dispatch(drug_safety_tools, "query_adverse_events"),
        "analyze_safety_signals": _module_dispatch(drug_safety_tools, "analyze_safety_signals"),
        "get_drug_label_warnings": _module_dispatch(drug_safety_tools, "get_drug_label_warnings"),
        "compare_drug_safety": _module_dispatch(drug_safety_tools, "compare_drug_safety"),
        "variant_analysis": _module_dispatch(strategy_tools, "variant_analysis"),
        "classify_variant": _module_dispatch(variant_tools, "classify_variant"),
        "get_population_frequency": _module_dispatch(variant_tools, "get_population_frequency"),
        "lookup_clinvar_variant": _module_dispatch(variant_tools, "lookup_clinvar_variant"),
        "bulk_gene_analysis": _module_dispatch(innovation_tools, "bulk_gene_analysis"),
        "compute_pathway_enrichment": _module_dispatch(innovation_tools, "compute_pathway_enrichment"),
        "search_biorxiv": _module_dispatch(innovation_tools, "search_biorxiv"),
        "get_protein_domain_structure": _module_dispatch(innovation_tools, "get_protein_domain_structure"),
        "analyze_coexpression": _module_dispatch(innovation_tools, "analyze_coexpression"),
        "get_cancer_hotspots": _module_dispatch(innovation_tools, "get_cancer_hotspots"),
        "predict_splice_impact": _module_dispatch(innovation_tools, "predict_splice_impact"),
        "drug_interaction_checker": _module_dispatch(strategy_tools, "drug_interaction_checker"),
        "protein_binding_pocket": _module_dispatch(strategy_tools, "protein_binding_pocket"),
        "biomarker_panel_design": _module_dispatch(strategy_tools, "biomarker_panel_design"),
        "pharmacogenomics_report": _module_dispatch(strategy_tools, "pharmacogenomics_report"),
        "protein_family_analysis": _module_dispatch(strategy_tools, "protein_family_analysis"),
        "network_enrichment": _module_dispatch(strategy_tools, "network_enrichment"),
        "rnaseq_deconvolution": _module_dispatch(strategy_tools, "rnaseq_deconvolution"),
        "structural_similarity": _module_dispatch(strategy_tools, "structural_similarity"),
        "rare_disease_diagnosis": _module_dispatch(strategy_tools, "rare_disease_diagnosis"),
        "genome_browser_snapshot": _module_dispatch(strategy_tools, "genome_browser_snapshot"),
    }


def _get_dispatch_table() -> dict[str, Callable[..., Any]]:
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is None:
        _DISPATCH_TABLE = _build_dispatch_table()
        logger.debug(f"Dispatch table initialized with {len(_DISPATCH_TABLE)} handlers")
    return _DISPATCH_TABLE


async def _raw_dispatch(name: str, args: dict[str, Any]) -> Any:
    """Raw dispatcher returning Python objects. Used by the query planner."""
    dispatch = _get_dispatch_table()
    if name not in dispatch:
        raise ValueError(f"Unknown tool '{name}'")
    return await dispatch[name](**args)


async def _dispatch(name: str, args: dict[str, Any]) -> str:
    """MCP-facing dispatcher — wraps results in JSON envelopes."""
    try:
        result = await _raw_dispatch(name, args)
        return format_success(name, result)
    except (ValueError, TypeError, LookupError, KeyError) as exc:
        return format_error(name, exc, {"arguments": args})
    except httpx.HTTPError as exc:
        return format_error(name, exc, {"arguments": args})
    except Exception as exc:
        logger.exception(f"Unexpected error in tool '{name}'")
        return format_error(name, exc, {"arguments": args})


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server — FIX #10: hardened against missing mcp.types.Icon
# ─────────────────────────────────────────────────────────────────────────────


def create_server() -> Server:
    _get_dispatch_table()

    server_kwargs: dict[str, Any] = {
        "instructions": (
            "Heuris-BioMCP — Connect ChatGPT, Claude, and other MCP clients "
            "to a curated strategic surface of about 30 life-science tools spanning literature, "
            "genomics, proteomics, clinical data, CRISPR, drug safety, variant interpretation, "
            "and translational workflow design."
        ),
        "website_url": _server_website_url(),
    }

    # Only add website_url if mcp SDK supports it (≥1.3.0)
    try:
        from mcp.types import Icon

        server_kwargs["icons"] = [
            Icon(src=_server_icon_url(), mimeType="image/jpeg", sizes=["512x512"])
        ]
    except ImportError:
        pass

    try:
        server = Server(SERVER_NAME, version=__version__, **server_kwargs)
    except TypeError:
        server_kwargs.pop("icons", None)
        server_kwargs.pop("website_url", None)
        try:
            server = Server(SERVER_NAME, version=__version__, **server_kwargs)
        except TypeError:
            server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    if hasattr(server, "list_resources") and hasattr(server, "read_resource"):
        @server.list_resources()
        async def list_resources() -> list[Resource]:
            return _list_resource_definitions()

        @server.read_resource()
        async def read_resource(uri: Any) -> list[ReadResourceContents]:
            return _read_resource_contents(uri)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        text = await _dispatch(name, arguments or {})
        return [TextContent(type="text", text=text)]

    return server


class _StreamableHTTPASGIApp:
    def __init__(self, session_manager: Any) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.session_manager.handle_request(scope, receive, send)


async def _run() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("BIOMCP_LOG_LEVEL", "INFO"),
        format=("<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"),
        colorize=True,
    )

    n_tools = len(TOOLS)
    logger.info(f"🧬 {SERVER_DISPLAY_NAME} v{__version__} starting — {n_tools} tools registered")
    logger.info(f"   NCBI key : {'✓' if os.getenv('NCBI_API_KEY') else '✗ (3 req/s)'}")
    logger.info(f"   Boltz-2  : {'✓' if os.getenv('NVIDIA_BOLTZ2_API_KEY') else '✗'}")
    logger.info(f"   Evo2     : {'✓' if os.getenv('NVIDIA_EVO2_API_KEY') else '✗'}")
    logger.info(f"   BioGRID  : {'✓' if os.getenv('BIOGRID_API_KEY') else '✗'}")

    server = create_server()
    transport_mode = os.getenv("BIOMCP_TRANSPORT", "stdio")
    http_port      = int(os.getenv("BIOMCP_HTTP_PORT", "8080"))

    if transport_mode == "http":
        logger.info(f"   🌐 HTTP mode — port {http_port}")
        from mcp.server.sse import SseServerTransport
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import FileResponse, JSONResponse, Response
        from starlette.routing import Mount, Route

        sse_transport = SseServerTransport(MESSAGE_PATH)
        streamable_http_manager = StreamableHTTPSessionManager(app=server)
        streamable_http_app = _StreamableHTTPASGIApp(streamable_http_manager)

        async def handle_sse(request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())
            return Response()

        async def handle_health(request):
            return JSONResponse(_build_health_report(transport_mode="http"))

        async def handle_readiness(request):
            payload = _build_readiness_report(transport_mode="http")
            return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

        async def handle_tool_health(request):
            return JSONResponse(_build_tool_health_report())

        async def handle_root(request):
            return JSONResponse(_build_root_report(transport_mode="http"))

        async def handle_streamable_http_head(request):
            return JSONResponse(
                {
                    "service": SERVER_NAME,
                    "status": "ok",
                    "transport": "streamable_http",
                    "endpoint": STREAMABLE_HTTP_PATH,
                }
            )

        async def handle_logo(request):
            logo_path = _resolve_logo_path()
            if logo_path is None:
                return Response(status_code=404)
            return FileResponse(logo_path, media_type="image/jpeg")

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette):
            async with streamable_http_manager.run():
                try:
                    yield
                finally:
                    await close_http_client()

        app = Starlette(
            lifespan=lifespan,
            routes=[
                Route(LOGO_ROUTE_PATH, endpoint=handle_logo, methods=["GET"]),
                Route("/health", endpoint=handle_health, methods=["GET"]),
                Route("/healthz", endpoint=handle_health, methods=["GET"]),
                Route("/readyz", endpoint=handle_readiness, methods=["GET"]),
                Route("/tool-health", endpoint=handle_tool_health, methods=["GET"]),
                Route(SSE_PATH, endpoint=handle_sse, methods=["GET"]),
                Route(STREAMABLE_HTTP_PATH, endpoint=handle_streamable_http_head, methods=["HEAD"]),
                Route(STREAMABLE_HTTP_PATH, endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
                Mount(MESSAGE_PATH, app=sse_transport.handle_post_message),
                Route("/", endpoint=handle_root, methods=["GET"]),
            ],
        )
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])
        import uvicorn
        await uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=http_port)).serve()
    else:
        logger.info("   📟 STDIO mode")
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream,
                    server.create_initialization_options(),
                )
        finally:
            await close_http_client()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("BioMCP interrupted.")
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
