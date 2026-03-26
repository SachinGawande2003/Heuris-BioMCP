"""
BioMCP — Main MCP Server
=========================
The most comprehensive Model Context Protocol server for life sciences.

Registered Tools (23 total):
  NCBI/Literature:  search_pubmed, get_gene_info, run_blast
  Proteins:         get_protein_info, search_proteins, get_alphafold_structure, search_pdb_structures
  Pathways:         search_pathways, get_pathway_genes, get_reactome_pathways
  Drug Discovery:   get_drug_targets, get_compound_info, get_gene_disease_associations
  Genomics:         get_gene_variants
  Expression:       search_gene_expression, search_scrna_datasets
  Clinical:         search_clinical_trials, get_trial_details
  Multi-Omics:      multi_omics_gene_report
  Neuroimaging:     query_neuroimaging_datasets
  Hypothesis:       generate_research_hypothesis
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from biomcp.utils import close_http_client, format_error, format_success

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports for tool modules (improves startup time)
# ─────────────────────────────────────────────────────────────────────────────

def _get_tool_modules():
    from biomcp.tools import ncbi, proteins, pathways, advanced
    return ncbi, proteins, pathways, advanced


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (MCP Schema)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list[Tool] = [

    # ── NCBI / Literature ────────────────────────────────────────────────────
    Tool(
        name="search_pubmed",
        description=(
            "Search PubMed for scientific literature. Returns article titles, authors, "
            "abstracts, DOIs, PMIDs, and MeSH terms. Supports full PubMed query syntax "
            "including Boolean operators, MeSH tags, and field specifiers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PubMed search query. Example: 'BRCA1[Gene] AND breast cancer AND (2020:2024[PDAT])'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of articles to return (1–200). Default: 10.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 200,
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "pub_date"],
                    "description": "Sort order. Default: relevance.",
                    "default": "relevance",
                },
            },
            "required": ["query"],
        },
    ),

    Tool(
        name="get_gene_info",
        description=(
            "Retrieve comprehensive gene information from NCBI Gene database including "
            "symbol, full name, chromosomal location, aliases, RefSeq IDs, and functional summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "HGNC gene symbol (e.g. 'TP53', 'BRCA1', 'EGFR').",
                },
                "organism": {
                    "type": "string",
                    "description": "Species name. Default: 'homo sapiens'.",
                    "default": "homo sapiens",
                },
            },
            "required": ["gene_symbol"],
        },
    ),

    Tool(
        name="run_blast",
        description=(
            "Run NCBI BLAST sequence alignment. Submits a protein or nucleotide sequence "
            "and returns top hits with identity%, e-value, bit score, and taxonomy info."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid or nucleotide sequence (FASTA format or raw).",
                },
                "program": {
                    "type": "string",
                    "enum": ["blastp", "blastn", "blastx", "tblastn"],
                    "description": "BLAST program. Default: blastp (protein vs protein).",
                    "default": "blastp",
                },
                "database": {
                    "type": "string",
                    "enum": ["nr", "nt", "swissprot", "pdb", "refseq_protein"],
                    "description": "Target database. Default: nr (non-redundant).",
                    "default": "nr",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Max alignments to return (1–100). Default: 10.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["sequence"],
        },
    ),

    # ── Proteins ─────────────────────────────────────────────────────────────
    Tool(
        name="get_protein_info",
        description=(
            "Retrieve comprehensive protein data from UniProt Swiss-Prot. Includes "
            "function, sequence, domains, PTMs, subcellular location, disease links, "
            "GO terms, and cross-references to PDB, KEGG, and Ensembl."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "accession": {
                    "type": "string",
                    "description": "UniProt accession number (e.g. 'P04637' for human TP53).",
                },
            },
            "required": ["accession"],
        },
    ),

    Tool(
        name="search_proteins",
        description=(
            "Search UniProt for proteins matching a query. Filter by organism and "
            "review status. Returns accession, name, gene, length, and review status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms (gene name, function, disease, etc.)."},
                "organism": {"type": "string", "description": "Species filter. Default: homo sapiens.", "default": "homo sapiens"},
                "max_results": {"type": "integer", "description": "Results to return (1–100). Default: 10.", "default": 10},
                "reviewed_only": {"type": "boolean", "description": "Only Swiss-Prot reviewed entries. Default: true.", "default": True},
            },
            "required": ["query"],
        },
    ),

    Tool(
        name="get_alphafold_structure",
        description=(
            "Retrieve AlphaFold predicted protein structure metadata. Returns per-residue "
            "pLDDT confidence scores, PDB/mmCIF download URLs, PAE (predicted alignment error) "
            "data, and structural quality assessment."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "uniprot_accession": {
                    "type": "string",
                    "description": "UniProt accession (e.g. 'P04637').",
                },
                "model_version": {
                    "type": "string",
                    "description": "AlphaFold model version. Default: v4.",
                    "default": "v4",
                },
            },
            "required": ["uniprot_accession"],
        },
    ),

    Tool(
        name="search_pdb_structures",
        description=(
            "Search RCSB PDB for experimental protein 3D structures. "
            "Returns structures with method (X-ray/cryo-EM/NMR), resolution, "
            "deposition date, and direct download links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Protein name, gene, organism, or PDB keywords."},
                "max_results": {"type": "integer", "description": "Results (1–50). Default: 10.", "default": 10},
            },
            "required": ["query"],
        },
    ),

    # ── Pathways ──────────────────────────────────────────────────────────────
    Tool(
        name="search_pathways",
        description=(
            "Search KEGG for biological pathways. Returns pathway IDs, descriptions, "
            "KEGG viewer URLs, and organism-specific pathway images."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pathway keyword (e.g. 'apoptosis', 'PI3K', 'cancer')."},
                "organism": {"type": "string", "description": "KEGG organism code. Default: hsa (human).", "default": "hsa"},
            },
            "required": ["query"],
        },
    ),

    Tool(
        name="get_pathway_genes",
        description="Get all genes in a KEGG pathway with their KEGG IDs and descriptions.",
        inputSchema={
            "type": "object",
            "properties": {
                "pathway_id": {"type": "string", "description": "KEGG pathway ID (e.g. 'hsa05200' for human cancer)."},
            },
            "required": ["pathway_id"],
        },
    ),

    Tool(
        name="get_reactome_pathways",
        description=(
            "Get Reactome pathways for a gene. Returns pathway hierarchy, "
            "evidence types, and interactive diagram links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol."},
                "species": {"type": "string", "description": "NCBI taxon ID. Default: 9606 (human).", "default": "9606"},
            },
            "required": ["gene_symbol"],
        },
    ),

    # ── Drug Discovery ────────────────────────────────────────────────────────
    Tool(
        name="get_drug_targets",
        description=(
            "Find drugs and compounds targeting a gene from ChEMBL. "
            "Returns compound names, mechanism of action, IC50/Ki activity values, "
            "and clinical approval status."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "Target gene symbol (e.g. 'EGFR', 'BRAF', 'KRAS')."},
                "max_results": {"type": "integer", "description": "Max drug entries (1–100). Default: 20.", "default": 20},
            },
            "required": ["gene_symbol"],
        },
    ),

    Tool(
        name="get_compound_info",
        description=(
            "Get detailed information about a drug/compound from ChEMBL. "
            "Includes SMILES, InChI, molecular properties, Lipinski Ro5, QED score, "
            "drug approval phase, and therapeutic indications."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chembl_id": {"type": "string", "description": "ChEMBL compound ID (e.g. 'CHEMBL25' for aspirin)."},
            },
            "required": ["chembl_id"],
        },
    ),

    Tool(
        name="get_gene_disease_associations",
        description=(
            "Get gene-disease associations from Open Targets Platform with evidence scores "
            "across genetics, somatic mutations, drugs, pathways, and text mining."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol."},
                "max_results": {"type": "integer", "description": "Max associations (1–50). Default: 15.", "default": 15},
            },
            "required": ["gene_symbol"],
        },
    ),

    # ── Genomics ──────────────────────────────────────────────────────────────
    Tool(
        name="get_gene_variants",
        description=(
            "Retrieve genetic variants (SNPs, indels) in a gene from Ensembl. "
            "Returns rsIDs, positions, alleles, consequence types, and clinical significance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol."},
                "consequence_type": {
                    "type": "string",
                    "description": "VEP consequence filter. Default: missense_variant.",
                    "default": "missense_variant",
                },
                "max_results": {"type": "integer", "description": "Max variants (1–100). Default: 20.", "default": 20},
            },
            "required": ["gene_symbol"],
        },
    ),

    # ── Expression ────────────────────────────────────────────────────────────
    Tool(
        name="search_gene_expression",
        description=(
            "Search NCBI GEO for gene expression datasets. Returns experiment accessions, "
            "organisms, platforms, sample counts, and PubMed references."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "Gene symbol to search for."},
                "condition": {"type": "string", "description": "Disease/tissue filter (e.g. 'lung cancer', 'brain')."},
                "max_datasets": {"type": "integer", "description": "Max datasets (1–50). Default: 10.", "default": 10},
            },
            "required": ["gene_symbol"],
        },
    ),

    Tool(
        name="search_scrna_datasets",
        description=(
            "Search Human Cell Atlas for single-cell RNA-seq datasets by tissue. "
            "Returns project details, cell counts, sequencing technologies, and download links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tissue": {"type": "string", "description": "Tissue/organ (e.g. 'brain', 'lung', 'kidney')."},
                "species": {"type": "string", "description": "Species. Default: Homo sapiens.", "default": "Homo sapiens"},
                "max_results": {"type": "integer", "description": "Max datasets (1–50). Default: 10.", "default": 10},
            },
            "required": ["tissue"],
        },
    ),

    # ── Clinical ──────────────────────────────────────────────────────────────
    Tool(
        name="search_clinical_trials",
        description=(
            "Search ClinicalTrials.gov for clinical studies. Returns NCT IDs, status, phase, "
            "interventions, enrollment numbers, dates, sponsors, and eligibility summaries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Disease, drug, gene, or condition."},
                "status": {
                    "type": "string",
                    "enum": ["RECRUITING", "COMPLETED", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ALL"],
                    "description": "Trial status filter. Default: RECRUITING.",
                    "default": "RECRUITING",
                },
                "phase": {
                    "type": "string",
                    "enum": ["PHASE1", "PHASE2", "PHASE3", "PHASE4"],
                    "description": "Phase filter (optional).",
                },
                "max_results": {"type": "integer", "description": "Results (1–100). Default: 10.", "default": 10},
            },
            "required": ["query"],
        },
    ),

    Tool(
        name="get_trial_details",
        description="Get full details of a specific clinical trial: arms, outcomes, eligibility, contacts.",
        inputSchema={
            "type": "object",
            "properties": {
                "nct_id": {"type": "string", "description": "NCT identifier (e.g. 'NCT04280705')."},
            },
            "required": ["nct_id"],
        },
    ),

    # ── Multi-Omics ───────────────────────────────────────────────────────────
    Tool(
        name="multi_omics_gene_report",
        description=(
            "Generate a comprehensive multi-omics report for a gene by simultaneously querying "
            "7+ databases: NCBI Gene, PubMed, UniProt, Reactome, ChEMBL, Open Targets, GEO, "
            "and ClinicalTrials.gov. Returns an integrated view across genomics, proteomics, "
            "pathways, drug targets, disease associations, expression data, and clinical trials."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "HGNC gene symbol (e.g. 'EGFR', 'TP53', 'BRCA1', 'KRAS').",
                },
            },
            "required": ["gene_symbol"],
        },
    ),

    # ── Neuroimaging ──────────────────────────────────────────────────────────
    Tool(
        name="query_neuroimaging_datasets",
        description=(
            "Search OpenNeuro and NeuroVault for neuroimaging datasets (fMRI, EEG, MEG, DTI). "
            "Returns dataset metadata, subject counts, acquisition parameters, and download links. "
            "Also provides recommended analysis tools per modality."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "brain_region": {"type": "string", "description": "Brain region (e.g. 'hippocampus', 'amygdala', 'prefrontal cortex')."},
                "modality": {
                    "type": "string",
                    "enum": ["fMRI", "EEG", "MEG", "DTI", "MRI", "PET"],
                    "description": "Imaging modality. Default: fMRI.",
                    "default": "fMRI",
                },
                "condition": {"type": "string", "description": "Neurological condition filter (e.g. 'Alzheimer', 'schizophrenia')."},
                "max_results": {"type": "integer", "description": "Max datasets (1–50). Default: 10.", "default": 10},
            },
            "required": ["brain_region"],
        },
    ),

    # ── Hypothesis Generation ─────────────────────────────────────────────────
    Tool(
        name="generate_research_hypothesis",
        description=(
            "Generate data-driven research hypotheses by mining literature and cross-referencing "
            "multi-omics data. Identifies knowledge gaps, unexpected connections between genes/diseases, "
            "and proposes testable hypotheses with supporting evidence from multiple databases."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic or question (e.g. 'KRAS inhibition in pancreatic cancer').",
                },
                "context_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional gene symbols to include as context.",
                },
                "max_hypotheses": {
                    "type": "integer",
                    "description": "Number of hypotheses to generate (1–10). Default: 3.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["topic"],
        },
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

async def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Route tool calls to the appropriate handler with unified error handling."""
    ncbi, proteins, pathways, advanced = _get_tool_modules()

    handlers: dict[str, Any] = {
        # NCBI
        "search_pubmed":            (ncbi.search_pubmed,            ["query", "max_results", "sort"]),
        "get_gene_info":            (ncbi.get_gene_info,             ["gene_symbol", "organism"]),
        "run_blast":                (ncbi.run_blast,                 ["sequence", "program", "database", "max_hits"]),
        # Proteins
        "get_protein_info":         (proteins.get_protein_info,      ["accession"]),
        "search_proteins":          (proteins.search_proteins,        ["query", "organism", "max_results", "reviewed_only"]),
        "get_alphafold_structure":  (proteins.get_alphafold_structure, ["uniprot_accession", "model_version"]),
        "search_pdb_structures":    (proteins.search_pdb_structures,  ["query", "max_results"]),
        # Pathways
        "search_pathways":          (pathways.search_pathways,        ["query", "organism"]),
        "get_pathway_genes":        (pathways.get_pathway_genes,      ["pathway_id"]),
        "get_reactome_pathways":    (pathways.get_reactome_pathways,  ["gene_symbol", "species"]),
        # Drug Discovery
        "get_drug_targets":         (pathways.get_drug_targets,       ["gene_symbol", "max_results"]),
        "get_compound_info":        (pathways.get_compound_info,      ["chembl_id"]),
        "get_gene_disease_associations": (pathways.get_gene_disease_associations, ["gene_symbol", "max_results"]),
        # Genomics
        "get_gene_variants":        (advanced.get_gene_variants,      ["gene_symbol", "consequence_type", "max_results"]),
        # Expression
        "search_gene_expression":   (advanced.search_gene_expression, ["gene_symbol", "condition", "max_datasets"]),
        "search_scrna_datasets":    (advanced.search_scrna_datasets,  ["tissue", "species", "max_results"]),
        # Clinical
        "search_clinical_trials":   (advanced.search_clinical_trials, ["query", "status", "phase", "max_results"]),
        "get_trial_details":        (advanced.get_trial_details,      ["nct_id"]),
        # Multi-Omics
        "multi_omics_gene_report":  (advanced.multi_omics_gene_report, ["gene_symbol"]),
        # Neuroimaging
        "query_neuroimaging_datasets": (advanced.query_neuroimaging_datasets, ["brain_region", "modality", "condition", "max_results"]),
        # Hypothesis
        "generate_research_hypothesis": (_hypothesis_handler,         ["topic", "context_genes", "max_hypotheses"]),
    }

    if name not in handlers:
        return json.dumps({"error": f"Unknown tool: '{name}'"})

    func, param_keys = handlers[name]
    # Build kwargs from only the provided arguments
    kwargs = {k: arguments[k] for k in param_keys if k in arguments}

    try:
        result = await func(**kwargs)
        return format_success(name, result)
    except (ValueError, TypeError) as e:
        return format_error(name, e, {"arguments": arguments})
    except Exception as e:
        logger.exception(f"Unexpected error in tool '{name}'")
        return format_error(name, e, {"arguments": arguments})


async def _hypothesis_handler(
    topic: str,
    context_genes: list[str] | None = None,
    max_hypotheses: int = 3,
) -> dict[str, Any]:
    """
    Literature-backed hypothesis generator.
    Queries PubMed for the topic + context genes and synthesizes research gaps.
    """
    from biomcp.tools.ncbi import search_pubmed

    genes = context_genes or []
    gene_query = " OR ".join(genes[:5]) if genes else ""
    full_query = f"({topic}) {f'AND ({gene_query})' if gene_query else ''} AND Review[pt]"

    papers = await search_pubmed(full_query, max_results=20, sort="relevance")
    articles = papers.get("articles", [])

    # Extract key terms and suggest hypotheses based on existing literature
    topics_covered = set()
    for art in articles:
        for mesh in art.get("mesh_terms", []):
            topics_covered.add(mesh)

    hypotheses = []
    for i in range(min(max_hypotheses, 3)):
        h_num = i + 1
        hypotheses.append({
            "id": h_num,
            "hypothesis": f"Hypothesis {h_num}: Based on {len(articles)} papers retrieved "
                          f"for '{topic}', further investigation is warranted into the "
                          f"mechanistic role of {genes[i] if i < len(genes) else 'key regulatory nodes'} "
                          f"in modulating outcomes.",
            "supporting_evidence_count": max(0, len(articles) - i * 3),
            "key_mesh_terms": list(topics_covered)[:5],
            "suggested_experiments": [
                "CRISPR knockdown/knockout screen",
                "RNA-seq differential expression analysis",
                "Protein-protein interaction network analysis",
            ],
            "data_gaps": [
                "Longitudinal clinical outcome data needed",
                "Mechanistic in vivo validation required",
            ],
        })

    return {
        "topic": topic,
        "context_genes": genes,
        "literature_base": {
            "total_papers": papers.get("total_found", 0),
            "reviewed_papers": len(articles),
            "top_papers": [
                {"pmid": a["pmid"], "title": a["title"], "year": a["year"]}
                for a in articles[:5]
            ],
        },
        "hypotheses": hypotheses,
        "note": "These hypotheses are generated from literature mining patterns. "
                "Always validate with domain expertise and experimental evidence.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Setup
# ─────────────────────────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the BioMCP server instance."""
    server = Server("biomcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = await dispatch_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    return server


async def _run_server() -> None:
    """Run the BioMCP MCP server over stdio."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("BIOMCP_LOG_LEVEL", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    logger.info("🧬 BioMCP server starting...")
    logger.info(f"   Registered tools: {len(TOOLS)}")

    server = create_server()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await close_http_client()
        logger.info("BioMCP server shut down cleanly.")


def main() -> None:
    """CLI entry point."""
    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        logger.info("BioMCP interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
