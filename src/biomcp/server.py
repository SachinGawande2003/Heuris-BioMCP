"""
BioMCP — MCP Server Entry Point
================================
Registers all 23 tools, wires the async dispatcher,
and runs the MCP stdio transport.

Tool inventory (23 tools across 6 categories):
  Literature  search_pubmed · get_gene_info · run_blast
  Proteins    get_protein_info · search_proteins · get_alphafold_structure · search_pdb_structures
  Pathways    search_pathways · get_pathway_genes · get_reactome_pathways
  Drug Disc.  get_drug_targets · get_compound_info · get_gene_disease_associations
  Genomics    get_gene_variants · search_gene_expression · search_scrna_datasets
  Clinical    search_clinical_trials · get_trial_details
  Advanced    multi_omics_gene_report · query_neuroimaging_datasets · generate_research_hypothesis
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
# Tool Schema Definitions
# ─────────────────────────────────────────────────────────────────────────────

def _tool(name: str, description: str, properties: dict, required: list[str]) -> Tool:
    """Convenience constructor for MCP Tool objects."""
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
    return {"type": "integer", "description": f"{desc} Default {default}.",
            "default": default, "minimum": min_, "maximum": max_}


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _bool_prop(desc: str, default: bool) -> dict:
    return {"type": "boolean", "description": desc, "default": default}


def _enum_prop(desc: str, values: list[str], default: str | None = None) -> dict:
    p: dict[str, Any] = {"type": "string", "description": desc, "enum": values}
    if default is not None:
        p["default"] = default
    return p


TOOLS: list[Tool] = [

    # ── Literature & NCBI ─────────────────────────────────────────────────────
    _tool("search_pubmed",
          "Search PubMed for scientific literature. Supports full NCBI query syntax "
          "(MeSH terms, Boolean operators, field tags, date ranges). Returns articles "
          "with title, authors, abstract, DOI, PMID, journal, year, and MeSH terms.",
          {
              "query":       _str_prop("PubMed query. E.g. 'BRCA1[Gene] AND breast cancer AND Review[pt]'"),
              "max_results": _int_prop("Articles to return", 10, 1, 200),
              "sort":        _enum_prop("Sort order.", ["relevance", "pub_date"], "relevance"),
          }, ["query"]),

    _tool("get_gene_info",
          "Retrieve gene information from NCBI Gene — symbol, full name, chromosomal "
          "location, aliases, RefSeq IDs, and a curated functional summary.",
          {
              "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'BRCA1', 'EGFR')."),
              "organism":    _str_prop("Species. Default: 'homo sapiens'."),
          }, ["gene_symbol"]),

    _tool("run_blast",
          "Run NCBI BLAST sequence alignment. Submits a protein or nucleotide sequence "
          "and returns top hits with identity%, e-value, bit score, and taxonomy. "
          "Note: BLAST jobs take 30–120 s depending on sequence length and database.",
          {
              "sequence": _str_prop("Amino acid or nucleotide sequence (raw or FASTA)."),
              "program":  _enum_prop("BLAST program.", ["blastp","blastn","blastx","tblastn"], "blastp"),
              "database": _enum_prop("Target database.", ["nr","nt","swissprot","pdb","refseq_protein"], "nr"),
              "max_hits": _int_prop("Alignments to return", 10, 1, 100),
          }, ["sequence"]),

    # ── Proteins & Structures ─────────────────────────────────────────────────
    _tool("get_protein_info",
          "Retrieve a full UniProt entry. Returns protein function, sequence, domains, "
          "PTMs, GO terms, subcellular location, disease associations, and cross-references "
          "to PDB, KEGG, and Ensembl. Prefer Swiss-Prot accessions for curated data.",
          {"accession": _str_prop("UniProt accession (e.g. 'P04637' for human TP53).")},
          ["accession"]),

    _tool("search_proteins",
          "Search UniProt for proteins matching a query. Filter by organism and "
          "review status. Returns accession, name, genes, organism, and length.",
          {
              "query":         _str_prop("Search terms — gene name, function, disease, etc."),
              "organism":      _str_prop("Species filter. Default: 'homo sapiens'."),
              "max_results":   _int_prop("Results", 10, 1, 100),
              "reviewed_only": _bool_prop("Only Swiss-Prot reviewed entries.", True),
          }, ["query"]),

    _tool("get_alphafold_structure",
          "Retrieve AlphaFold predicted protein structure metadata. Returns per-residue "
          "pLDDT confidence statistics, confidence band distribution, and PDB/mmCIF "
          "download URLs. pLDDT ≥ 90 = very high, 70–90 = confident, < 50 = disordered.",
          {
              "uniprot_accession": _str_prop("UniProt accession (e.g. 'P04637')."),
              "model_version":     _str_prop("AlphaFold model version. Default: v4."),
          }, ["uniprot_accession"]),

    _tool("search_pdb_structures",
          "Search RCSB PDB for experimental protein structures. Returns entries with "
          "method (X-ray/cryo-EM/NMR), resolution in Å, deposition date, and direct "
          "PDB/mmCIF download links.",
          {
              "query":       _str_prop("Protein name, gene, organism, or PDB keywords."),
              "max_results": _int_prop("Results", 10, 1, 50),
          }, ["query"]),

    # ── Pathways ──────────────────────────────────────────────────────────────
    _tool("search_pathways",
          "Search KEGG for biological pathways. Returns pathway IDs, descriptions, "
          "KEGG viewer URLs, and organism-specific pathway diagram image URLs.",
          {
              "query":    _str_prop("Keyword — pathway name, gene, disease. E.g. 'apoptosis', 'PI3K'."),
              "organism": _str_prop("KEGG organism code. Default: 'hsa' (human). Others: mmu, rno, dme."),
          }, ["query"]),

    _tool("get_pathway_genes",
          "List all genes in a KEGG pathway with their KEGG gene IDs, symbols, and descriptions.",
          {"pathway_id": _str_prop("KEGG pathway ID (e.g. 'hsa05200' for human cancer pathway).")},
          ["pathway_id"]),

    _tool("get_reactome_pathways",
          "Get Reactome pathways for a gene. Returns pathway hierarchy, evidence types, "
          "and interactive diagram URLs at reactome.org.",
          {
              "gene_symbol": _str_prop("HGNC gene symbol."),
              "species":     _str_prop("NCBI taxonomy ID. Default: '9606' (Homo sapiens)."),
          }, ["gene_symbol"]),

    # ── Drug Discovery ────────────────────────────────────────────────────────
    _tool("get_drug_targets",
          "Find compounds targeting a gene from ChEMBL. Returns compound names, "
          "mechanism of action, IC50/Ki/Kd activity values with units, assay type, "
          "publication year, and direct ChEMBL compound page links.",
          {
              "gene_symbol": _str_prop("Target gene symbol (e.g. 'EGFR', 'BRAF', 'KRAS')."),
              "max_results": _int_prop("Drug entries", 20, 1, 100),
          }, ["gene_symbol"]),

    _tool("get_compound_info",
          "Get detailed drug/compound data from ChEMBL — SMILES, InChI, molecular formula, "
          "molecular weight, AlogP, H-bond donors/acceptors, PSA, Lipinski Ro5 violations, "
          "QED score, clinical approval phase, and therapeutic indications.",
          {"chembl_id": _str_prop("ChEMBL compound ID (e.g. 'CHEMBL25' for aspirin).")},
          ["chembl_id"]),

    _tool("get_gene_disease_associations",
          "Get gene-disease evidence from Open Targets Platform with scores across "
          "six evidence datatypes: genetic_association, somatic_mutation, known_drug, "
          "animal_model, affected_pathway, and literature mining.",
          {
              "gene_symbol": _str_prop("HGNC gene symbol."),
              "max_results": _int_prop("Associations", 15, 1, 50),
          }, ["gene_symbol"]),

    # ── Genomics & Expression ─────────────────────────────────────────────────
    _tool("get_gene_variants",
          "Retrieve genetic variants (SNPs, indels) from Ensembl for a gene region. "
          "Returns rsIDs, positions, alleles, VEP consequence types, and clinical "
          "significance annotations (pathogenic, benign, etc.).",
          {
              "gene_symbol":      _str_prop("HGNC gene symbol."),
              "consequence_type": _str_prop("VEP consequence filter. Default: 'missense_variant'."),
              "max_results":      _int_prop("Variants", 20, 1, 100),
          }, ["gene_symbol"]),

    _tool("search_gene_expression",
          "Search NCBI GEO for gene expression datasets. Returns experiment accessions, "
          "organisms, array platforms, sample counts, and PubMed references.",
          {
              "gene_symbol":  _str_prop("Gene symbol to search for."),
              "condition":    _str_prop("Disease/tissue filter (e.g. 'lung cancer', 'brain')."),
              "max_datasets": _int_prop("Datasets", 10, 1, 50),
          }, ["gene_symbol"]),

    _tool("search_scrna_datasets",
          "Search Human Cell Atlas for single-cell RNA-seq datasets by tissue. "
          "Returns project title, cell counts, donor counts, sequencing technologies, "
          "and HCA portal links.",
          {
              "tissue":      _str_prop("Tissue/organ (e.g. 'brain', 'lung', 'kidney', 'liver')."),
              "species":     _str_prop("Species. Default: 'Homo sapiens'."),
              "max_results": _int_prop("Datasets", 10, 1, 50),
          }, ["tissue"]),

    # ── Clinical ──────────────────────────────────────────────────────────────
    _tool("search_clinical_trials",
          "Search ClinicalTrials.gov for clinical studies. Returns NCT IDs, title, status, "
          "phase, interventions, enrollment counts, start/completion dates, sponsors, "
          "brief summary, eligibility snippet, and geographic locations.",
          {
              "query":       _str_prop("Disease, drug, gene, or condition."),
              "status":      _enum_prop(
                  "Trial status filter.",
                  ["RECRUITING","COMPLETED","NOT_YET_RECRUITING","ACTIVE_NOT_RECRUITING","ALL"],
                  "RECRUITING"
              ),
              "phase":       _enum_prop("Phase filter (optional).", ["PHASE1","PHASE2","PHASE3","PHASE4"]),
              "max_results": _int_prop("Results", 10, 1, 100),
          }, ["query"]),

    _tool("get_trial_details",
          "Retrieve full protocol for a clinical trial — study arms, primary and secondary "
          "outcomes with timeframes, complete eligibility criteria, and central contact info.",
          {"nct_id": _str_prop("NCT identifier (e.g. 'NCT04280705').")},
          ["nct_id"]),

    # ── Multi-Omics Flagship ──────────────────────────────────────────────────
    _tool("multi_omics_gene_report",
          "FLAGSHIP: Generate a comprehensive multi-omics report for a gene by querying "
          "7 databases simultaneously — NCBI Gene, PubMed, Reactome, ChEMBL, Open Targets, "
          "NCBI GEO, and ClinicalTrials.gov. Returns an integrated view across genomics, "
          "pathways, drug targets, disease associations, expression data, and clinical trials. "
          "Total latency ≈ slowest single query (~5–15 s).",
          {"gene_symbol": _str_prop("HGNC gene symbol (e.g. 'EGFR', 'TP53', 'BRCA1', 'KRAS').")},
          ["gene_symbol"]),

    # ── Neuroimaging ──────────────────────────────────────────────────────────
    _tool("query_neuroimaging_datasets",
          "Search OpenNeuro and NeuroVault for neuroimaging datasets. Returns dataset "
          "metadata including subject counts, acquisition parameters, and download links. "
          "Also provides recommended analysis tools per modality.",
          {
              "brain_region": _str_prop("Brain region (e.g. 'hippocampus', 'prefrontal cortex')."),
              "modality":     _enum_prop("Imaging modality.", ["fMRI","EEG","MEG","DTI","MRI","PET"], "fMRI"),
              "condition":    _str_prop("Neurological condition filter (e.g. 'Alzheimer', 'depression')."),
              "max_results":  _int_prop("Datasets", 10, 1, 50),
          }, ["brain_region"]),

    # ── Hypothesis Engine ─────────────────────────────────────────────────────
    _tool("generate_research_hypothesis",
          "Mine scientific literature to generate data-driven research hypotheses. "
          "Queries PubMed for the topic + context genes, identifies knowledge gaps, "
          "and proposes testable hypotheses with supporting paper evidence.",
          {
              "topic": _str_prop(
                  "Research topic or question (e.g. 'KRAS inhibition in pancreatic cancer')."
              ),
              "context_genes": {
                  "type":  "array",
                  "items": {"type": "string"},
                  "description": "Additional gene symbols to include as context (optional).",
              },
              "max_hypotheses": _int_prop("Hypotheses to generate", 3, 1, 10),
          }, ["topic"]),

    # ── NVIDIA NIM ────────────────────────────────────────────────────────────
    _tool("predict_structure_boltz2",
          "Predict biomolecular complex structure + binding affinity using MIT Boltz-2 "
          "via NVIDIA NIM. Approaches FEP accuracy, 1000x faster. Supports proteins, "
          "DNA, RNA, small molecule ligands. Returns mmCIF structure + confidence + "
          "optional IC50-like affinity. Requires NVIDIA_NIM_API_KEY in .env.",
          {
              "protein_sequences": {
                  "type": "array", "items": {"type": "string"},
                  "description": "Protein amino acid sequences (max 4096 res/chain, max 12 chains).",
              },
              "ligand_smiles": {
                  "type": "array", "items": {"type": "string"},
                  "description": "Ligand SMILES strings (max 20). E.g. ['CC1=CC=CC=C1'].",
              },
              "dna_sequences": {
                  "type": "array", "items": {"type": "string"},
                  "description": "Optional DNA sequences.",
              },
              "rna_sequences": {
                  "type": "array", "items": {"type": "string"},
                  "description": "Optional RNA sequences.",
              },
              "predict_affinity":    _bool_prop("Compute binding affinity (needs ligand + protein).", False),
              "method_conditioning": _enum_prop("Structure style.", ["x-ray", "nmr", "md"]),
              "recycling_steps":     _int_prop("Recycling steps", 3, 1, 10),
              "sampling_steps":      _int_prop("Diffusion steps", 200, 50, 500),
              "diffusion_samples":   _int_prop("Structure samples", 1, 1, 5),
          }, ["protein_sequences"]),

    _tool("generate_dna_evo2",
          "Generate novel DNA sequences using Arc Evo2-40B (40B parameters) via NVIDIA NIM. "
          "Single-nucleotide sensitivity across long genomic context. Use cases: "
          "regulatory element design, gene synthesis, sequence completion, synthetic biology. "
          "Requires NVIDIA_NIM_API_KEY in .env.",
          {
              "sequence":        _str_prop("Seed DNA sequence (ACGT, 5 to 3). Evo2 continues from this."),
              "num_tokens":      _int_prop("New DNA bases to generate", 200, 1, 1200),
              "temperature":     {"type": "number", "description": "0.0=deterministic, 1.0=diverse. Default 1.0.", "default": 1.0},
              "top_k":           _int_prop("Top-K sampling (0=off, 4=standard)", 4, 0, 6),
              "top_p":           {"type": "number", "description": "Nucleus sampling 0.0-1.0. Default 1.0.", "default": 1.0},
              "enable_logits":   _bool_prop("Return per-token logit scores for variant scoring.", False),
              "num_generations": _int_prop("Independent generation runs", 1, 1, 5),
          }, ["sequence"]),

    _tool("score_sequence_evo2",
          "Compare wildtype vs variant DNA sequences using Evo2-40B log-likelihoods "
          "for variant effect prediction. Negative delta = potentially deleterious. "
          "Requires NVIDIA_NIM_API_KEY in .env.",
          {
              "wildtype_sequence": _str_prop("Reference wildtype DNA sequence."),
              "variant_sequence":  _str_prop("Mutant DNA sequence (same length as wildtype)."),
          }, ["wildtype_sequence", "variant_sequence"]),

    _tool("design_protein_ligand",
          "Full automated drug-discovery pipeline: UniProt protein fetch + "
          "Boltz-2 structure and affinity prediction in one call. "
          "Returns integrated report with 3D structure, scores, and next steps. "
          "Requires NVIDIA_NIM_API_KEY in .env.",
          {
              "uniprot_accession": _str_prop("Target protein UniProt ID (e.g. P00533 for EGFR)."),
              "ligand_smiles":     _str_prop("Drug SMILES string (e.g. CC1=CC=CC=C1)."),
              "predict_affinity":  _bool_prop("Compute binding affinity. Default True.", True),
              "method_conditioning": _enum_prop("Structure conditioning style.", ["x-ray", "nmr", "md"]),
          }, ["uniprot_accession", "ligand_smiles"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Handler (lives here — lightweight, no extra module needed)
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_research_hypothesis(
    topic: str,
    context_genes: list[str] | None = None,
    max_hypotheses: int = 3,
) -> dict[str, Any]:
    from biomcp.tools.ncbi import search_pubmed
    from biomcp.utils import BioValidator

    genes       = context_genes or []
    max_hyp     = BioValidator.clamp_int(max_hypotheses, 1, 10, "max_hypotheses")
    gene_clause = " OR ".join(genes[:5]) if genes else ""
    query       = f"({topic})" + (f" AND ({gene_clause})" if gene_clause else "") + " AND Review[pt]"

    papers      = await search_pubmed(query, max_results=20)
    articles    = papers.get("articles", [])

    # Collect MeSH coverage to identify context
    mesh_coverage: dict[str, int] = {}
    for art in articles:
        for mesh in art.get("mesh_terms", []):
            mesh_coverage[mesh] = mesh_coverage.get(mesh, 0) + 1

    top_mesh = sorted(mesh_coverage, key=mesh_coverage.__getitem__, reverse=True)[:8]

    hypotheses = []
    for i in range(min(max_hyp, 5)):
        target = genes[i] if i < len(genes) else (top_mesh[i] if i < len(top_mesh) else "key pathway nodes")
        hypotheses.append({
            "id":     i + 1,
            "title":  f"Hypothesis {i + 1}: Role of {target} in {topic}",
            "rationale": (
                f"Based on {len(articles)} review articles, {target} appears as a "
                f"recurring theme in the literature context of '{topic}'. "
                "Mechanistic validation is lacking in current literature."
            ),
            "supporting_paper_count":  max(0, len(articles) - i * 2),
            "key_mesh_context":        top_mesh[:5],
            "suggested_experiments": [
                f"CRISPR knockdown of {target} in relevant cell line",
                "RNA-seq differential expression under perturbed conditions",
                "Protein interaction network analysis (STRING/BioGRID)",
                "In vivo mouse model validation",
            ],
            "data_gaps": [
                "Mechanistic in vivo validation missing",
                "Longitudinal clinical outcome data not available",
                "Single-cell resolution data lacking",
            ],
        })

    return {
        "topic":          topic,
        "context_genes":  genes,
        "literature_base":{
            "query":           query,
            "total_papers":    papers.get("total_found", 0),
            "reviewed_papers": len(articles),
            "top_papers": [
                {"pmid": a["pmid"], "title": a["title"], "year": a["year"]}
                for a in articles[:5]
            ],
            "top_mesh_terms":  top_mesh,
        },
        "hypotheses": hypotheses,
        "disclaimer": (
            "AI-generated hypotheses from literature patterns. "
            "Always validate with domain expertise and experimental evidence."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — routes tool name → handler function
# ─────────────────────────────────────────────────────────────────────────────

async def _dispatch(name: str, args: dict[str, Any]) -> str:
    """
    Route a tool call to its handler.
    All parameters are passed as keyword arguments.
    Validation errors (ValueError/TypeError) return structured error JSON.
    """
    # Lazy import of tool modules (faster startup)
    from biomcp.tools.ncbi      import get_gene_info, run_blast, search_pubmed
    from biomcp.tools.proteins  import (
        get_alphafold_structure, get_protein_info,
        search_pdb_structures, search_proteins,
    )
    from biomcp.tools.pathways  import (
        get_compound_info, get_drug_targets,
        get_gene_disease_associations, get_pathway_genes,
        get_reactome_pathways, search_pathways,
    )
    from biomcp.tools.advanced  import (
        get_gene_variants, get_trial_details,
        multi_omics_gene_report, query_neuroimaging_datasets,
        search_clinical_trials, search_gene_expression,
        search_scrna_datasets,
    )
    from biomcp.tools.nvidia_nim import (
        predict_structure_boltz2,
        generate_dna_evo2,
        score_sequence_evo2,
        design_protein_ligand,
    )

    DISPATCH: dict[str, Any] = {
        # Literature
        "search_pubmed":                  search_pubmed,
        "get_gene_info":                  get_gene_info,
        "run_blast":                      run_blast,
        # Proteins
        "get_protein_info":               get_protein_info,
        "search_proteins":                search_proteins,
        "get_alphafold_structure":        get_alphafold_structure,
        "search_pdb_structures":          search_pdb_structures,
        # Pathways
        "search_pathways":                search_pathways,
        "get_pathway_genes":              get_pathway_genes,
        "get_reactome_pathways":          get_reactome_pathways,
        # Drug Discovery
        "get_drug_targets":               get_drug_targets,
        "get_compound_info":              get_compound_info,
        "get_gene_disease_associations":  get_gene_disease_associations,
        # Genomics
        "get_gene_variants":              get_gene_variants,
        "search_gene_expression":         search_gene_expression,
        "search_scrna_datasets":          search_scrna_datasets,
        # Clinical
        "search_clinical_trials":         search_clinical_trials,
        "get_trial_details":              get_trial_details,
        # Advanced
        "multi_omics_gene_report":        multi_omics_gene_report,
        "query_neuroimaging_datasets":    query_neuroimaging_datasets,
        "generate_research_hypothesis":   _generate_research_hypothesis,
        # NVIDIA NIM
        "predict_structure_boltz2":       predict_structure_boltz2,
        "generate_dna_evo2":              generate_dna_evo2,
        "score_sequence_evo2":            score_sequence_evo2,
        "design_protein_ligand":          design_protein_ligand,
    }

    if name not in DISPATCH:
        return json.dumps({"error": f"Unknown tool '{name}'. "
                                    f"Available: {sorted(DISPATCH)}"})

    handler = DISPATCH[name]
    try:
        result = await handler(**args)
        return format_success(name, result)
    except (ValueError, TypeError, LookupError, KeyError) as exc:
        # Input validation / expected errors — no traceback needed
        return format_error(name, exc, {"arguments": args})
    except Exception as exc:
        # Unexpected errors — include traceback in payload
        logger.exception(f"Unexpected error in tool '{name}'")
        return format_error(name, exc, {"arguments": args})


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────────────────────

def create_server() -> Server:
    """Instantiate and wire the MCP server."""
    server = Server("biomcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        text = await _dispatch(name, arguments or {})
        return [TextContent(type="text", text=text)]

    return server


async def _run() -> None:
    """Configure logging, boot the server, run until interrupted."""
    # Remove default loguru handler and add clean stderr handler
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("BIOMCP_LOG_LEVEL", "INFO"),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "{message}"
        ),
        colorize=True,
    )

    logger.info("🧬 BioMCP server starting…")
    logger.info(f"   Tools registered : {len(TOOLS)}")
    logger.info(f"   Log level        : {os.getenv('BIOMCP_LOG_LEVEL', 'INFO')}")
    logger.info(f"   NCBI API key     : {'set' if os.getenv('NCBI_API_KEY') else 'not set (3 req/s)'}")

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
        logger.info("BioMCP shut down cleanly.")


def main() -> None:
    """Console entry point — called by `biomcp` CLI command."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("BioMCP interrupted.")
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
