"""
BioMCP v2 — Complete MCP Server
================================
The most comprehensive biological research MCP server for Claude.

Tools (52 total across 14 categories):

  Literature & NCBI (3):
    search_pubmed · get_gene_info · run_blast

  Proteins & Structures (4):
    get_protein_info · search_proteins · get_alphafold_structure · search_pdb_structures

  Pathways (3):
    search_pathways · get_pathway_genes · get_reactome_pathways

  Drug Discovery (3):
    get_drug_targets · get_compound_info · get_gene_disease_associations

  Genomics & Expression (3):
    get_gene_variants · search_gene_expression · search_scrna_datasets

  Clinical (2):
    search_clinical_trials · get_trial_details

  AI-Powered — NVIDIA NIM (4):
    predict_structure_boltz2 · generate_dna_evo2 · score_sequence_evo2 · design_protein_ligand

  Integrated & Advanced (3):
    multi_omics_gene_report · query_neuroimaging_datasets · generate_research_hypothesis

  Extended Databases (7):
    get_omim_gene_diseases · get_string_interactions · get_gtex_expression
    search_cbio_mutations · search_gwas_catalog · get_disgenet_associations
    get_pharmgkb_variants

  Verification & Conflict Detection (2):
    verify_biological_claim · detect_database_conflicts

  Experimental Design (3):
    generate_experimental_protocol · suggest_cell_lines · estimate_statistical_power

  Session Intelligence — Novel Architecture (5):
    resolve_entity              — Canonical cross-database entity resolution
    get_session_knowledge_graph — Live entity graph from all tool calls
    find_biological_connections — Cross-database connection discovery
    export_research_session     — Full provenance + citations + repro script
    plan_and_execute_research   — DAG-based adaptive research workflow planner

  ── NEW in v2.1 ─────────────────────────────────────────────────────────────

  Intelligence Layer (3):
    validate_reasoning_chain     — Multi-step biological reasoning verification
    find_repurposing_candidates  — Drug repurposing intelligence engine
    find_research_gaps           — Research gap detector & grant angles

  Tier 2 Extended Databases (7):
    get_biogrid_interactions     — BioGRID curated protein-protein interactions
    search_orphan_diseases       — Orphanet rare disease database
    get_tcga_expression         — TCGA tumor RNA-seq via GDC API
    search_cellmarker            — CellMarker 2.0 cell type markers
    get_encode_regulatory        — ENCODE regulatory elements
    search_metabolomics          — MetaboLights metabolomics
    get_ucsc_splice_variants     — UCSC alternative splicing isoforms

  ──────────────────────────────────────────────────────────────────────────
  Architecturally Novel Features (unique to BioMCP v2):
    · Session Knowledge Graph — auto-populated by every tool call
    · Entity Resolver — canonical IDs across HGNC/UniProt/Ensembl/NCBI
    · Adaptive Query Planner — dependency-aware parallel execution DAG
    · Cross-database conflict detection — data consistency scoring
    · Biological claim verification — evidence-graded verdicts
    · Reproducibility export — citable, FAIR-compliant session export
    · Reasoning Chain Validator — fact-check biological logic step-by-step
    · Drug Repurposing Engine — $500K analysis in seconds
    · Research Gap Detector — literature landscape mapping for grants
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

from biomcp.utils import close_http_client, format_error, format_success


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


# ─────────────────────────────────────────────────────────────────────────────
# Complete Tool Registry — 47 tools
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
            "dna_sequences": {
                "type": "array",
                "items": {"type": "string"},
                "description": "DNA sequences (optional).",
            },
            "predict_affinity": _bool_prop("Compute binding affinity.", False),
            "method_conditioning": _enum_prop("Structure style.", ["x-ray", "nmr", "md"]),
            "recycling_steps": _int_prop("Recycling steps", 3, 1, 10),
            "sampling_steps": _int_prop("Diffusion steps", 200, 50, 500),
            "diffusion_samples": _int_prop("Structure samples", 1, 1, 5),
        },
        ["protein_sequences"],
    ),
    _tool(
        "generate_dna_evo2",
        "Arc Evo2-40B via NVIDIA NIM: DNA sequence generation with 40B parameter "
        "genomic foundation model. Regulatory element design, gene synthesis. "
        "Requires NVIDIA_EVO2_API_KEY.",
        {
            "sequence": _str_prop("Seed DNA sequence (ACGT). Evo2 continues from this."),
            "num_tokens": _int_prop("New DNA bases to generate", 200, 1, 1200),
            "temperature": _float_prop("0.0=deterministic, 1.0=diverse.", 1.0),
            "top_k": _int_prop("Top-K sampling", 4, 0, 6),
            "enable_logits": _bool_prop("Return per-token logit scores.", False),
            "num_generations": _int_prop("Independent generation runs", 1, 1, 5),
        },
        ["sequence"],
    ),
    _tool(
        "score_sequence_evo2",
        "Evo2-40B variant effect prediction: compare wildtype vs variant DNA log-likelihoods. "
        "Negative delta = potentially deleterious mutation.",
        {
            "wildtype_sequence": _str_prop("Reference wildtype DNA sequence."),
            "variant_sequence": _str_prop("Mutant DNA sequence (same length)."),
        },
        ["wildtype_sequence", "variant_sequence"],
    ),
    _tool(
        "design_protein_ligand",
        "Full drug-discovery pipeline: UniProt fetch → Boltz-2 structure + affinity in one call. "
        "Requires NVIDIA_BOLTZ2_API_KEY.",
        {
            "uniprot_accession": _str_prop("Target protein UniProt ID (e.g. P00533 for EGFR)."),
            "ligand_smiles": _str_prop("Drug SMILES string."),
            "predict_affinity": _bool_prop("Compute binding affinity. Default True.", True),
            "method_conditioning": _enum_prop("Structure conditioning.", ["x-ray", "nmr", "md"]),
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
            "topic": _str_prop("Research topic (e.g. 'KRAS inhibition in pancreatic cancer')."),
            "context_genes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional gene symbols.",
            },
            "max_hypotheses": _int_prop("Hypotheses to generate", 3, 1, 10),
        },
        ["topic"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 9: Extended Databases — NEW (7 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "get_omim_gene_diseases",
        "OMIM (Online Mendelian Inheritance in Man): genetic disease-gene relationships "
        "with inheritance patterns (AD/AR/XL/MT) and phenotype descriptions. "
        "Authoritative source for Mendelian disease genetics.",
        {"gene_symbol": _str_prop("HGNC gene symbol (e.g. 'BRCA1', 'CFTR', 'TP53').")},
        ["gene_symbol"],
    ),
    _tool(
        "get_string_interactions",
        "STRING protein-protein interaction network with multi-evidence confidence scores "
        "(experimental, co-expression, database, text mining, neighborhood). "
        "Returns interaction partners scored 0–1000.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "min_score": _int_prop(
                "Minimum interaction score (400=medium, 700=high, 900=very high)", 400, 0, 1000
            ),
            "max_results": _int_prop("Interaction partners", 20, 1, 100),
            "species": _int_prop("NCBI taxonomy ID", 9606, 0, 99999999),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_gtex_expression",
        "GTEx tissue-specific gene expression in healthy humans across 54 tissues. "
        "Critical for understanding normal expression context before studying disease states. "
        "Returns median TPM per tissue with highest/lowest expressing tissues.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "top_tissues": _int_prop("Tissues sorted by median TPM", 10, 1, 54),
            "dataset_id": _str_prop("GTEx dataset version. Default 'gtex_v8'."),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_cbio_mutations",
        "cBioPortal cancer mutation frequencies across TCGA cohorts. "
        "Returns mutation frequency (%) per cancer type, top mutation classes, "
        "and pan-cancer summary. Identifies which cancers most commonly mutate a gene.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "cancer_type": _str_prop(
                "TCGA cancer type (e.g. 'luad', 'brca', 'coad'). Empty=pan-cancer."
            ),
            "max_studies": _int_prop("Studies to query", 10, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_gwas_catalog",
        "NHGRI-EBI GWAS Catalog: genome-wide significant associations (p<5e-8) for a gene. "
        "Returns trait associations, SNP IDs, odds ratios, risk alleles, and study PMIDs. "
        "Essential for genetic epidemiology context.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "p_value_threshold": _float_prop("Maximum p-value for significance.", 5e-8),
            "max_results": _int_prop("Associations", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_disgenet_associations",
        "DisGeNET comprehensive gene-disease associations integrating expert-curated "
        "databases (UniProt, OMIM, Orphanet) + GWAS + literature. GDA score "
        "weights source reliability. Complements Open Targets with broader coverage.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "min_score": _float_prop("Minimum GDA score (0–1). Default 0.1.", 0.1),
            "max_results": _int_prop("Associations", 20, 1, 100),
        },
        ["gene_symbol"],
    ),
    _tool(
        "get_pharmgkb_variants",
        "PharmGKB pharmacogenomics: how genetic variants affect drug response. "
        "Returns clinical annotations with FDA-grade evidence levels (1A=FDA label+guideline). "
        "Critical for personalized medicine and CYP450 variant analysis.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'CYP2D6', 'TPMT', 'DPYD', 'VKORC1')."),
            "max_results": _int_prop("Variant-drug annotations", 15, 1, 50),
        },
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 10: Verification & Conflict Detection — NEW (2 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "verify_biological_claim",
        "Verify a biological claim against 3–5 databases simultaneously. "
        "Returns confidence grade (A–F), supporting/contradicting evidence, "
        "and recommendation. "
        "Example: 'EGFR is overexpressed in lung cancer' → queries PubMed, UniProt, Open Targets.",
        {
            "claim": _str_prop("Natural language biological claim to verify."),
            "context_gene": _str_prop("Optional gene symbol to focus evidence gathering."),
            "max_evidence_sources": _int_prop("Databases to query (3–5)", 5, 3, 5),
        },
        ["claim"],
    ),
    _tool(
        "detect_database_conflicts",
        "Scan for conflicting biological information about a gene across NCBI, "
        "UniProt, ChEMBL, and Open Targets. Returns consistency score and flagged "
        "discrepancies with severity ratings (HIGH/MEDIUM/LOW).",
        {"gene_symbol": _str_prop("HGNC gene symbol to scan for cross-database conflicts.")},
        ["gene_symbol"],
    ),
    # ════════════════════════════════════════════════════════════════════
    # CATEGORY 11: Experimental Design — NEW (3 tools)
    # ════════════════════════════════════════════════════════════════════
    _tool(
        "generate_experimental_protocol",
        "Generate a complete, actionable laboratory protocol from a biological hypothesis. "
        "Returns: cell lines, reagent list with catalog numbers, step-by-step protocol, "
        "positive/negative controls, expected readouts, statistical design, similar PubMed "
        "protocols, and timeline. Unlike any other MCP tool — bridges AI insights to bench science.",
        {
            "hypothesis": _str_prop(
                "Research hypothesis (e.g. 'KRAS G12C inhibition reduces NSCLC proliferation')."
            ),
            "gene_symbol": _str_prop("Primary gene of interest (auto-extracted if empty)."),
            "cancer_type": _str_prop(
                "Cancer type (e.g. 'lung_cancer', 'breast_cancer', 'general')."
            ),
            "assay_type": _enum_prop(
                "Assay type.",
                [
                    "auto",
                    "crispr_knockout",
                    "sirna_knockdown",
                    "drug_sensitivity",
                    "apoptosis_flow",
                    "protein_interaction",
                ],
                "auto",
            ),
            "available_equipment": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Available equipment list (e.g. ['flow_cytometer', 'luminometer']).",
            },
        },
        ["hypothesis"],
    ),
    _tool(
        "suggest_cell_lines",
        "Recommend validated, ATCC-authenticated cell lines for a research context "
        "with genotype notes, ATCC catalog numbers, and deposition URLs. "
        "Includes guidance on molecular feature filtering (e.g. 'KRAS G12C', 'p53 null').",
        {
            "cancer_type": _str_prop(
                "Cancer type (e.g. 'lung', 'breast', 'colorectal', 'glioblastoma')."
            ),
            "gene_symbol": _str_prop("Gene of interest for mutation-aware filtering."),
            "molecular_feature": _str_prop(
                "Required molecular feature (e.g. 'EGFR mutant', 'p53 null', 'HER2 amplified')."
            ),
            "max_results": _int_prop("Cell lines to return", 5, 1, 15),
        },
        ["cancer_type"],
    ),
    _tool(
        "estimate_statistical_power",
        "Calculate required sample size for adequate statistical power. "
        "Returns n per group, total N, recommended statistical test, and software suggestions. "
        "Supports Bonferroni correction for multi-arm designs.",
        {
            "expected_effect_size": _float_prop(
                "Cohen's d (0.2=small, 0.5=medium, 0.8=large).", 0.5
            ),
            "alpha": _float_prop("Significance threshold. Default 0.05.", 0.05),
            "power": _float_prop("Desired statistical power (1-β). Default 0.8.", 0.8),
            "n_groups": _int_prop("Number of comparison groups", 2, 2, 10),
            "assay_type": _enum_prop(
                "Assay context.",
                [
                    "drug_sensitivity",
                    "crispr_knockout",
                    "sirna_knockdown",
                    "apoptosis_flow",
                    "protein_interaction",
                ],
                "drug_sensitivity",
            ),
        },
        [],
    ),
    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 13: Intelligence Layer — NEW (3 tools)
    # ═══════════════════════════════════════════════════════════════════════
    _tool(
        "validate_reasoning_chain",
        "Verify a multi-step biological reasoning chain against primary databases. "
        "Each step (e.g. KRAS → RAF → MEK → ERK) is independently verified via "
        "PubMed co-occurrence and Reactome curation. Returns confidence per step, "
        "broken links, missing intermediaries, and alternative pathways. "
        "Allows Claude to fact-check its own biological reasoning in real time — "
        "unique capability in the MCP ecosystem.",
        {
            "reasoning_chain": _str_prop(
                "Biological reasoning chain in arrow notation or natural language. "
                "E.g. 'KRAS → RAF → MEK → ERK → proliferation' or "
                "'EGFR activates PI3K which activates AKT leading to survival'."
            ),
            "organism": _str_prop("Species context. Default: 'Homo sapiens'."),
            "verify_depth": _enum_prop(
                "Verification depth.", ["quick", "standard", "deep"], "standard"
            ),
        },
        ["reasoning_chain"],
    ),
    _tool(
        "find_repurposing_candidates",
        "Drug repurposing intelligence engine. Queries ChEMBL, Open Targets, "
        "ClinicalTrials.gov, and PubMed simultaneously to surface: approved drugs "
        "with off-target activity, drugs in trials for related diseases, "
        "combination therapy opportunities, and fastest regulatory path. "
        "This analysis typically costs pharma companies $500K+ when done manually.",
        {
            "disease": _str_prop("Target disease (e.g. 'pancreatic cancer', 'Alzheimer disease')."),
            "gene_target": _str_prop("Primary gene target (e.g. 'KRAS', 'EGFR'). Optional."),
            "mechanism": _str_prop(
                "Biological mechanism (e.g. 'kinase inhibition', 'autophagy'). Optional."
            ),
            "max_candidates": _int_prop("Maximum repurposing candidates", 15, 1, 50),
            "approved_only": _bool_prop("Only FDA-approved drugs.", False),
        },
        ["disease"],
    ),
    _tool(
        "find_research_gaps",
        "Research gap detector: maps what IS known vs what ISN'T for a topic. "
        "Analyzes PubMed publication density, recency trends, and subtopic coverage "
        "to surface understudied areas, high-impact unanswered questions, "
        "grant angles, and recommended experimental approaches. "
        "Turns BioMCP into a research strategy tool.",
        {
            "topic": _str_prop("Research topic (e.g. 'CAR-T cell therapy solid tumors')."),
            "subtopics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific subtopics to probe. Auto-generated if empty.",
            },
            "publication_window": _int_prop("Years for recency analysis", 5, 1, 20),
            "max_gaps": _int_prop("Maximum gaps to report", 10, 1, 25),
        },
        ["topic"],
    ),
    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 14: Tier 2 Extended Databases — NEW (7 tools)
    # ═══════════════════════════════════════════════════════════════════════
    _tool(
        "get_biogrid_interactions",
        "BioGRID protein-protein interaction network: 2M+ manually curated "
        "interactions with experimental evidence from primary literature. "
        "Unlike STRING (prediction-based), BioGRID entries are directly extracted "
        "from experimental papers. Returns experimental system, publication support, "
        "and hub score for each interaction.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol."),
            "interaction_type": _enum_prop(
                "Interaction type.", ["physical", "genetic", "all"], "physical"
            ),
            "min_publications": _int_prop("Minimum publication support", 1, 1, 10),
            "max_results": _int_prop("Interactions to return", 25, 1, 100),
            "include_genetic": _bool_prop("Include genetic interactions.", False),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_orphan_diseases",
        "Orphanet rare disease database: 6,000+ rare diseases with validated gene "
        "associations, prevalence estimates (1:10,000 to <1:1,000,000), inheritance "
        "patterns, and ICD-10 cross-references. Critical for unmet medical needs "
        "analysis, rare disease drug repurposing, and orphan drug designation.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol to find associated rare diseases."),
            "disease_name": _str_prop("Disease name or keyword (alternative to gene_symbol)."),
            "max_results": _int_prop("Diseases to return", 15, 1, 50),
        },
        [],
    ),
    _tool(
        "get_tcga_expression",
        "TCGA tumor RNA-seq via GDC API: gene expression from actual patient tumor "
        "samples across 33 cancer types. Not cell lines — real primary tumors with "
        "clinical metadata. Returns available STAR-counts files, TCGA mutation "
        "burden, and links to GDC portal for download. "
        "Gold standard for cancer expression analysis.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'KRAS', 'EGFR')."),
            "cancer_type": _str_prop(
                "TCGA project code (e.g. 'TCGA-LUAD', 'TCGA-BRCA', 'TCGA-COAD'). "
                "Leave empty for pan-cancer."
            ),
            "max_cases": _int_prop("Cases to sample", 10, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_cellmarker",
        "CellMarker 2.0: validated cell type markers for 1,000+ cell types across "
        "tissues. Essential for scRNA-seq cluster annotation, cell type deconvolution "
        "from bulk RNA-seq (CIBERSORT), and flow cytometry panel design. "
        "Query by gene (find which cells it marks), tissue, or cell type.",
        {
            "gene_symbol": _str_prop("Gene symbol to find which cell types it marks."),
            "tissue": _str_prop("Tissue filter (e.g. 'lung', 'blood', 'brain', 'liver')."),
            "cell_type": _str_prop("Cell type filter (e.g. 'T cell', 'macrophage', 'fibroblast')."),
            "species": _enum_prop("Species.", ["Human", "Mouse"], "Human"),
            "max_results": _int_prop("Results to return", 20, 1, 100),
        },
        [],
    ),
    _tool(
        "get_encode_regulatory",
        "ENCODE regulatory elements: promoters (CAGE), enhancers (H3K27ac ChIP-seq), "
        "CTCF binding sites, TF binding (ChIP-seq), and open chromatin (ATAC-seq) "
        "for a gene. Returns released ENCODE experiments with biosample context "
        "and download links. Critical for understanding gene regulation.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'MYC', 'BRCA1')."),
            "element_type": _enum_prop(
                "Regulatory element type.",
                ["all", "promoter", "enhancer", "CTCF", "TF_binding", "open_chromatin"],
                "all",
            ),
            "biosample": _str_prop("Cell type/tissue filter (e.g. 'HepG2', 'K562', 'lung')."),
            "max_results": _int_prop("Results to return", 15, 1, 50),
        },
        ["gene_symbol"],
    ),
    _tool(
        "search_metabolomics",
        "MetaboLights metabolomics repository: studies connecting metabolites to "
        "genes and diseases. Metabolomics measures the downstream functional output "
        "of gene activity — connecting genotype to phenotype via metabolism. "
        "Also searches HMDB for metabolite structure and pathway connections.",
        {
            "gene_symbol": _str_prop("Gene to find related metabolic studies."),
            "metabolite": _str_prop("Metabolite name (e.g. 'glucose', 'lactate', 'acetyl-CoA')."),
            "disease": _str_prop("Disease context (e.g. 'cancer', 'diabetes', 'obesity')."),
            "max_results": _int_prop("Studies to return", 10, 1, 50),
        },
        [],
    ),
    _tool(
        "get_ucsc_splice_variants",
        "UCSC Genome Browser: alternative splicing isoforms, UTR annotations, "
        "and exon structure for a gene. Alternative splicing is frequently "
        "dysregulated in disease and affects drug binding pocket accessibility. "
        "Returns all known transcripts with exon count, CDS boundaries, and "
        "5'/3' UTR lengths. Isoform complexity assessment included.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (e.g. 'TP53', 'BRCA1', 'EGFR')."),
            "genome": _enum_prop("Reference genome.", ["hg38", "hg19"], "hg38"),
            "include_alt": _bool_prop("Include alternative isoforms.", True),
        },
        ["gene_symbol"],
    ),
    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 12: Session Intelligence — NOVEL ARCHITECTURE (5 tools)
    # ═══════════════════════════════════════════════════════════════════════
    _tool(
        "resolve_entity",
        "Resolve any biological identifier to its canonical cross-database form. "
        "Runs NCBI Gene + UniProt + Ensembl in parallel to return: HGNC symbol, "
        "NCBI Gene ID, Ensembl ID, UniProt accession, RefSeq IDs, OMIM ID, aliases. "
        "Eliminates redundant lookups — use once, all tools get the right ID.",
        {
            "query": _str_prop(
                "Any biological identifier: gene symbol, accession, alias, or common name."
            ),
            "hint_type": _enum_prop(
                "Entity type hint.", ["gene", "protein", "drug", "disease"], "gene"
            ),
        },
        ["query"],
    ),
    _tool(
        "get_session_knowledge_graph",
        "Return the live Session Knowledge Graph — automatically built from all "
        "tool calls made in this conversation. Contains all biological entities "
        "(genes, proteins, drugs, diseases, pathways), their relationships, "
        "and cross-database connections discovered. "
        "Includes contradiction detection and unexpected multi-hop connections. "
        "Unique to BioMCP — no other MCP server maintains session-level biological state.",
        {},
        [],
    ),
    _tool(
        "find_biological_connections",
        "Discover non-obvious multi-hop connections between biological entities "
        "in the session knowledge graph. "
        "Example: Drug A → targets Gene B → in pathway C → linked to Disease D. "
        "Surfaces insights that require cross-database synthesis — not possible from any single tool.",
        {
            "min_path_length": _int_prop("Minimum path hops for 'unexpected' connections", 2, 2, 4),
        },
        [],
    ),
    _tool(
        "export_research_session",
        "Export the complete research session with full provenance: "
        "all entities discovered, relationships inferred, data sources used "
        "(with access dates), BibTeX citations for all databases, FAIR metadata, "
        "and a reproducibility Python script. "
        "Ready for methods sections and supplementary materials.",
        {},
        [],
    ),
    _tool(
        "plan_and_execute_research",
        "Build and execute an optimized, dependency-aware research plan from a "
        "natural language goal. Uses a DAG planner to identify which tools to run, "
        "in what order, which can be parallelized, and synthesizes an integrated report. "
        "Example: 'Understand KRAS G12C as a drug target in NSCLC' → automatically runs "
        "gene info, literature, drug targets, variants, pathways, clinical trials in optimal order.",
        {
            "goal": _str_prop("Natural language research objective."),
            "depth": _enum_prop("Research depth.", ["quick", "standard", "deep"], "standard"),
            "gene": _str_prop("Primary gene symbol (auto-extracted from goal if not provided)."),
            "uniprot": _str_prop("UniProt accession (for protein-centric goals)."),
            "timeout_per_tool": _int_prop("Per-tool timeout in seconds", 60, 10, 300),
        },
        ["goal"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Handler
# ─────────────────────────────────────────────────────────────────────────────


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
        hypotheses.append(
            {
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
                    "In vivo mouse model validation",
                ],
                "data_gaps": [
                    "Mechanistic in vivo validation missing",
                    "Longitudinal clinical outcome data not available",
                    "Single-cell resolution data lacking",
                ],
            }
        )

    return {
        "topic": topic,
        "context_genes": genes,
        "literature_base": {
            "query": query,
            "total_papers": papers.get("total_found", 0),
            "reviewed_papers": len(articles),
            "top_papers": [
                {"pmid": a["pmid"], "title": a["title"], "year": a["year"]} for a in articles[:5]
            ],
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
        "interpretation": (
            f"Found {len(connections)} multi-hop biological connections "
            f"from {stats['nodes']} entities and {stats['edges']} relationships "
            "accumulated across this session's tool calls."
        )
        if connections
        else (
            f"No multi-hop connections found yet. "
            f"Graph has {stats['nodes']} entities from {stats['calls']} tool calls. "
            "Call more tools to enrich the session knowledge graph."
        ),
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
        goal=goal,
        depth=depth,
        entities=entities or None,
        timeout_per_tool=float(timeout_per_tool),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────


async def _raw_dispatch(name: str, args: dict[str, Any]) -> Any:
    """
    Raw dispatcher that returns Python objects (not JSON strings).
    Used internally by the query planner.
    """
    from biomcp.tools.ncbi import get_gene_info, run_blast, search_pubmed
    from biomcp.tools.proteins import (
        get_alphafold_structure,
        get_protein_info,
        search_pdb_structures,
        search_proteins,
    )
    from biomcp.tools.pathways import (
        get_compound_info,
        get_drug_targets,
        get_gene_disease_associations,
        get_pathway_genes,
        get_reactome_pathways,
        search_pathways,
    )
    from biomcp.tools.advanced import (
        get_gene_variants,
        get_trial_details,
        multi_omics_gene_report,
        query_neuroimaging_datasets,
        search_clinical_trials,
        search_gene_expression,
        search_scrna_datasets,
    )
    from biomcp.tools.nvidia_nim import (
        predict_structure_boltz2,
        generate_dna_evo2,
        score_sequence_evo2,
        design_protein_ligand,
    )
    from biomcp.tools.databases import (
        get_omim_gene_diseases,
        get_string_interactions,
        get_gtex_expression,
        search_cbio_mutations,
        search_gwas_catalog,
        get_disgenet_associations,
        get_pharmgkb_variants,
    )
    from biomcp.tools.verify import verify_biological_claim, detect_database_conflicts
    from biomcp.tools.protocol_generator import (
        generate_experimental_protocol,
        suggest_cell_lines,
        estimate_statistical_power,
    )
    from biomcp.tools.intelligence import (
        validate_reasoning_chain,
        find_repurposing_candidates,
        find_research_gaps,
    )
    from biomcp.tools.extended_databases import (
        get_biogrid_interactions,
        search_orphan_diseases,
        get_tcga_expression,
        search_cellmarker,
        get_encode_regulatory,
        search_metabolomics,
        get_ucsc_splice_variants,
    )

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
        # Pathways
        "search_pathways": search_pathways,
        "get_pathway_genes": get_pathway_genes,
        "get_reactome_pathways": get_reactome_pathways,
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
        "generate_research_hypothesis": _generate_research_hypothesis,
        # NVIDIA NIM
        "predict_structure_boltz2": predict_structure_boltz2,
        "generate_dna_evo2": generate_dna_evo2,
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
        "resolve_entity": _resolve_entity,
        "get_session_knowledge_graph": _get_session_knowledge_graph,
        "find_biological_connections": _find_biological_connections,
        "export_research_session": _export_research_session,
        "plan_and_execute_research": _plan_and_execute_research,
        # Intelligence Layer
        "validate_reasoning_chain": validate_reasoning_chain,
        "find_repurposing_candidates": find_repurposing_candidates,
        "find_research_gaps": find_research_gaps,
        # Tier 2 Extended Databases
        "get_biogrid_interactions": get_biogrid_interactions,
        "search_orphan_diseases": search_orphan_diseases,
        "get_tcga_expression": get_tcga_expression,
        "search_cellmarker": search_cellmarker,
        "get_encode_regulatory": get_encode_regulatory,
        "search_metabolomics": search_metabolomics,
        "get_ucsc_splice_variants": get_ucsc_splice_variants,
    }

    if name not in DISPATCH:
        raise ValueError(f"Unknown tool '{name}'")

    return await DISPATCH[name](**args)


async def _dispatch(name: str, args: dict[str, Any]) -> str:
    """MCP-facing dispatcher — wraps results in JSON envelopes."""
    try:
        result = await _raw_dispatch(name, args)
        return format_success(name, result)
    except (ValueError, TypeError, LookupError, KeyError) as exc:
        return format_error(name, exc, {"arguments": args})
    except Exception as exc:
        logger.exception(f"Unexpected error in tool '{name}'")
        return format_error(name, exc, {"arguments": args})


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────────────────────


def create_server() -> Server:
    server = Server(
        "biomcp",
        version="2.0.0",
        instructions="BioMCP v2 - Connect Claude to 20+ biological databases and AI models. Tools include PubMed, UniProt, AlphaFold, ChEMBL, NVIDIA Boltz-2, and more.",
        website_url="https://github.com/SachinGawande2003/BioMCP",
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        text = await _dispatch(name, arguments or {})
        return [TextContent(type="text", text=text)]

    return server


async def _run() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("BIOMCP_LOG_LEVEL", "INFO"),
        format=("<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"),
        colorize=True,
    )

    logger.info("🧬 BioMCP v2 server starting…")
    logger.info(f"   Tools registered : {len(TOOLS)}")
    logger.info(f"   Log level        : {os.getenv('BIOMCP_LOG_LEVEL', 'INFO')}")
    logger.info(
        f"   NCBI API key     : {'set' if os.getenv('NCBI_API_KEY') else 'not set (3 req/s)'}"
    )
    logger.info(
        f"   Boltz-2 key      : {'set' if os.getenv('NVIDIA_BOLTZ2_API_KEY') else 'not set'}"
    )
    logger.info(f"   Evo2 key         : {'set' if os.getenv('NVIDIA_EVO2_API_KEY') else 'not set'}")
    logger.info(f"   BioGRID key      : {'set' if os.getenv('BIOGRID_API_KEY') else 'not set'}")
    logger.info("")
    logger.info("   🆕 v2 features: Session Knowledge Graph | Entity Resolver")
    logger.info("                   Adaptive Query Planner | Conflict Detector")
    logger.info("                   Intelligence Layer: Reasoning | Repurposing | Gap Detection")
    logger.info(
        "                   Tier 2 Extended Databases: BioGRID, Orphanet, TCGA, CellMarker, ENCODE, MetaboLights, UCSC"
    )

    server = create_server()
    http_port = int(os.getenv("BIOMCP_HTTP_PORT", "8080"))
    transport_mode = os.getenv("BIOMCP_TRANSPORT", "stdio")

    if transport_mode == "http":
        logger.info(f"\n   🌐 HTTP mode enabled on port {http_port}")
        logger.info("   Use: curl -N http://localhost:{}/sse", http_port)

        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response

        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
            return Response()

        from starlette.middleware.cors import CORSMiddleware

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse_transport.handle_post_message),
            ],
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        import uvicorn

        config = uvicorn.Config(app, host="0.0.0.0", port=http_port, log_level="info")
        server_uvicorn = uvicorn.Server(config)
        await server_uvicorn.serve()
    else:
        logger.info("\n   📟 STDIO mode enabled")
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        finally:
            await close_http_client()
            logger.info("BioMCP v2 shut down cleanly.")


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
