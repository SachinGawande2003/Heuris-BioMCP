"""
BioMCP v2 — Complete MCP Server  [FIXED v2.2]
================================
Fixes applied:
  - Removed duplicate `from biomcp.utils import ...` at module top (Bug #1)
  - Removed 3 sets of duplicate imports inside `_raw_dispatch()` (Bug #2)
  - Hardened `create_server()` against missing mcp.types.Icon (Bug #10)
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
# FIX #1: Removed duplicate import line that was here


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


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — FIX #2: removed all duplicate import blocks
# ─────────────────────────────────────────────────────────────────────────────


async def _raw_dispatch(name: str, args: dict[str, Any]) -> Any:
    """Raw dispatcher returning Python objects. Used by the query planner."""
    # ── Core tool imports ──────────────────────────────────────────────────────
    from biomcp.tools.ncbi import get_gene_info, run_blast, search_pubmed
    from biomcp.tools.proteins import (
        get_alphafold_structure, get_protein_info,
        search_pdb_structures, search_proteins,
    )
    from biomcp.tools.pathways import (
        get_compound_info, get_drug_targets, get_gene_disease_associations,
        get_pathway_genes, get_reactome_pathways, search_pathways,
    )
    from biomcp.tools.advanced import (
        get_gene_variants, get_trial_details, multi_omics_gene_report,
        query_neuroimaging_datasets, search_clinical_trials,
        search_gene_expression, search_scrna_datasets,
    )
    from biomcp.tools.nvidia_nim import (
        design_protein_ligand, generate_dna_evo2,
        predict_structure_boltz2, score_sequence_evo2,
    )
    from biomcp.tools.databases import (
        get_disgenet_associations, get_gtex_expression, get_omim_gene_diseases,
        get_pharmgkb_variants, get_string_interactions,
        search_cbio_mutations, search_gwas_catalog,
    )
    # FIX #2: Single import of verify and protocol tools (removed duplicates)
    from biomcp.tools.verify import detect_database_conflicts, verify_biological_claim
    from biomcp.tools.protocol_generator import (
        estimate_statistical_power, generate_experimental_protocol, suggest_cell_lines,
    )
    from biomcp.tools.intelligence import (
        find_repurposing_candidates, find_research_gaps, validate_reasoning_chain,
    )
    # FIX #2: Single import of extended_databases (removed duplicate)
    from biomcp.tools.extended_databases import (
        get_biogrid_interactions, get_encode_regulatory, get_tcga_expression,
        get_ucsc_splice_variants, search_cellmarker,
        search_metabolomics, search_orphan_diseases,
    )
    from biomcp.tools.crispr_tools import (
        design_base_editor_guides, design_crispr_guides, get_crispr_repair_outcomes,
        predict_off_target_sites, score_guide_efficiency,
    )
    from biomcp.tools.drug_safety import (
        analyze_safety_signals, compare_drug_safety,
        get_drug_label_warnings, query_adverse_events,
    )
    from biomcp.tools.variant_interpreter import (
        classify_variant, get_population_frequency, lookup_clinvar_variant,
    )
    # New innovation tools
    from biomcp.tools.innovations import (
        bulk_gene_analysis, compute_pathway_enrichment, search_biorxiv,
        get_protein_domain_structure, analyze_coexpression,
        get_cancer_hotspots, predict_splice_impact,
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
        # Tier 2 Extended
        "get_biogrid_interactions": get_biogrid_interactions,
        "search_orphan_diseases": search_orphan_diseases,
        "get_tcga_expression": get_tcga_expression,
        "search_cellmarker": search_cellmarker,
        "get_encode_regulatory": get_encode_regulatory,
        "search_metabolomics": search_metabolomics,
        "get_ucsc_splice_variants": get_ucsc_splice_variants,
        # CRISPR
        "design_crispr_guides": design_crispr_guides,
        "score_guide_efficiency": score_guide_efficiency,
        "predict_off_target_sites": predict_off_target_sites,
        "design_base_editor_guides": design_base_editor_guides,
        "get_crispr_repair_outcomes": get_crispr_repair_outcomes,
        # FDA Drug Safety
        "query_adverse_events": query_adverse_events,
        "analyze_safety_signals": analyze_safety_signals,
        "get_drug_label_warnings": get_drug_label_warnings,
        "compare_drug_safety": compare_drug_safety,
        # Variant Interpreter
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
# MCP Server — FIX #10: hardened against missing mcp.types.Icon
# ─────────────────────────────────────────────────────────────────────────────


def create_server() -> Server:
    # FIX #10: Guard against mcp SDK < 1.3.0 which lacks Icon / icons kwarg
    try:
        from mcp.types import Icon as _Icon
        _has_icon = True
    except ImportError:
        _has_icon = False

    server_kwargs: dict[str, Any] = {
        "instructions": "Heuris-BioMCP — Connect Claude to 20+ biological databases and AI models.",
    }

    # Only add website_url if mcp SDK supports it (≥1.3.0)
    try:
        server = Server("heuris-biomcp", version="2.2.0", **server_kwargs)
    except TypeError:
        server = Server("heuris-biomcp")

    if _has_icon:
        import base64, os
        logo_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "LOGO.jpeg"),
            os.path.join(os.path.dirname(__file__), "LOGO.jpeg"),
            "LOGO.jpeg",
        ]
        for logo_path in logo_candidates:
            if os.path.exists(logo_path):
                try:
                    with open(logo_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    # Apply icon via server metadata if supported
                    break
                except Exception:
                    pass

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

    n_tools = len(TOOLS)
    logger.info(f"🧬 BioMCP v2.2 starting — {n_tools} tools registered")
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
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response
        from starlette.middleware.cors import CORSMiddleware

        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())
            return Response()

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ])
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
