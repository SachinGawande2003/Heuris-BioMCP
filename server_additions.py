"""
BioMCP Server Additions — New Tool Registrations
==================================================
Add these tool schemas to the TOOLS list in server.py
and add the imports + dispatch entries to _raw_dispatch().

INSTRUCTIONS:
1. Copy the CRISPR_TOOLS, DRUG_SAFETY_TOOLS, VARIANT_TOOLS lists
   and extend TOOLS with them in server.py
2. Add the imports and dispatch entries shown in _raw_dispatch() additions
3. Copy the 3 new .py files into src/biomcp/tools/
"""

from mcp.types import Tool
from biomcp.server import _tool, _str_prop, _int_prop, _bool_prop, _float_prop, _enum_prop

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 15: CRISPR Design Suite (5 tools) — NEW
# ─────────────────────────────────────────────────────────────────────────────

CRISPR_TOOLS = [
    _tool(
        "design_crispr_guides",
        "Design CRISPR sgRNA guides for a target gene with Doench 2016 efficiency scoring. "
        "Fetches coding sequence from Ensembl, identifies all PAM sites, scores candidates by "
        "GC content, positional nucleotides, poly-T avoidance, and seed region quality. "
        "Returns ranked guides with ordering sequences ready for synthesis.",
        {
            "gene_symbol":     _str_prop("HGNC gene symbol (e.g. 'TP53', 'KRAS', 'BRCA1')."),
            "target_region":   _enum_prop(
                "Genomic region to target.",
                ["early_exons", "all_coding", "promoter"], "early_exons",
            ),
            "cas_variant":     _enum_prop(
                "Cas nuclease to design for.",
                ["SpCas9", "SaCas9", "Cas12a", "CjCas9"], "SpCas9",
            ),
            "n_guides":        _int_prop("Top guides to return", 5, 1, 20),
            "exclude_restriction_sites": {
                "type": "array", "items": {"type": "string"},
                "description": "Restriction enzymes to avoid (e.g. ['BsmBI', 'BbsI']). Default: both.",
            },
            "min_score":       _float_prop("Minimum efficiency score to include (0–100). Default 40.", 40.0),
        },
        ["gene_symbol"],
    ),
    _tool(
        "score_guide_efficiency",
        "Score a user-provided sgRNA sequence using the Doench 2016 RS2-inspired multi-feature model. "
        "Returns efficiency score (0–100) with full feature breakdown: GC content, positional "
        "nucleotide weights, poly-T avoidance, seed region quality, and restriction site conflicts. "
        "Generates ready-to-order oligo sequences for direct synthesis.",
        {
            "guide_sequence": _str_prop("17–24nt guide RNA sequence (5'→3', DNA convention)."),
            "pam_sequence":   _str_prop("PAM sequence for verification (e.g. 'TGG' for SpCas9). Optional."),
            "cas_variant":    _enum_prop("Cas variant.", ["SpCas9", "SaCas9", "Cas12a"], "SpCas9"),
        },
        ["guide_sequence"],
    ),
    _tool(
        "predict_off_target_sites",
        "Predict CRISPR off-target risk using seed-region analysis and optional NCBI BLAST. "
        "Analyzes seed region GC content, repetitive sequence elements, and genomic similarity. "
        "When use_blast=True, submits 12nt seed to NCBI BLAST to identify similar human genomic loci. "
        "Returns specificity score (0–100) and risk tier (LOW/MEDIUM/HIGH).",
        {
            "guide_sequence": _str_prop("20nt sgRNA sequence (5'→3', DNA convention)."),
            "cas_variant":    _enum_prop("Cas variant.", ["SpCas9", "SaCas9", "Cas12a"], "SpCas9"),
            "mismatches":     _int_prop("Maximum mismatches to consider as off-target", 3, 1, 5),
            "use_blast":      _bool_prop("Submit seed region to NCBI BLAST for genomic hits (~30s extra).", True),
        },
        ["guide_sequence"],
    ),
    _tool(
        "design_base_editor_guides",
        "Design guides for precision base editing (CBE or ABE) to introduce specific mutations. "
        "Cytosine Base Editors (CBE) convert C→T; Adenine Base Editors (ABE) convert A→G. "
        "Automatically selects editor type from target mutation, identifies compatible guides "
        "with target base in edit window, and flags bystander edit risks.",
        {
            "gene_symbol":     _str_prop("HGNC gene symbol."),
            "target_mutation": _str_prop(
                "Target mutation (e.g. 'G12D', 'p.Arg175His', 'c.524G>A', 'W53*')."
            ),
            "editor_type":     _enum_prop(
                "Base editor type.", ["CBE", "ABE", "auto"], "auto"
            ),
        },
        ["gene_symbol", "target_mutation"],
    ),
    _tool(
        "get_crispr_repair_outcomes",
        "Predict CRISPR-Cas9 repair outcomes: NHEJ frameshift probability, indel distribution, "
        "and HDR efficiency estimate. Based on Shen et al. 2018 (Nature Methods) and FORECasT "
        "model principles. Returns top predicted indel types, knockout probability, and "
        "recommendations for maximizing HDR efficiency.",
        {
            "gene_symbol":     _str_prop("HGNC gene symbol."),
            "guide_sequence":  _str_prop("20nt sgRNA sequence."),
            "repair_template": _str_prop("Optional HDR template sequence (ssODN) for precise edit."),
            "cell_line":       _enum_prop(
                "Cell line context for efficiency calibration.",
                ["generic", "HEK293", "HeLa", "primary"], "generic",
            ),
        },
        ["gene_symbol", "guide_sequence"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 16: FDA Drug Safety Intelligence (4 tools) — NEW
# ─────────────────────────────────────────────────────────────────────────────

DRUG_SAFETY_TOOLS = [
    _tool(
        "query_adverse_events",
        "Query FDA FAERS (Adverse Event Reporting System) for drug safety signals. "
        "Access millions of adverse event reports with outcome breakdown (deaths, hospitalizations), "
        "demographic analysis, event category filtering, and yearly trend data. "
        "Real-time pharmacovigilance data with no API key required.",
        {
            "drug_name":    _str_prop("Drug name (generic or brand). E.g. 'adalimumab', 'Humira', 'ibuprofen'."),
            "event_type":   _enum_prop(
                "Event category to focus on.",
                ["all", "cardiac", "hepatic", "hematologic", "neurological",
                 "renal", "hypersensitivity", "respiratory", "oncology"], "all",
            ),
            "serious_only": _bool_prop("Only return serious adverse events (hospitalization, death).", False),
            "max_results":  _int_prop("Maximum reports to analyze", 50, 10, 500),
            "patient_sex":  _enum_prop("Filter by patient sex.", ["", "male", "female"], ""),
            "age_group":    _enum_prop(
                "Patient age group.", ["", "pediatric", "adult", "elderly"], ""
            ),
        },
        ["drug_name"],
    ),
    _tool(
        "analyze_safety_signals",
        "Pharmacovigilance disproportionality analysis on FDA FAERS data. "
        "Calculates PRR (Proportional Reporting Ratio), ROR (Reporting Odds Ratio), "
        "and IC (Information Component/Bayesian). Applies WHO UMC signal detection criteria "
        "(PRR ≥ 2, χ² ≥ 4, n ≥ 3). Essential for automated signal detection and "
        "clinical safety literature context.",
        {
            "drug_name":  _str_prop("Drug of interest."),
            "event_terms": {
                "type": "array", "items": {"type": "string"},
                "description": "Adverse event MedDRA terms to analyze. Auto-selects top events if empty.",
            },
            "comparators": {
                "type": "array", "items": {"type": "string"},
                "description": "Comparator drugs (e.g. drug class members).",
            },
        },
        ["drug_name"],
    ),
    _tool(
        "get_drug_label_warnings",
        "Retrieve FDA-approved drug label safety sections directly from DailyMed. "
        "Returns black box warnings, contraindications, adverse reactions, drug interactions, "
        "pregnancy category, and use in specific populations — the legally binding FDA label. "
        "Requires no API key; uses OpenFDA public database.",
        {
            "drug_name": _str_prop("Generic or brand drug name (e.g. 'warfarin', 'Coumadin')."),
            "sections": {
                "type": "array", "items": {"type": "string"},
                "description": "Label sections to retrieve. Default: all safety sections.",
            },
        },
        ["drug_name"],
    ),
    _tool(
        "compare_drug_safety",
        "Head-to-head safety comparison between 2–5 drugs using FDA FAERS data. "
        "Identifies events reported disproportionately for one drug vs class members, "
        "event categories unique to specific drugs, and relative reporting volume. "
        "Essential for drug class comparisons, SOC vs novel agent analysis, and "
        "generic vs brand equivalence assessment.",
        {
            "drugs": {
                "type": "array", "items": {"type": "string"},
                "description": "List of 2–5 drug names to compare.",
            },
            "event_category": _enum_prop(
                "Focus event category.",
                ["all", "cardiac", "hepatic", "hematologic", "neurological"], "all",
            ),
        },
        ["drugs"],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 17: Variant Interpreter (3 tools) — NEW
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_TOOLS = [
    _tool(
        "classify_variant",
        "Classify a genetic variant using ACMG/AMP 2015 guidelines (Richards et al.) — "
        "the global standard for clinical variant interpretation. Automatically gathers "
        "population frequency (gnomAD), functional predictions (VEP/SIFT/PolyPhen), "
        "and ClinVar data to apply evidence codes and return 5-tier classification: "
        "Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign. "
        "Research use only — not for clinical decisions without expert review.",
        {
            "gene_symbol":   _str_prop("HGNC gene symbol (e.g. 'BRCA1', 'TP53')."),
            "variant":       _str_prop(
                "Variant notation: protein ('p.Arg175His', 'R175H'), cDNA ('c.524G>A'), "
                "rsID ('rs28934578'), or HGVS ('NM_000546.5:c.524G>A')."
            ),
            "inheritance":   _enum_prop(
                "Inheritance pattern — affects PM2 threshold.",
                ["AD", "AR", "XL", "unknown"], "unknown",
            ),
            "consequence":   _str_prop("VEP consequence if known (e.g. 'missense_variant', 'stop_gained')."),
            "proband_phenotype": _str_prop("Clinical phenotype to assess disease relevance. Optional."),
        },
        ["gene_symbol", "variant"],
    ),
    _tool(
        "get_population_frequency",
        "Query gnomAD v4 for population-specific allele frequencies across all major human populations. "
        "Returns global AF, population breakdown (African, European, East Asian, South Asian, etc.), "
        "homozygote counts, and ACMG PM2/BS1 interpretation guidance. "
        "Essential for variant pathogenicity assessment — provides the world's largest "
        "reference dataset for human genetic variation.",
        {
            "variant_id":  _str_prop("Variant in rsID ('rs28934578') or gnomAD format ('17-7674220-C-T')."),
            "dataset":     _enum_prop("gnomAD dataset version.", ["gnomad_r4", "gnomad_r2_1"], "gnomad_r4"),
            "populations": {
                "type": "array", "items": {"type": "string"},
                "description": "Populations to report (e.g. ['afr', 'eas', 'nfe']). Default: all.",
            },
        },
        ["variant_id"],
    ),
    _tool(
        "lookup_clinvar_variant",
        "Search ClinVar for clinical significance classifications and submission data. "
        "ClinVar is the FDA-recognized variant database for clinical interpretation. "
        "Returns star rating (review status), submitter consensus, associated phenotypes, "
        "and links to full submission records. Supports rsID, HGVS notation, protein change, "
        "or direct ClinVar variation ID search.",
        {
            "gene_symbol": _str_prop("HGNC gene symbol (optional, used with variant for disambiguation)."),
            "variant":     _str_prop("Variant notation (rsID, HGVS, protein change). Optional."),
            "clinvar_id":  _str_prop("Direct ClinVar variation ID (most specific). Optional."),
            "max_results": _int_prop("Maximum results to return", 5, 1, 20),
        },
        [],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH ADDITIONS for _raw_dispatch() in server.py
# ─────────────────────────────────────────────────────────────────────────────
# Add these imports at the top of _raw_dispatch():
"""
from biomcp.tools.crispr_tools import (
    design_crispr_guides,
    score_guide_efficiency,
    predict_off_target_sites,
    design_base_editor_guides,
    get_crispr_repair_outcomes,
)
from biomcp.tools.drug_safety import (
    query_adverse_events,
    analyze_safety_signals,
    get_drug_label_warnings,
    compare_drug_safety,
)
from biomcp.tools.variant_interpreter import (
    classify_variant,
    get_population_frequency,
    lookup_clinvar_variant,
)
"""

# Add these entries to the DISPATCH dict:
NEW_DISPATCH_ENTRIES = """
    # CRISPR Design Suite
    "design_crispr_guides":      design_crispr_guides,
    "score_guide_efficiency":    score_guide_efficiency,
    "predict_off_target_sites":  predict_off_target_sites,
    "design_base_editor_guides": design_base_editor_guides,
    "get_crispr_repair_outcomes":get_crispr_repair_outcomes,
    # FDA Drug Safety Intelligence
    "query_adverse_events":      query_adverse_events,
    "analyze_safety_signals":    analyze_safety_signals,
    "get_drug_label_warnings":   get_drug_label_warnings,
    "compare_drug_safety":       compare_drug_safety,
    # Variant Interpreter
    "classify_variant":          classify_variant,
    "get_population_frequency":  get_population_frequency,
    "lookup_clinvar_variant":    lookup_clinvar_variant,
"""

# To extend TOOLS in server.py, add after the current TOOLS list:
"""
TOOLS.extend(CRISPR_TOOLS)
TOOLS.extend(DRUG_SAFETY_TOOLS)
TOOLS.extend(VARIANT_TOOLS)
"""

# Update docstring tool count: 47 existing + 12 new = 59 total tools

print("server_additions.py — copy the 3 new .py files and follow INSTRUCTIONS above.")
print(f"New tool count: 47 existing + 5 CRISPR + 4 drug safety + 3 variant = 59 total")
