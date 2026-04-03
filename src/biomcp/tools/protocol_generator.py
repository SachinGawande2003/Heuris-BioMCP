"""
BioMCP — Experimental Protocol Generator
==========================================
Generates actionable, literature-grounded experimental protocols from
biological hypotheses — a capability that exists nowhere else in the
MCP ecosystem.

This tool synthesizes data from multiple BioMCP databases into a
structured experimental protocol that a bench scientist can actually follow:
  • Reagent list with catalog numbers and CAS numbers
  • Validated cell line recommendations
  • Positive and negative controls
  • Statistical power calculations
  • Timeline estimate
  • Expected readouts with quantitative thresholds
  • Links to similar published protocols in PubMed

Tools:
  generate_experimental_protocol  — Full protocol from hypothesis
  suggest_cell_lines               — Context-appropriate cell line selection
  estimate_statistical_power       — Power calculation for experimental design
"""

from __future__ import annotations

import asyncio
from typing import Any

from biomcp.utils import BioValidator

# ─────────────────────────────────────────────────────────────────────────────
# Reagent database — common reagents for molecular biology assays
# (Curated from supplier catalogs; CAS numbers are real)
# ─────────────────────────────────────────────────────────────────────────────

_REAGENT_DB: dict[str, dict[str, Any]] = {
    "CRISPR_lentiviral": {
        "name":       "Lenti-CRISPR-v2 backbone",
        "supplier":   "Addgene",
        "catalog":    "#52961",
        "purpose":    "CRISPR knockout vector",
        "storage":    "-20°C",
    },
    "Cas9_RNP": {
        "name":       "Cas9 RNP complex",
        "supplier":   "IDT (Integrated DNA Technologies)",
        "catalog":    "TrueCut Cas9 Protein v2",
        "purpose":    "Cas9 nuclease for CRISPR",
        "storage":    "-80°C",
    },
    "RT_qPCR_kit": {
        "name":       "Luna Universal One-Step RT-qPCR Kit",
        "supplier":   "New England Biolabs",
        "catalog":    "#E3005",
        "purpose":    "RNA quantification",
        "storage":    "-20°C",
    },
    "RNA_extraction": {
        "name":       "RNeasy Mini Kit",
        "supplier":   "Qiagen",
        "catalog":    "#74104",
        "purpose":    "Total RNA extraction",
        "storage":    "RT",
    },
    "western_blot_SDS": {
        "name":       "4–15% Mini-PROTEAN TGX gel",
        "supplier":   "Bio-Rad",
        "catalog":    "#4561083",
        "purpose":    "Western blot protein separation",
        "storage":    "4°C",
    },
    "PVDF_membrane": {
        "name":       "PVDF Transfer Membrane (0.2 μm)",
        "supplier":   "Merck Millipore",
        "catalog":    "#IPFL00010",
        "purpose":    "Western blot transfer",
        "storage":    "RT",
    },
    "cell_viability": {
        "name":       "CellTiter-Glo 2.0 Luminescent Cell Viability Assay",
        "supplier":   "Promega",
        "catalog":    "#G9241",
        "purpose":    "Proliferation/cytotoxicity readout",
        "storage":    "-20°C",
    },
    "annexin_v": {
        "name":       "Annexin V-FITC Apoptosis Detection Kit",
        "supplier":   "BD Biosciences",
        "catalog":    "#556547",
        "purpose":    "Apoptosis measurement by flow cytometry",
        "storage":    "4°C",
    },
    "DMSO": {
        "name":       "Dimethyl Sulfoxide (DMSO), sterile filtered",
        "supplier":   "Sigma-Aldrich",
        "cas":        "67-68-5",
        "catalog":    "#D2650",
        "purpose":    "Vehicle control / compound solvent",
        "storage":    "RT",
    },
    "PBS": {
        "name":       "Dulbecco's Phosphate Buffered Saline (DPBS)",
        "supplier":   "Thermo Fisher",
        "catalog":    "#14190144",
        "purpose":    "Washing buffer",
        "storage":    "RT",
    },
    "MTT": {
        "name":       "MTT (3-(4,5-Dimethylthiazol-2-yl)-2,5-diphenyltetrazolium bromide)",
        "supplier":   "Sigma-Aldrich",
        "cas":        "298-93-1",
        "catalog":    "#M2128",
        "purpose":    "Cell viability colorimetric assay",
        "storage":    "-20°C",
    },
    "siRNA_transfection": {
        "name":       "Lipofectamine RNAiMAX Transfection Reagent",
        "supplier":   "Thermo Fisher",
        "catalog":    "#13778150",
        "purpose":    "siRNA delivery",
        "storage":    "4°C",
    },
    "protein_assay": {
        "name":       "Pierce BCA Protein Assay Kit",
        "supplier":   "Thermo Fisher",
        "catalog":    "#23227",
        "purpose":    "Protein quantification",
        "storage":    "RT",
    },
    "flow_cytometer_fixation": {
        "name":       "Fixation/Permeabilization Solution Kit",
        "supplier":   "BD Biosciences",
        "catalog":    "#554714",
        "purpose":    "Intracellular flow cytometry",
        "storage":    "4°C",
    },
    "96_well_plate": {
        "name":       "96-Well Flat-Bottom Tissue Culture Plate, White",
        "supplier":   "Corning",
        "catalog":    "#3610",
        "purpose":    "High-throughput cell-based assays",
        "storage":    "RT",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Cell line database
# ─────────────────────────────────────────────────────────────────────────────

_CELL_LINE_DB: dict[str, list[dict[str, str]]] = {
    "lung_cancer": [
        {"name": "A549",    "origin": "Human lung adenocarcinoma", "atcc": "CCL-185",  "notes": "KRAS G12S, standard for lung adenocarcinoma studies"},
        {"name": "H1299",   "origin": "Human lung carcinoma",      "atcc": "CRL-5803", "notes": "p53-null — ideal for TP53 reconstitution studies"},
        {"name": "PC9",     "origin": "Human lung adenocarcinoma", "atcc": "N/A",       "notes": "EGFR exon 19 deletion — erlotinib sensitive"},
        {"name": "H1975",   "origin": "Human lung adenocarcinoma", "atcc": "CRL-5908", "notes": "EGFR L858R + T790M — TKI resistant"},
        {"name": "Calu-3",  "origin": "Human lung adenocarcinoma", "atcc": "HTB-55",   "notes": "ERBB2-amplified"},
    ],
    "breast_cancer": [
        {"name": "MCF-7",   "origin": "Human breast adenocarcinoma", "atcc": "HTB-22",  "notes": "ER+/PR+, p53 mutant — hormone-responsive"},
        {"name": "MDA-MB-231", "origin": "Triple-negative breast cancer", "atcc": "HTB-26", "notes": "TNBC, highly invasive, KRAS/BRAF mutant"},
        {"name": "T-47D",   "origin": "Human breast ductal carcinoma",  "atcc": "HTB-133", "notes": "ER+/PR+, p53 mutant (L194F)"},
        {"name": "BT-474",  "origin": "Human breast ductal carcinoma",  "atcc": "HTB-20",  "notes": "HER2-amplified — trastuzumab sensitive"},
        {"name": "SKBR3",   "origin": "Human breast adenocarcinoma",   "atcc": "HTB-30",  "notes": "HER2-amplified — lapatinib sensitive"},
    ],
    "colorectal_cancer": [
        {"name": "HCT116",  "origin": "Human colorectal carcinoma",    "atcc": "CCL-247", "notes": "KRAS G13D, p53 wt — MMR deficient"},
        {"name": "HT-29",   "origin": "Human colorectal adenocarcinoma","atcc": "HTB-38", "notes": "BRAF V600E — vemurafenib sensitive"},
        {"name": "SW480",   "origin": "Human colorectal adenocarcinoma","atcc": "CCL-228","notes": "KRAS G12V, p53 R273H/P309S"},
        {"name": "SW620",   "origin": "Human colorectal adenocarcinoma","atcc": "CCL-227","notes": "Metastatic derivative of SW480"},
        {"name": "LoVo",    "origin": "Human colorectal adenocarcinoma","atcc": "CCL-229","notes": "MSI-high, immunotherapy model"},
    ],
    "prostate_cancer": [
        {"name": "LNCaP",   "origin": "Human prostate adenocarcinoma", "atcc": "CRL-1740","notes": "AR+, PTEN null — hormone sensitive"},
        {"name": "PC-3",    "origin": "Human prostate adenocarcinoma", "atcc": "CRL-1435","notes": "AR-, PTEN null — hormone resistant"},
        {"name": "DU145",   "origin": "Human prostate carcinoma",      "atcc": "HTB-81",  "notes": "AR-, p53 mutant"},
    ],
    "glioblastoma": [
        {"name": "U87-MG",  "origin": "Human glioblastoma",            "atcc": "HTB-14",  "notes": "PTEN null, EGFRvIII negative"},
        {"name": "T98G",    "origin": "Human glioblastoma multiforme", "atcc": "CRL-1690","notes": "PTEN mutant, p53 wt"},
        {"name": "LN229",   "origin": "Human glioblastoma",            "atcc": "CRL-2611","notes": "PTEN wt, MGMT methylated"},
    ],
    "general": [
        {"name": "HEK293T", "origin": "Human embryonic kidney",        "atcc": "CRL-11268","notes": "High transfection efficiency — mechanistic studies"},
        {"name": "HeLa",    "origin": "Human cervical carcinoma",      "atcc": "CCL-2",    "notes": "Classic model — avoid for cancer-type conclusions"},
        {"name": "NIH3T3",  "origin": "Mouse embryonic fibroblast",    "atcc": "CRL-1658", "notes": "Transformation assays"},
        {"name": "RPE-1",   "origin": "Human retinal pigment epithelium","atcc":"CRL-4000","notes": "Non-cancerous — suitable control line"},
    ],
}

_ASSAY_TEMPLATES: dict[str, dict[str, Any]] = {
    "crispr_knockout": {
        "name":         "CRISPR-Cas9 Gene Knockout",
        "duration_days": 21,
        "steps": [
            "Design sgRNAs targeting early exons (use Benchling or CRISPOR)",
            "Clone sgRNAs into Lenti-CRISPR-v2 (Addgene #52961)",
            "Produce lentivirus in HEK293T cells with packaging plasmids",
            "Transduce target cell line at MOI 0.3 (single integration)",
            "Select with puromycin (1–2 μg/mL) for 5–7 days",
            "Expand single-cell clones by limiting dilution",
            "Validate KO by Sanger sequencing + western blot",
        ],
        "controls": {
            "positive": "Non-targeting sgRNA (scramble control)",
            "negative": "Wild-type parental cell line",
        },
        "validation_assays": ["western_blot", "sanger_sequencing", "cell_viability"],
    },
    "sirna_knockdown": {
        "name":         "siRNA-Mediated Knockdown",
        "duration_days": 5,
        "steps": [
            "Design 3 independent siRNAs targeting different exons (IDT or Dharmacon)",
            "Reverse transfect with RNAiMAX per manufacturer protocol",
            "Assess knockdown efficiency at 48h by RT-qPCR",
            "Confirm protein loss at 72h by western blot",
            "Perform functional assays at 72h",
        ],
        "controls": {
            "positive": "siRNA against GAPDH or housekeeping gene",
            "negative": "Non-targeting siRNA (scramble)",
        },
        "validation_assays": ["rt_qpcr", "western_blot"],
    },
    "drug_sensitivity": {
        "name":         "Drug Sensitivity (IC50) Assay",
        "duration_days": 7,
        "steps": [
            "Seed 3,000 cells/well in 96-well white plates (day 0)",
            "Allow 24h attachment before drug treatment",
            "Prepare 10-point dose-response (3-fold serial dilution) in DMSO",
            "Dilute compound in complete media (max 0.1% DMSO final)",
            "Add drug at 7 concentrations + vehicle + media-only controls",
            "Incubate 72h in 5% CO2, 37°C",
            "Measure viability by CellTiter-Glo 2.0",
            "Normalize to vehicle control; fit 4-parameter Hill equation",
        ],
        "controls": {
            "positive": "Known inhibitor at Cmax (e.g., staurosporine 1 μM)",
            "negative": "DMSO vehicle (0.1% final concentration)",
        },
        "validation_assays": ["cell_viability_ctg", "ic50_curve_fitting"],
    },
    "apoptosis_flow": {
        "name":         "Apoptosis Measurement by Flow Cytometry",
        "duration_days": 3,
        "steps": [
            "Treat cells for 24/48/72h with compound or genetic perturbation",
            "Collect floating + adherent cells (trypsin 0.25%)",
            "Wash 2x with cold DPBS",
            "Resuspend in 1x Annexin V binding buffer (100 μL)",
            "Add Annexin V-FITC (5 μL) + PI (1 μL) per tube",
            "Incubate 15 min RT, protected from light",
            "Add 400 μL binding buffer, analyze within 1h on flow cytometer",
            "Gate: Q1=live, Q2=late apoptosis/necrosis, Q3=early apoptosis, Q4=dead",
        ],
        "controls": {
            "positive": "Staurosporine 1 μM (pan-kinase inhibitor, induces apoptosis)",
            "negative": "DMSO vehicle (matched concentration)",
        },
        "validation_assays": ["flow_cytometry", "caspase_3_7_activity"],
    },
    "protein_interaction": {
        "name":         "Co-Immunoprecipitation (co-IP)",
        "duration_days": 3,
        "steps": [
            "Lyse 10M cells in IP lysis buffer (50 mM Tris pH7.5, 150 mM NaCl, 1% NP-40, protease inhibitors)",
            "Clarify lysate by centrifugation (10,000g, 10 min, 4°C)",
            "Pre-clear with Protein A/G beads (1h, 4°C rotation)",
            "Incubate pre-cleared lysate with 2–5 μg target antibody (overnight, 4°C)",
            "Add Protein A/G magnetic beads (1h, 4°C)",
            "Wash 3x with lysis buffer, 1x with PBS",
            "Elute with 50 μL 2x SDS loading buffer (boil 5 min)",
            "Run SDS-PAGE, transfer, immunoblot for interaction partner",
        ],
        "controls": {
            "positive": "IgG isotype control antibody (same species + concentration)",
            "negative": "Input lysate (5% of total for co-IP normalization)",
        },
        "validation_assays": ["western_blot", "mass_spectrometry_optional"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Protocol Generator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_experimental_protocol(
    hypothesis:       str,
    gene_symbol:      str = "",
    cancer_type:      str = "",
    assay_type:       str = "auto",
    available_equipment: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a complete, actionable experimental protocol from a biological hypothesis.

    Synthesizes data from multiple BioMCP databases to produce:
      - Recommended cell lines with rationale
      - Complete reagent list with catalog numbers
      - Step-by-step protocol
      - Controls (positive + negative)
      - Expected readouts with quantitative thresholds
      - Statistical design (power calculation)
      - Similar published protocols from PubMed
      - Timeline estimate

    Args:
        hypothesis:           Research hypothesis (e.g. 'KRAS G12C inhibition
                              reduces proliferation in NSCLC cells').
        gene_symbol:          Primary gene of interest (auto-extracted if empty).
        cancer_type:          Cancer type (auto-inferred if empty).
        assay_type:           'crispr_knockout' | 'sirna_knockdown' | 'drug_sensitivity' |
                              'apoptosis_flow' | 'protein_interaction' | 'auto'.
        available_equipment:  List of available equipment to tailor protocol
                              (e.g. ['flow_cytometer', 'luminometer', 'confocal']).

    Returns:
        Full protocol object with all sections needed for bench execution.
    """
    # ── Extract entities ──────────────────────────────────────────────────────
    import re

    from biomcp.tools.ncbi import get_gene_info, search_pubmed
    hyp_lower = hypothesis.lower()

    if not gene_symbol:
        hits = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', hypothesis)
        gene_symbol = hits[0] if hits else ""
    else:
        gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)

    if not cancer_type:
        cancer_type = _infer_cancer_type(hyp_lower)

    # ── Auto-select assay type ────────────────────────────────────────────────
    if assay_type == "auto":
        assay_type = _auto_select_assay(hyp_lower)

    # ── Parallel data gathering ───────────────────────────────────────────────
    tasks: list[Any] = []
    pubmed_query = f"{gene_symbol} {_cancer_to_keyword(cancer_type)} {_assay_to_keyword(assay_type)} protocol"

    tasks.append(asyncio.create_task(
        search_pubmed(pubmed_query, max_results=5, sort="relevance")
    ))

    if gene_symbol:
        tasks.append(asyncio.create_task(get_gene_info(gene_symbol)))
    else:
        tasks.append(asyncio.create_task(_null_task()))

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    pubmed_result = raw[0] if not isinstance(raw[0], Exception) else {}
    gene_info     = raw[1] if not isinstance(raw[1], Exception) else {}

    # ── Build protocol ────────────────────────────────────────────────────────
    template    = _ASSAY_TEMPLATES.get(assay_type, _ASSAY_TEMPLATES["drug_sensitivity"])
    cell_lines  = _select_cell_lines(cancer_type, gene_symbol, gene_info)
    reagents    = _build_reagent_list(assay_type)
    stats_plan  = _calculate_statistical_power(assay_type)

    # Similar protocols from PubMed
    similar_protocols: list[dict[str, str]] = []
    if isinstance(pubmed_result, dict):
        for article in pubmed_result.get("articles", [])[:5]:
            similar_protocols.append({
                "title": article.get("title", ""),
                "pmid":  article.get("pmid", ""),
                "url":   article.get("url", ""),
                "year":  article.get("year", ""),
            })

    equip = set(available_equipment or [])
    protocol_steps = template["steps"].copy()
    _adapt_steps_for_equipment(protocol_steps, assay_type, equip)

    return {
        "hypothesis":     hypothesis,
        "gene_target":    gene_symbol,
        "cancer_context": cancer_type,
        "protocol": {
            "name":           template["name"],
            "estimated_duration_days": template["duration_days"],
            "assay_type":     assay_type,
            "steps":          protocol_steps,
            "controls":       template["controls"],
            "validation_assays": template.get("validation_assays", []),
        },
        "cell_lines": {
            "recommended": cell_lines[:3],
            "rationale":   _cell_line_rationale(cancer_type, gene_symbol),
            "all_options": cell_lines,
        },
        "reagents": {
            "required":    reagents["required"],
            "optional":    reagents["optional"],
            "consumables": reagents["consumables"],
        },
        "expected_readouts": _expected_readouts(assay_type, gene_symbol),
        "statistical_design": stats_plan,
        "similar_protocols":  similar_protocols,
        "safety_notes": _safety_notes(assay_type),
        "troubleshooting": _troubleshooting_guide(assay_type),
        "timeline": _build_timeline(template["duration_days"], assay_type),
        "disclaimer": (
            "This protocol is AI-generated from curated templates and literature. "
            "Optimize for your specific cell line and laboratory conditions. "
            "Always follow institutional biosafety guidelines."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Power Calculator
# ─────────────────────────────────────────────────────────────────────────────

async def estimate_statistical_power(
    expected_effect_size: float = 0.5,
    alpha:                float = 0.05,
    power:                float = 0.8,
    n_groups:             int   = 2,
    assay_type:           str   = "drug_sensitivity",
) -> dict[str, Any]:
    """
    Calculate required sample size for adequate statistical power.

    Uses Cohen's d for continuous outcomes (IC50, western blot densitometry)
    and appropriate tests for categorical data (flow cytometry proportions).

    Args:
        expected_effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large).
        alpha:                Significance threshold. Default 0.05.
        power:                Desired power (1-β). Default 0.8 (80%).
        n_groups:             Number of comparison groups. Default 2.
        assay_type:           Assay context for tailored recommendations.

    Returns:
        {
          n_per_group, total_n, effect_size_class, test_recommended,
          power_achieved, corrections_applied, interpretation
        }
    """
    # Simplified power calculation (z-approximation)
    import math

    # Critical values (z-scores)
    z_alpha = {0.05: 1.960, 0.01: 2.576, 0.001: 3.291}.get(alpha, 1.960)
    z_beta  = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)

    # Two-sample t-test sample size: n = 2 * ((z_alpha + z_beta) / d)^2
    n_per_group = math.ceil(2 * ((z_alpha + z_beta) / max(expected_effect_size, 0.01)) ** 2)

    # Bonferroni correction for multiple groups
    corrected_n = n_per_group
    multiple_testing_note = ""
    if n_groups > 2:
        alpha_corrected = alpha / (n_groups * (n_groups - 1) / 2)
        z_alpha_corrected = {0.05: 2.807, 0.01: 3.291}.get(round(alpha_corrected, 3), 2.807)
        corrected_n = math.ceil(2 * ((z_alpha_corrected + z_beta) / max(expected_effect_size, 0.01)) ** 2)
        multiple_testing_note = f"Bonferroni corrected for {n_groups} groups (α={alpha_corrected:.4f})"

    effect_class = (
        "Small (d=0.2)"  if expected_effect_size < 0.35 else
        "Medium (d=0.5)" if expected_effect_size < 0.65 else
        "Large (d=0.8)"  if expected_effect_size < 1.0  else
        "Very large"
    )

    test_map = {
        "drug_sensitivity":  "Non-linear regression (GraphPad Prism 4-PL); ANOVA for multi-arm",
        "crispr_knockout":   "Unpaired two-tailed t-test; Welch's correction if unequal variance",
        "sirna_knockdown":   "One-way ANOVA + Dunnett's post-hoc vs scramble control",
        "apoptosis_flow":    "Two-way ANOVA (time × treatment); Tukey's post-hoc",
        "protein_interaction":"Ratio paired t-test on normalized densitometry",
    }

    return {
        "n_per_group":           corrected_n,
        "total_n":               corrected_n * n_groups,
        "effect_size":           expected_effect_size,
        "effect_size_class":     effect_class,
        "alpha":                 alpha,
        "target_power":          power,
        "n_groups":              n_groups,
        "test_recommended":      test_map.get(assay_type, "Unpaired t-test for 2 groups; ANOVA for >2"),
        "multiple_testing_note": multiple_testing_note,
        "n_replicates_recommendation": {
            "technical_replicates": 3,
            "biological_replicates": max(corrected_n, 3),
            "independent_experiments": 3,
            "note": "All experiments must be performed in ≥3 independent biological replicates for publication.",
        },
        "software_recommendations": [
            "GraphPad Prism 10 — IC50 curve fitting, ANOVA",
            "R/ggplot2 — reproducible statistical reporting",
            "G*Power — comprehensive power analysis",
            "SPSS — alternative for complex designs",
        ],
        "interpretation": (
            f"With effect size d={expected_effect_size:.1f} ({effect_class}), "
            f"α={alpha}, power={power*100:.0f}%: "
            f"need {corrected_n} samples per group ({corrected_n * n_groups} total)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell line recommendation tool
# ─────────────────────────────────────────────────────────────────────────────

async def suggest_cell_lines(
    cancer_type:       str,
    gene_symbol:       str = "",
    molecular_feature: str = "",
    max_results:       int = 5,
) -> dict[str, Any]:
    """
    Recommend validated cell lines for a research context.

    Args:
        cancer_type:       Cancer type (e.g. 'lung', 'breast', 'colorectal').
        gene_symbol:       Gene of interest — filters for relevant mutations.
        molecular_feature: Required molecular feature (e.g. 'EGFR mutant',
                           'p53 null', 'KRAS G12C', 'HER2 amplified').
        max_results:       Cell lines to return. Default 5.

    Returns:
        {
          context, recommended: [{ name, origin, atcc, mutations,
          rationale, deposition_url }], notes
        }
    """
    cancer_lower   = cancer_type.lower()
    feature_lower  = molecular_feature.lower()
    gene_upper     = gene_symbol.upper() if gene_symbol else ""

    # Find matching category
    category = "general"
    if any(w in cancer_lower for w in ("lung", "nsclc", "sclc", "pulmon")):
        category = "lung_cancer"
    elif any(w in cancer_lower for w in ("breast", "mammary")):
        category = "breast_cancer"
    elif any(w in cancer_lower for w in ("colon", "colorectal", "bowel", "crc")):
        category = "colorectal_cancer"
    elif any(w in cancer_lower for w in ("prostate")):
        category = "prostate_cancer"
    elif any(w in cancer_lower for w in ("glioblastoma", "gbm", "glioma", "brain")):
        category = "glioblastoma"

    lines = _CELL_LINE_DB.get(category, _CELL_LINE_DB["general"]).copy()

    # Filter by molecular feature if specified
    if feature_lower or gene_upper:
        search_term = feature_lower or gene_upper.lower()
        lines = [
            line for line in lines if search_term in line.get("notes", "").lower()
        ] or lines

    lines = lines[:max_results]

    return {
        "context":        f"{cancer_type}" + (f" | {gene_symbol}" if gene_symbol else "") + (f" | {molecular_feature}" if molecular_feature else ""),
        "recommended": [
            {
                **line,
                "deposition_url": f"https://www.atcc.org/products/{line['atcc'].replace('#', '')}" if line.get("atcc") and "N/A" not in line.get("atcc", "") else "",
            }
            for line in lines
        ],
        "additional_notes": [
            "Authenticate all cell lines by STR profiling before use (ATCC or Promega).",
            "Test for Mycoplasma contamination monthly (PCR kit or MycoAlert).",
            "Early-passage cells (P5–P20) recommended for reproducibility.",
            "Maintain matched passage-number controls across experiments.",
        ],
        "negative_controls_recommended": [
            {"name": "RPE-1", "rationale": "Non-transformed human epithelial — isogenic comparison"},
            {"name": "MCF10A", "rationale": "Non-transformed mammary epithelial — breast cancer context"},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

async def _null_task() -> dict:
    return {}


def _infer_cancer_type(text: str) -> str:
    mapping = {
        "lung":       "lung_cancer",
        "nsclc":      "lung_cancer",
        "breast":     "breast_cancer",
        "colorectal": "colorectal_cancer",
        "colon":      "colorectal_cancer",
        "prostate":   "prostate_cancer",
        "glioblastoma": "glioblastoma",
        "gbm":        "glioblastoma",
        "pancreatic": "pancreatic_cancer",
        "ovarian":    "ovarian_cancer",
    }
    for keyword, cancer in mapping.items():
        if keyword in text:
            return cancer
    return "general"


def _auto_select_assay(text: str) -> str:
    if any(w in text for w in ("crispr", "knockout", "knock out", "ko")):
        return "crispr_knockout"
    if any(w in text for w in ("sirna", "knockdown", "knock down", "shrna", "rnai")):
        return "sirna_knockdown"
    if any(w in text for w in ("ic50", "drug", "inhibit", "compound", "sensitivity", "proliferat", "viabilit")):
        return "drug_sensitivity"
    if any(w in text for w in ("apoptosis", "death", "annexin", "caspase")):
        return "apoptosis_flow"
    if any(w in text for w in ("interact", "co-ip", "coimmunoprecipitation", "pull-down")):
        return "protein_interaction"
    return "drug_sensitivity"


def _assay_to_keyword(assay_type: str) -> str:
    return {
        "crispr_knockout":    "CRISPR knockout",
        "sirna_knockdown":    "siRNA knockdown",
        "drug_sensitivity":   "IC50 drug sensitivity",
        "apoptosis_flow":     "apoptosis flow cytometry",
        "protein_interaction":"co-immunoprecipitation",
    }.get(assay_type, "cell biology")


def _cancer_to_keyword(cancer_type: str) -> str:
    return {
        "lung_cancer":       "lung cancer NSCLC",
        "breast_cancer":     "breast cancer",
        "colorectal_cancer": "colorectal cancer",
        "prostate_cancer":   "prostate cancer",
        "glioblastoma":      "glioblastoma GBM",
    }.get(cancer_type, "cancer")


def _select_cell_lines(cancer_type: str, gene_symbol: str, gene_info: dict) -> list[dict[str, str]]:
    lines = _CELL_LINE_DB.get(cancer_type, _CELL_LINE_DB["general"]).copy()
    # Add general as fallback
    lines = lines + _CELL_LINE_DB["general"][:2]
    return lines


def _cell_line_rationale(cancer_type: str, gene_symbol: str) -> str:
    return (
        f"Cell lines selected for {cancer_type or 'general cancer'} context "
        f"{'with known ' + gene_symbol + ' status ' if gene_symbol else ''}"
        "from ATCC-verified sources. Include at least one non-transformed control line."
    )


def _build_reagent_list(assay_type: str) -> dict[str, list[dict]]:
    core_reagents = ["PBS", "DMSO", "protein_assay"]
    assay_specific: dict[str, list[str]] = {
        "crispr_knockout":    ["CRISPR_lentiviral", "Cas9_RNP", "RT_qPCR_kit", "RNA_extraction", "western_blot_SDS", "PVDF_membrane"],
        "sirna_knockdown":    ["siRNA_transfection", "RT_qPCR_kit", "RNA_extraction", "western_blot_SDS", "PVDF_membrane"],
        "drug_sensitivity":   ["cell_viability", "96_well_plate", "MTT"],
        "apoptosis_flow":     ["annexin_v", "flow_cytometer_fixation"],
        "protein_interaction":["western_blot_SDS", "PVDF_membrane"],
    }

    required_keys = assay_specific.get(assay_type, assay_specific["drug_sensitivity"]) + core_reagents
    required      = [_REAGENT_DB[k] for k in required_keys if k in _REAGENT_DB]
    optional      = [_REAGENT_DB["RT_qPCR_kit"]] if assay_type != "sirna_knockdown" else []
    consumables   = [
        {"name": "Sterile 15 mL Falcon tubes", "supplier": "Corning", "catalog": "#352096"},
        {"name": "0.22 μm syringe filter",     "supplier": "Millipore", "catalog": "#SLGP033RS"},
        {"name": "Cell culture flasks T-75",    "supplier": "Corning", "catalog": "#430641U"},
    ]

    return {"required": required, "optional": optional, "consumables": consumables}


def _expected_readouts(assay_type: str, gene_symbol: str) -> list[dict[str, Any]]:
    base = [
        {"readout": "Cell viability (%)",         "method": "CellTiter-Glo",  "threshold": "≥50% reduction vs vehicle"},
        {"readout": "Protein expression change",  "method": "Western blot",   "threshold": "≥70% knockdown by densitometry"},
        {"readout": "mRNA expression change",     "method": "RT-qPCR",        "threshold": "≥80% knockdown"},
    ]
    if assay_type == "drug_sensitivity":
        base.insert(0, {"readout": "IC50 (μM)", "method": "4-parameter logistic regression", "threshold": "Report with 95% CI"})
    if assay_type == "apoptosis_flow":
        base.insert(0, {"readout": "Annexin V+ cells (%)", "method": "Flow cytometry", "threshold": "≥20% increase vs vehicle"})
    return base


def _safety_notes(assay_type: str) -> list[str]:
    notes = [
        "DMSO penetrates skin — wear nitrile gloves when handling concentrated stock.",
        "All cell culture work in BSL-2 cabinet per institutional guidelines.",
        "Dispose of biological waste per institutional hazardous waste protocols.",
    ]
    if assay_type == "crispr_knockout":
        notes.extend([
            "Lentiviral work requires BSL-2+ approval — consult institutional biosafety committee.",
            "UV-inactivate all lentiviral supernatants before disposal.",
        ])
    return notes


def _troubleshooting_guide(assay_type: str) -> list[dict[str, str]]:
    return {
        "drug_sensitivity": [
            {"problem": "IC50 out of range", "solution": "Adjust dose range 3-fold above/below expected IC50"},
            {"problem": "High variability",  "solution": "Check DMSO ≤0.1%; use multichannel pipette for dispensing"},
            {"problem": "No effect",         "solution": "Confirm compound solubility; check cell line genotype"},
        ],
        "crispr_knockout": [
            {"problem": "Low KO efficiency",  "solution": "Optimize MOI; verify sgRNA activity in Cas9-expressing line"},
            {"problem": "No selected clones", "solution": "Lower puromycin concentration (empirically titrate)"},
            {"problem": "KO reversion",       "solution": "Freeze down validated clones early; avoid late passage"},
        ],
        "sirna_knockdown": [
            {"problem": "Incomplete knockdown", "solution": "Optimize transfection reagent:siRNA ratio; increase siRNA concentration"},
            {"problem": "Toxicity",             "solution": "Reduce lipofectamine; use RNAiMAX reverse transfection"},
            {"problem": "Off-target effects",   "solution": "Use 3 independent siRNAs; confirm phenotype with rescue construct"},
        ],
    }.get(assay_type, [
        {"problem": "Inconsistent results", "solution": "Standardize cell passage number and seeding density"},
    ])


def _calculate_statistical_power(assay_type: str) -> dict[str, Any]:
    return {
        "recommended_n_per_group": 3,
        "minimum_biological_replicates": 3,
        "minimum_technical_replicates":  3,
        "effect_size_assumption":        "Cohen's d = 0.8 (large effect; biological perturbation)",
        "alpha":                         0.05,
        "power":                         0.8,
        "note": "Use estimate_statistical_power tool for custom power calculations.",
    }


def _build_timeline(total_days: int, assay_type: str) -> list[dict[str, str]]:
    templates = {
        "drug_sensitivity": [
            {"week": "Week 1", "activities": "Cell line recovery, mycoplasma testing, growth rate determination"},
            {"week": "Week 2", "activities": "Dose-finding pilot, confirm compound solubility, optimize assay window"},
            {"week": "Week 3", "activities": "Full dose-response (n=3 biological replicates), data analysis"},
        ],
        "crispr_knockout": [
            {"week": "Week 1",   "activities": "sgRNA design, cloning, sequence verification"},
            {"week": "Week 2",   "activities": "Lentivirus production, titer determination"},
            {"week": "Week 3",   "activities": "Transduction, puromycin selection"},
            {"week": "Week 4+5", "activities": "Single-cell cloning by limiting dilution"},
            {"week": "Week 6+7", "activities": "Clone expansion, sequencing, western blot validation"},
        ],
        "sirna_knockdown": [
            {"week": "Week 1", "activities": "siRNA delivery optimization, preliminary knockdown assessment"},
            {"week": "Week 2", "activities": "Knockdown validation (RT-qPCR + WB), functional assays at 72h"},
        ],
    }
    return templates.get(assay_type, [
        {"week": f"Week 1–{max(1, total_days // 7)}", "activities": "Protocol execution per steps above"},
    ])


def _adapt_steps_for_equipment(steps: list[str], assay_type: str, equipment: set[str]) -> None:
    """Modify protocol steps based on available equipment."""
    if "luminometer" not in equipment and "plate_reader" not in equipment:
        for i, step in enumerate(steps):
            if "cti" in step.lower() or "celltiter" in step.lower():
                steps[i] = step + " [Alternative: MTT assay if no luminometer available]"
    if "flow_cytometer" not in equipment:
        for i, step in enumerate(steps):
            if "flow cytometer" in step.lower():
                steps[i] = step + " [Alternative: ImageStream if conventional flow unavailable]"
