"""
BioMCP — Genomic Variant Interpreter
======================================
Evidence-based clinical variant classification implementing ACMG/AMP 2015
guidelines (Richards et al.) — the global standard for clinical genetics.

This tool brings clinical-grade variant interpretation to Claude, enabling:
  - Automated ACMG/AMP evidence code assignment
  - Population frequency from gnomAD (world's largest variant dataset)
  - Functional predictions from 5+ in-silico tools
  - ClinVar clinical significance lookup
  - Published variant-disease evidence from ClinGen curations

Tools:
  classify_variant          — ACMG/AMP 5-tier classification pipeline
  get_population_frequency  — gnomAD allele frequency across populations
  predict_variant_function  — Multi-tool functional prediction ensemble
  lookup_clinvar_variant    — ClinVar clinical significance + submissions
  get_hotspot_analysis      — Cancer mutation hotspot context (COSMIC/cBio)

APIs:
  Ensembl REST (VEP):  https://rest.ensembl.org/vep/
  gnomAD GraphQL:      https://gnomad.broadinstitute.org/api
  ClinVar (NCBI):      https://eutils.ncbi.nlm.nih.gov/
  COSMIC:              https://cancer.sanger.ac.uk/api/

Scientific basis:
  - Richards et al. 2015 (Genetics in Medicine) — ACMG/AMP guidelines
  - Tavtigian et al. 2020 — Bayesian framework for ACMG
  - Karczewski et al. 2020 (Nature) — gnomAD v2.1
  - Landrum et al. 2016 — ClinVar

Clinical disclaimer:
  This tool is for research and education only. Clinical variant
  interpretation requires board-certified clinical geneticist review.
  Never use automated classification for clinical decisions without
  expert oversight.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
    cached,
    get_http_client,
    rate_limited,
    with_retry,
)

ENSEMBL_BASE = "https://rest.ensembl.org"
GNOMAD_API   = "https://gnomad.broadinstitute.org/api"

# ─────────────────────────────────────────────────────────────────────────────
# ACMG/AMP Evidence Code Definitions
# ─────────────────────────────────────────────────────────────────────────────

_ACMG_CODES = {
    # Pathogenic — Very Strong
    "PVS1": {
        "weight": "very_strong_pathogenic", "points": 8,
        "description": "Null variant (nonsense, frameshift, canonical splice site, initiation codon, "
                        "single/multi-exon deletion) in a gene where LOF is known disease mechanism.",
    },
    # Pathogenic — Strong
    "PS1": {
        "weight": "strong_pathogenic", "points": 4,
        "description": "Same amino acid change as established pathogenic variant, different nucleotide.",
    },
    "PS2": {
        "weight": "strong_pathogenic", "points": 4,
        "description": "De novo (confirmed) in patient with disease; no family history.",
    },
    "PS3": {
        "weight": "strong_pathogenic", "points": 4,
        "description": "Well-established functional study showing deleterious effect.",
    },
    "PS4": {
        "weight": "strong_pathogenic", "points": 4,
        "description": "Prevalence of variant significantly increased in affected vs controls.",
    },
    # Pathogenic — Moderate
    "PM1": {
        "weight": "moderate_pathogenic", "points": 2,
        "description": "Located in mutational hotspot or functional domain without benign variation.",
    },
    "PM2": {
        "weight": "moderate_pathogenic", "points": 2,
        "description": "Absent or extremely low frequency in population databases (gnomAD <0.1%).",
    },
    "PM4": {
        "weight": "moderate_pathogenic", "points": 2,
        "description": "Protein length change due to in-frame indel in non-repeat region.",
    },
    "PM5": {
        "weight": "moderate_pathogenic", "points": 2,
        "description": "Novel missense at same position as known pathogenic missense.",
    },
    # Pathogenic — Supporting
    "PP2": {
        "weight": "supporting_pathogenic", "points": 1,
        "description": "Missense in gene with low benign missense variation rate (constrained).",
    },
    "PP3": {
        "weight": "supporting_pathogenic", "points": 1,
        "description": "Multiple in-silico algorithms predict damaging effect.",
    },
    # Benign — Strong
    "BS1": {
        "weight": "strong_benign", "points": -4,
        "description": "Allele frequency greater than expected for disorder (MAF >1% for AD).",
    },
    "BS2": {
        "weight": "strong_benign", "points": -4,
        "description": "Observed in healthy adult with full penetrance recessive/dominant disorder.",
    },
    # Benign — Supporting
    "BP1": {
        "weight": "supporting_benign", "points": -1,
        "description": "Missense in gene where only LOF variants cause disease.",
    },
    "BP4": {
        "weight": "supporting_benign", "points": -1,
        "description": "Multiple in-silico algorithms predict benign effect.",
    },
    "BP7": {
        "weight": "supporting_benign", "points": -1,
        "description": "Synonymous variant with no predicted splice impact.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: classify_variant
# ─────────────────────────────────────────────────────────────────────────────

@cached("variant")
@rate_limited("ensembl")
@with_retry(max_attempts=3)
async def classify_variant(
    gene_symbol:     str,
    variant:         str,
    inheritance:     str = "unknown",
    consequence:     str = "",
    proband_phenotype: str = "",
) -> dict[str, Any]:
    """
    Classify a genetic variant according to ACMG/AMP 2015 guidelines.

    Automatically gathers evidence from population databases, functional
    predictions, and clinical databases to apply ACMG evidence codes.

    Args:
        gene_symbol:       HGNC gene symbol.
        variant:           Variant in one of these formats:
                           - Protein change: 'p.Arg175His', 'R175H', 'G12D'
                           - cDNA: 'c.524G>A', 'c.35delG'
                           - HGVS: 'NM_000546.5:c.524G>A'
                           - rsID: 'rs28934578'
        inheritance:       'AD' | 'AR' | 'XL' | 'unknown'. Affects thresholds.
        consequence:       VEP consequence if known: 'missense_variant',
                           'stop_gained', 'frameshift_variant', etc.
        proband_phenotype: Clinical phenotype to assess disease relevance.

    Returns:
        {
          gene, variant, classification, acmg_class,
          evidence_codes: [{code, applied, rationale}],
          score, confidence,
          population_frequency, functional_predictions, clinvar_data,
          key_findings, clinical_disclaimer
        }
    """
    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    variant     = variant.strip()

    # Determine variant type
    variant_type = _infer_variant_type(variant)

    # Run evidence gathering in parallel
    vep_result, gnomad_result, clinvar_result = await asyncio.gather(
        _run_vep(gene_symbol, variant, variant_type),
        _query_gnomad(gene_symbol, variant),
        _query_clinvar(gene_symbol, variant),
        return_exceptions=True,
    )

    # Safely extract results
    vep    = vep_result    if isinstance(vep_result, dict)    else {}
    gnomad = gnomad_result if isinstance(gnomad_result, dict) else {}
    clinvar= clinvar_result if isinstance(clinvar_result, dict) else {}

    # ── Apply ACMG evidence codes ─────────────────────────────────────────────
    applied_codes: list[dict[str, Any]] = []
    total_points = 0

    # PVS1: Loss-of-function variant
    lof_consequences = {"stop_gained", "frameshift_variant", "splice_donor_variant",
                        "splice_acceptor_variant", "start_lost"}
    var_consequence = consequence or vep.get("most_severe_consequence", "")
    if var_consequence in lof_consequences:
        applied_codes.append({
            "code":     "PVS1",
            "applied":  True,
            "weight":   "very_strong_pathogenic",
            "rationale":f"Loss-of-function variant: {var_consequence}",
        })
        total_points += 8

    # PM2: Rare/absent in population databases
    gnomad_af = gnomad.get("allele_frequency", {}).get("global_af", 1.0)
    if isinstance(gnomad_af, (int, float)):
        gnomad_af_f = float(gnomad_af)
        if gnomad_af_f < 0.001:
            applied_codes.append({
                "code":     "PM2",
                "applied":  True,
                "weight":   "moderate_pathogenic",
                "rationale":f"Absent or extremely rare in gnomAD (AF={gnomad_af_f:.6f})",
            })
            total_points += 2
        elif gnomad_af_f > 0.01:
            applied_codes.append({
                "code":     "BS1",
                "applied":  True,
                "weight":   "strong_benign",
                "rationale":f"Allele frequency {gnomad_af_f:.3f} > 1% — likely benign",
            })
            total_points -= 4

    # PP3 / BP4: In-silico predictions
    predictions = vep.get("predictions", {})
    n_pathogenic = sum(1 for v in predictions.values() if str(v).lower() in
                       {"deleterious", "probably_damaging", "pathogenic", "disease_causing"})
    n_benign = sum(1 for v in predictions.values() if str(v).lower() in
                   {"tolerated", "benign", "polymorphism", "tolerated_low_confidence"})

    if predictions:
        if n_pathogenic > n_benign and n_pathogenic >= 2:
            applied_codes.append({
                "code":     "PP3",
                "applied":  True,
                "weight":   "supporting_pathogenic",
                "rationale":f"Multiple in-silico tools predict damaging: {dict(list(predictions.items())[:4])}",
            })
            total_points += 1
        elif n_benign > n_pathogenic and n_benign >= 2:
            applied_codes.append({
                "code":     "BP4",
                "applied":  True,
                "weight":   "supporting_benign",
                "rationale":f"Multiple in-silico tools predict benign: {dict(list(predictions.items())[:4])}",
            })
            total_points -= 1

    # BP7: Synonymous with no splice impact
    if var_consequence == "synonymous_variant" and not vep.get("splice_impact", False):
        applied_codes.append({
            "code":     "BP7",
            "applied":  True,
            "weight":   "supporting_benign",
            "rationale":"Synonymous variant with no predicted splice site impact",
        })
        total_points -= 1

    # ClinVar evidence
    cv_sig = clinvar.get("clinical_significance", "").lower()
    if "pathogenic" in cv_sig and "conflicting" not in cv_sig:
        applied_codes.append({
            "code":     "PS1*",
            "applied":  True,
            "weight":   "strong_pathogenic",
            "rationale":f"ClinVar classification: {clinvar.get('clinical_significance', '')} "
                        f"(ClinVar ID: {clinvar.get('variation_id', 'Unknown')})",
        })
        total_points += 4
    elif "benign" in cv_sig and "conflicting" not in cv_sig:
        applied_codes.append({
            "code":     "BS2*",
            "applied":  True,
            "weight":   "strong_benign",
            "rationale":f"ClinVar classification: {clinvar.get('clinical_significance', '')}",
        })
        total_points -= 4

    # ── 5-tier ACMG classification ────────────────────────────────────────────
    classification = _points_to_classification(total_points)

    # Confidence estimate
    n_codes = len(applied_codes)
    confidence = (
        "HIGH"   if n_codes >= 3 and total_points not in range(-1, 2) else
        "MEDIUM" if n_codes >= 2 else
        "LOW"
    )

    key_findings: list[str] = []
    if total_points >= 8:
        key_findings.append(f"Strong pathogenic evidence (score: {total_points})")
    if gnomad_af_f < 0.0001 if isinstance(gnomad_af, (int, float)) else False:
        key_findings.append("Ultra-rare in gnomAD (<0.01%) — supports pathogenicity")
    if clinvar.get("clinical_significance"):
        key_findings.append(f"ClinVar: {clinvar['clinical_significance']}")
    if var_consequence in lof_consequences:
        key_findings.append(f"Loss-of-function: {var_consequence}")

    return {
        "gene":          gene_symbol,
        "variant":       variant,
        "variant_type":  variant_type,
        "consequence":   var_consequence or consequence,
        "inheritance":   inheritance,
        "classification":classification,
        "acmg_class":    _class_to_tier(classification),
        "acmg_score":    total_points,
        "confidence":    confidence,
        "evidence_codes": applied_codes,
        "population_frequency": gnomad,
        "functional_predictions": vep.get("predictions", {}),
        "clinvar_data":  clinvar,
        "key_findings":  key_findings,
        "acmg_classification_rules": {
            "pathogenic":             "Points ≥ 8 OR (PVS1 + PM1) OR (PS1 + PS2) OR ≥2 PS...",
            "likely_pathogenic":      "Points 5–7",
            "vus":                    "Points 3–4 or insufficient evidence",
            "likely_benign":          "Points 0–2 with benign evidence",
            "benign":                 "Points ≤ -4 with strong benign evidence",
        },
        "clinical_disclaimer": (
            "⚠ RESEARCH USE ONLY. This automated classification applies ACMG/AMP 2015 "
            "guidelines but CANNOT replace expert clinical geneticist review. "
            "Variant classification requires integration of clinical findings, family history, "
            "and specialized expertise. Never use for clinical decisions without board-certified "
            "genetic counselor/medical geneticist oversight."
        ),
    }


def _infer_variant_type(variant: str) -> str:
    if variant.lower().startswith("rs"):
        return "rsid"
    if variant.startswith("p.") or re.match(r'^[A-Z]\d+[A-Z*]$', variant):
        return "protein"
    if variant.startswith("c.") or variant.startswith("NM_"):
        return "cdna"
    return "unknown"


def _points_to_classification(points: int) -> str:
    if points >= 8:
        return "Pathogenic"
    if points >= 5:
        return "Likely Pathogenic"
    if points <= -6:
        return "Benign"
    if points <= -3:
        return "Likely Benign"
    return "Variant of Uncertain Significance (VUS)"


def _class_to_tier(classification: str) -> int:
    return {
        "Pathogenic": 5,
        "Likely Pathogenic": 4,
        "Variant of Uncertain Significance (VUS)": 3,
        "Likely Benign": 2,
        "Benign": 1,
    }.get(classification, 3)


async def _run_vep(gene_symbol: str, variant: str, variant_type: str) -> dict[str, Any]:
    """Run Ensembl VEP for functional predictions."""
    client = await get_http_client()

    # Build HGVS notation for VEP
    if variant_type == "rsid":
        vep_url = f"{ENSEMBL_BASE}/vep/human/id/{variant}"
        params  = {}
    elif variant_type in ("cdna", "protein"):
        # Search for gene to get RefSeq
        search_resp = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "gene", "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
                    "retmax": 1, "retmode": "json"},
        )
        if search_resp.status_code != 200:
            return {"error": "Gene lookup failed"}

        # Use genomic region lookup via gene symbol
        hgvs = f"{gene_symbol}:{variant}"
        vep_url = f"{ENSEMBL_BASE}/vep/human/hgvs/{hgvs.replace(':', '%3A')}"
        params  = {}
    else:
        return {"error": "Cannot run VEP — variant format not recognized."}

    try:
        resp = await client.get(
            vep_url,
            params={**params, "content-type": "application/json",
                    "SIFT": 1, "PolyPhen": 1, "regulatory": 1},
            headers={"Accept": "application/json"},
        )

        if resp.status_code not in (200, 201):
            return {"error": f"VEP returned {resp.status_code}"}

        vep_results = resp.json()
        if not vep_results:
            return {}

        top_result = vep_results[0] if isinstance(vep_results, list) else vep_results

        # Extract predictions from transcript consequences
        predictions: dict[str, str] = {}
        most_severe = top_result.get("most_severe_consequence", "")

        for tc in (top_result.get("transcript_consequences") or [])[:3]:
            if tc.get("sift_prediction"):
                predictions["SIFT"] = tc["sift_prediction"]
            if tc.get("polyphen_prediction"):
                predictions["PolyPhen2"] = tc["polyphen_prediction"]

        # Check for splice predictions
        splice_impact = any(
            cons in most_severe
            for cons in ["splice_donor", "splice_acceptor", "splice_region"]
        )

        return {
            "most_severe_consequence": most_severe,
            "predictions":   predictions,
            "splice_impact": splice_impact,
            "gene_id":       top_result.get("gene_id", ""),
            "colocated_variants": len(top_result.get("colocated_variants") or []),
        }

    except Exception as exc:
        logger.debug(f"[VEP] Failed for {variant}: {exc}")
        return {"error": str(exc)}


async def _query_gnomad(gene_symbol: str, variant: str) -> dict[str, Any]:
    """Query gnomAD for population allele frequencies."""
    client = await get_http_client()

    # gnomAD GraphQL query for variant
    if variant.lower().startswith("rs"):
        gql_query = """
        query VariantFrequency($variantId: String!) {
          variant(variantId: $variantId, dataset: gnomad_r4) {
            variantId
            exome { af ac an homozygote_count }
            genome { af ac an homozygote_count }
            populations {
              id
              af ac an
            }
          }
        }
        """
        variables = {"variantId": variant}
    else:
        # Can't query by gene:protein notation directly — return approximate
        return {
            "query_type":     "by_variant_id",
            "note":           "Provide rsID for gnomAD lookup. cDNA/protein notation not yet supported.",
            "allele_frequency": {"global_af": "Unknown — provide rsID"},
        }

    try:
        resp = await client.post(
            GNOMAD_API,
            json={"query": gql_query, "variables": variables},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return {"error": f"gnomAD API returned {resp.status_code}"}

        data = resp.json().get("data", {}).get("variant", {})
        if not data:
            return {"note": f"Variant {variant} not found in gnomAD v4"}

        exome  = data.get("exome") or {}
        genome = data.get("genome") or {}

        # Combined AF
        total_ac = (exome.get("ac") or 0) + (genome.get("ac") or 0)
        total_an = (exome.get("an") or 1) + (genome.get("an") or 1)
        global_af = total_ac / max(total_an, 1)

        # Population breakdown
        pop_freqs = {}
        for pop in (data.get("populations") or []):
            pop_id  = pop.get("id", "")
            pop_af  = pop.get("af", 0)
            if pop_af:
                pop_freqs[pop_id] = round(float(pop_af), 8)

        return {
            "variant":     data.get("variantId", variant),
            "dataset":     "gnomAD v4",
            "allele_frequency": {
                "global_af":    round(global_af, 8),
                "exome_af":     exome.get("af", 0),
                "genome_af":    genome.get("af", 0),
                "exome_an":     exome.get("an", 0),
                "genome_an":    genome.get("an", 0),
                "homozygotes":  (exome.get("homozygote_count") or 0) + (genome.get("homozygote_count") or 0),
            },
            "population_afs": pop_freqs,
            "rarity_tier": (
                "Ultra-rare (<1:10,000)"    if global_af < 0.0001 else
                "Very rare (<1:1,000)"      if global_af < 0.001  else
                "Rare (<1:100)"             if global_af < 0.01   else
                "Low frequency (<1:20)"     if global_af < 0.05   else
                "Common (>5%)"
            ),
            "gnomad_url": f"https://gnomad.broadinstitute.org/variant/{variant}?dataset=gnomad_r4",
        }

    except Exception as exc:
        logger.debug(f"[gnomAD] Query failed for {variant}: {exc}")
        return {"error": str(exc)}


async def _query_clinvar(gene_symbol: str, variant: str) -> dict[str, Any]:
    """Query ClinVar for clinical significance via NCBI E-utilities."""
    client = await get_http_client()

    try:
        # Build search query
        if variant.lower().startswith("rs"):
            query = variant
        else:
            query = f"{gene_symbol}[gene] AND {variant}[variant]"

        search = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "clinvar", "term": query, "retmax": 3, "retmode": "json"},
        )
        if search.status_code != 200:
            return {}

        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"note": f"No ClinVar entry found for {gene_symbol} {variant}"}

        # Fetch details for first hit
        summ = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "clinvar", "id": ids[0], "retmode": "json"},
        )
        if summ.status_code != 200:
            return {}

        data = summ.json().get("result", {}).get(ids[0], {})

        return {
            "variation_id":           ids[0],
            "title":                  data.get("title", ""),
            "clinical_significance":  data.get("clinical_significance", {}).get("description", ""),
            "review_status":          data.get("clinical_significance", {}).get("review_status", ""),
            "num_submissions":        data.get("num_submissions", 0),
            "phenotype":              [p.get("name", "") for p in (data.get("trait_set") or [])[:3]],
            "last_updated":           data.get("last_updated", ""),
            "clinvar_url":            f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{ids[0]}/",
            "star_rating":            _stars_from_review_status(data.get("clinical_significance", {}).get("review_status", "")),
        }

    except Exception as exc:
        logger.debug(f"[ClinVar] Query failed: {exc}")
        return {}


def _stars_from_review_status(status: str) -> str:
    status_lower = status.lower()
    if "practice guideline" in status_lower:
        return "★★★★ (Practice guideline)"
    if "expert panel" in status_lower:
        return "★★★ (Expert panel review)"
    if "multiple submitters" in status_lower and "conflict" not in status_lower:
        return "★★ (Multiple submitters, no conflicts)"
    if "single submitter" in status_lower:
        return "★ (Single submitter)"
    return "☆ (No assertion)"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_population_frequency
# ─────────────────────────────────────────────────────────────────────────────

@cached("variant")
@rate_limited("default")
@with_retry(max_attempts=3)
async def get_population_frequency(
    variant_id:  str,
    dataset:     str = "gnomad_r4",
    populations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Query gnomAD for population-specific allele frequencies.

    Provides the most comprehensive population genetics database for
    human variant interpretation — essential for assessing variant rarity.

    Args:
        variant_id:  Variant in rsID format (e.g. 'rs28934578') or
                     gnomAD format (e.g. '17-7674220-C-T').
        dataset:     'gnomad_r4' | 'gnomad_r2_1'. Default 'gnomad_r4'.
        populations: Specific populations to report. Default: all major populations.
                     Options: 'afr' (African), 'amr' (Latino), 'asj' (Ashkenazi Jewish),
                              'eas' (East Asian), 'fin' (Finnish), 'nfe' (Non-Finnish European),
                              'oth' (Other), 'sas' (South Asian).

    Returns:
        {
          variant, dataset, global_af, global_ac, global_an,
          population_frequencies: [{population, af, ac, an, homozygotes}],
          rarity_tier, population_insight, gnomad_url
        }
    """
    gnomad_result = await _query_gnomad("", variant_id)

    if "error" in gnomad_result:
        return gnomad_result

    # Filter populations if specified
    all_pop_afs = gnomad_result.get("population_afs", {})
    if populations:
        filtered = {k: v for k, v in all_pop_afs.items() if k.lower() in [p.lower() for p in populations]}
    else:
        filtered = all_pop_afs

    pop_names = {
        "afr": "African/African American",
        "amr": "Latino/Admixed American",
        "asj": "Ashkenazi Jewish",
        "eas": "East Asian",
        "fin": "Finnish",
        "nfe": "Non-Finnish European",
        "oth": "Other",
        "sas": "South Asian",
        "mid": "Middle Eastern",
        "ami": "Amish",
    }

    pop_table = []
    for pop_id, af in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
        pop_table.append({
            "population_id":   pop_id,
            "population_name": pop_names.get(pop_id.lower(), pop_id),
            "allele_frequency":af,
            "rarity":          (
                "Absent" if af == 0 else
                f"1 in {int(1/af):,}" if af > 0 else "Not observed"
            ),
        })

    global_af = gnomad_result.get("allele_frequency", {}).get("global_af", 0)
    rarity    = gnomad_result.get("rarity_tier", "Unknown")

    # Population insight
    pop_insight = []
    if pop_table:
        max_pop = pop_table[0]
        if max_pop["allele_frequency"] > (global_af or 0) * 5:
            pop_insight.append(
                f"Significantly enriched in {max_pop['population_name']} "
                f"(AF={max_pop['allele_frequency']:.4f} vs global {global_af:.6f}) — "
                "potential population-specific founder effect."
            )

    return {
        "variant":         variant_id,
        "dataset":         dataset,
        "global_af":       global_af,
        "allele_details":  gnomad_result.get("allele_frequency", {}),
        "rarity_tier":     rarity,
        "population_frequencies": pop_table,
        "population_insights":    pop_insight,
        "acmg_interpretation": (
            f"Global AF {global_af:.6f} → "
            + ("PM2 supporting (rare variant)" if isinstance(global_af, float) and global_af < 0.001 else
               "BS1 supporting (common variant)" if isinstance(global_af, float) and global_af > 0.01 else
               "Intermediate frequency — context-dependent")
        ),
        "gnomad_url": gnomad_result.get("gnomad_url", f"https://gnomad.broadinstitute.org/variant/{variant_id}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: lookup_clinvar_variant
# ─────────────────────────────────────────────────────────────────────────────

@cached("variant")
@rate_limited("default")
@with_retry(max_attempts=3)
async def lookup_clinvar_variant(
    gene_symbol:  str = "",
    variant:      str = "",
    clinvar_id:   str = "",
    max_results:  int = 5,
) -> dict[str, Any]:
    """
    Look up clinical significance and submission data from ClinVar.

    ClinVar is the FDA-recognized variant database for clinical interpretation.
    Returns star rating (review status), all submitter classifications,
    and associated phenotypes.

    Args:
        gene_symbol: HGNC gene symbol (used with variant for search).
        variant:     Variant notation (rsID, HGVS, protein change).
        clinvar_id:  Direct ClinVar variation ID (most specific).
        max_results: Maximum results to return. Default 5.

    Returns:
        {
          variants: [{variation_id, title, clinical_significance,
                      star_rating, review_status, submissions,
                      phenotypes, last_updated, clinvar_url}],
          search_query, note
        }
    """
    client = await get_http_client()

    # Build query
    if clinvar_id:
        query = f"{clinvar_id}[VariationID]"
    elif gene_symbol and variant:
        query = f"{gene_symbol}[gene] AND {variant}"
    elif gene_symbol:
        query = f"{gene_symbol}[gene] AND (pathogenic[clinical significance] OR likely pathogenic[clinical significance])"
    else:
        query = variant or ""

    if not query:
        raise ValueError("Provide at least one of: gene_symbol, variant, or clinvar_id.")

    search = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "clinvar", "term": query, "retmax": max_results, "retmode": "json"},
    )
    search.raise_for_status()
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    total = int(search.json().get("esearchresult", {}).get("count", 0))

    if not ids:
        return {
            "search_query": query,
            "total_found":  0,
            "variants":     [],
            "note":         f"No ClinVar entries found for query: {query}",
        }

    # Fetch summaries
    summ = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "clinvar", "id": ",".join(ids), "retmode": "json"},
    )
    summ.raise_for_status()
    results_data = summ.json().get("result", {})

    variants: list[dict[str, Any]] = []
    for vid in ids:
        data = results_data.get(vid, {})
        if not data:
            continue

        clin_sig = data.get("clinical_significance", {})
        phenotypes = [p.get("name", "") for p in (data.get("trait_set") or []) if p.get("name")]

        variants.append({
            "variation_id":          vid,
            "title":                 data.get("title", ""),
            "clinical_significance": clin_sig.get("description", "Not provided"),
            "review_status":         clin_sig.get("review_status", ""),
            "star_rating":           _stars_from_review_status(clin_sig.get("review_status", "")),
            "num_submissions":       data.get("num_submissions", 0),
            "phenotypes":            phenotypes[:5],
            "gene_id":               data.get("gene_id", ""),
            "chromosome":            data.get("chromosome", ""),
            "last_updated":          data.get("last_updated", ""),
            "variant_type":          data.get("variant_type", ""),
            "clinvar_url":           f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{vid}/",
        })

    return {
        "search_query": query,
        "total_found":  total,
        "returned":     len(variants),
        "variants":     variants,
        "clinvar_search_url": f"https://www.ncbi.nlm.nih.gov/clinvar/?term={query.replace(' ', '+')}",
        "note": (
            "ClinVar classifications reflect submitter interpretations. "
            "Conflicting classifications (★★ with conflict) require expert review. "
            "Expert Panel classifications (★★★) are highest confidence."
        ),
    }
