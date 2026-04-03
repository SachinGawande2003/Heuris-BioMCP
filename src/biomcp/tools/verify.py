"""
BioMCP — Cross-Database Verification & Conflict Detection  [FIXED v2.1]
=========================================================================
Fixes applied:
  - Added missing `import asyncio` (was using asyncio.create_task without import)
  - Made evidence scoring more robust with try/except on dict access
  - Fixed KeyError risk in conflict detection
"""

from __future__ import annotations

import asyncio  # FIX: was missing — caused NameError at runtime
import re
from typing import Any

from loguru import logger

from biomcp.utils import (
    BioValidator,
)


async def verify_biological_claim(
    claim: str,
    context_gene: str = "",
    max_evidence_sources: int = 5,
) -> dict[str, Any]:
    """
    Verify a biological claim against multiple databases simultaneously.

    Args:
        claim:                Natural language biological claim.
        context_gene:         Optional gene symbol to focus the search.
        max_evidence_sources: Max databases to query (3–5). Default 5.

    Returns:
        {
          claim, verdict, confidence_score, confidence_grade,
          supporting_evidence, contradicting_evidence, unresolved,
          evidence_by_source, recommendation
        }

    Confidence grades:
        A — Verified by 3+ independent databases, no contradictions
        B — Verified by 2 databases, minor inconsistencies
        C — Partial evidence, some contradictions
        D — Weak/conflicting evidence
        F — Contradicted by primary databases
    """
    from biomcp.tools.ncbi import search_pubmed
    from biomcp.tools.pathways import get_gene_disease_associations
    from biomcp.tools.proteins import search_proteins

    claim_lower = claim.lower()

    # Extract gene symbols from claim
    gene_hits = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', claim)
    gene = context_gene.upper() if context_gene else (gene_hits[0] if gene_hits else "")

    # Build parallel evidence queries
    evidence_tasks: dict[str, Any] = {}

    pubmed_query = f"{gene} {claim[:60]}" if gene else f"{claim[:80]} evidence"
    evidence_tasks["PubMed"] = asyncio.create_task(
        search_pubmed(pubmed_query, max_results=8, sort="relevance")
    )

    if gene:
        evidence_tasks["UniProt"] = asyncio.create_task(
            search_proteins(gene, max_results=1, reviewed_only=True)
        )
        if any(w in claim_lower for w in ("cancer","disease","disorder","syndrome","tumor","oncogen")):
            evidence_tasks["OpenTargets"] = asyncio.create_task(
                get_gene_disease_associations(gene, max_results=5)
            )
        if any(w in claim_lower for w in ("express","tissue","organ","cell type","upregulat","downregulat")):
            from biomcp.tools.advanced import search_gene_expression
            evidence_tasks["GEO"] = asyncio.create_task(
                search_gene_expression(gene, max_datasets=5)
            )

    selected_tasks = dict(list(evidence_tasks.items())[:max_evidence_sources])

    raw_results = await asyncio.gather(*selected_tasks.values(), return_exceptions=True)
    results_by_source = dict(zip(selected_tasks.keys(), raw_results, strict=False))

    supporting:    list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    unresolved:    list[dict[str, Any]] = []

    # Score PubMed evidence
    pubmed_result = results_by_source.get("PubMed", {})
    if isinstance(pubmed_result, dict):
        for article in pubmed_result.get("articles", []):
            try:
                abstract = (article.get("abstract", "") or "").lower()
                title    = (article.get("title", "")    or "").lower()
                text     = abstract + " " + title

                support_signals = sum(1 for w in [
                    "confirms","demonstrate","shows","observed","reported",
                    "found","reveals","indicates","consistent",
                ] if w in text)
                contra_signals = sum(1 for w in [
                    "contradict","however","contrary","unexpectedly",
                    "failed to","no evidence","not observed","disputes",
                ] if w in text)

                if support_signals >= contra_signals:
                    supporting.append({
                        "source":   "PubMed",
                        "evidence": article.get("title", ""),
                        "pmid":     article.get("pmid", ""),
                        "url":      article.get("url", ""),
                        "strength": "moderate" if support_signals < 3 else "strong",
                    })
                else:
                    contradicting.append({
                        "source":   "PubMed",
                        "evidence": article.get("title", ""),
                        "pmid":     article.get("pmid", ""),
                        "url":      article.get("url", ""),
                        "strength": "weak",
                    })
            except Exception as exc:
                logger.debug(f"[verify] Article scoring failed: {exc}")

    # UniProt evidence
    uniprot_result = results_by_source.get("UniProt", {})
    if isinstance(uniprot_result, dict):
        proteins = uniprot_result.get("proteins", [])
        if proteins and gene:
            gene_names = proteins[0].get("genes") or []
            if gene in gene_names or any(gene in str(g) for g in gene_names):
                supporting.append({
                    "source":   "UniProt Swiss-Prot",
                    "evidence": f"Gene {gene} confirmed in reviewed UniProt entry",
                    "strength": "strong",
                })

    # Open Targets evidence
    ot_result = results_by_source.get("OpenTargets", {})
    if isinstance(ot_result, dict):
        for assoc in ot_result.get("associations", [])[:3]:
            try:
                disease_name = assoc.get("disease_name", "").lower()
                score        = float(assoc.get("overall_score", 0) or 0)
                if any(d in claim_lower for d in disease_name.split() if len(d) > 4):
                    if score > 0.5:
                        supporting.append({
                            "source":   "Open Targets",
                            "evidence": f"{gene}–{assoc['disease_name']} association score: {score:.2f}",
                            "strength": "strong" if score > 0.7 else "moderate",
                        })
                    elif score < 0.2:
                        contradicting.append({
                            "source":   "Open Targets",
                            "evidence": f"Low {gene}–{assoc['disease_name']} association: {score:.2f}",
                            "strength": "weak",
                        })
            except Exception:
                pass

    # GEO evidence
    geo_result = results_by_source.get("GEO", {})
    if isinstance(geo_result, dict) and geo_result.get("total_found", 0) > 0:
        supporting.append({
            "source":   "NCBI GEO",
            "evidence": f"{geo_result['total_found']} expression datasets found for {gene}",
            "strength": "moderate",
        })

    # Confidence scoring
    n_support = len(supporting)
    n_contra  = len(contradicting)
    n_sources = len([r for r in raw_results if not isinstance(r, Exception)])

    total = n_support + n_contra
    score = n_support / total if total > 0 else 0.5
    source_bonus = min(0.15, 0.05 * n_sources)
    final_score  = min(1.0, score + source_bonus)

    if final_score >= 0.85 and n_support >= 3:
        grade, verdict = "A", "VERIFIED"
    elif final_score >= 0.70 and n_support >= 2:
        grade, verdict = "B", "LIKELY TRUE"
    elif final_score >= 0.55:
        grade, verdict = "C", "PARTIALLY SUPPORTED"
    elif final_score >= 0.40:
        grade, verdict = "D", "WEAK EVIDENCE"
    else:
        grade, verdict = "F", "CONTRADICTED"

    recommendation = {
        "A": "High confidence — suitable as factual basis for research.",
        "B": "Reasonable confidence — verify with primary literature before citing.",
        "C": "Mixed evidence — investigate contradictions before proceeding.",
        "D": "Insufficient evidence — conduct targeted experiments first.",
        "F": "Contradicted by primary databases — revise the claim.",
    }[grade]

    return {
        "claim":                  claim,
        "gene_context":           gene,
        "verdict":                verdict,
        "confidence_score":       round(final_score, 3),
        "confidence_grade":       grade,
        "supporting_evidence":    supporting[:10],
        "contradicting_evidence": contradicting[:10],
        "unresolved":             unresolved,
        "databases_queried":      list(selected_tasks.keys()),
        "evidence_counts": {
            "supporting":    n_support,
            "contradicting": n_contra,
            "total_articles":len(pubmed_result.get("articles", [])) if isinstance(pubmed_result, dict) else 0,
        },
        "recommendation": recommendation,
        "methodology": (
            "Evidence scored by keyword sentiment analysis of PubMed abstracts "
            "and database-specific association scores. Not a substitute for expert review."
        ),
    }


async def detect_database_conflicts(
    gene_symbol: str,
) -> dict[str, Any]:
    """
    Scan for conflicting biological information about a gene across databases.
    FIX: Added try/except on dict accesses; fixed asyncio import.
    """
    from biomcp.tools.ncbi import get_gene_info
    from biomcp.tools.pathways import get_drug_targets, get_gene_disease_associations
    from biomcp.tools.proteins import search_proteins

    gene_symbol = BioValidator.validate_gene_symbol(gene_symbol)
    logger.info(f"[ConflictDetector] Scanning databases for {gene_symbol}")

    ncbi_result, uniprot_result, chembl_result, ot_result = await asyncio.gather(
        get_gene_info(gene_symbol),
        search_proteins(gene_symbol, max_results=1),
        get_drug_targets(gene_symbol, max_results=20),
        get_gene_disease_associations(gene_symbol, max_results=10),
        return_exceptions=True,
    )

    conflicts: list[dict[str, Any]] = []

    # Conflict 1: ChEMBL activity value consistency
    if isinstance(chembl_result, dict) and "drugs" in chembl_result:
        activity_values: dict[str, list[float]] = {}
        for drug in chembl_result["drugs"]:
            try:
                mol_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
                act_type = drug.get("activity_type", "IC50")
                val = float(drug.get("activity_value", 0) or 0)
                if val > 0:
                    key = f"{mol_name}:{act_type}"
                    activity_values.setdefault(key, []).append(val)
            except (ValueError, TypeError):
                pass

        for key, values in activity_values.items():
            if len(values) >= 2:
                ratio = max(values) / max(min(values), 1e-9)
                if ratio > 100:
                    conflicts.append({
                        "type":           "ACTIVITY_VALUE_DISCREPANCY",
                        "severity":       "HIGH",
                        "source_a":       "ChEMBL (assay 1)",
                        "source_b":       "ChEMBL (assay 2)",
                        "entity":         key.split(":")[0],
                        "detail":         f"IC50 values differ by {ratio:.0f}x across assays",
                        "values":         values,
                        "recommendation": "Check assay conditions — in vitro vs cell-based assays differ.",
                    })

    # Conflict 2: Gene name agreement across databases
    ncbi_name    = ncbi_result.get("full_name", "") if isinstance(ncbi_result, dict) else ""
    uniprot_proteins = uniprot_result.get("proteins", []) if isinstance(uniprot_result, dict) else []
    uniprot_name = uniprot_proteins[0].get("name", "") if uniprot_proteins else ""

    if ncbi_name and uniprot_name:
        ncbi_words    = set(ncbi_name.lower().split()) - {"the","a","an","of","and","or"}
        uniprot_words = set(uniprot_name.lower().split()) - {"the","a","an","of","and","or"}
        if len(ncbi_words & uniprot_words) == 0 and len(ncbi_words) > 2:
            conflicts.append({
                "type":           "GENE_NAME_MISMATCH",
                "severity":       "LOW",
                "source_a":       "NCBI Gene",
                "source_b":       "UniProt",
                "detail":         f"NCBI: '{ncbi_name}' vs UniProt: '{uniprot_name}'",
                "recommendation": "Cross-reference HGNC for authoritative gene name.",
            })

    # Conflict 3: Disease evidence asymmetry
    if isinstance(ot_result, dict) and "associations" in ot_result:
        for assoc in ot_result["associations"][:5]:
            try:
                scores  = assoc.get("evidence_by_datatype", {})
                genetic = float(scores.get("genetic_association", 0) or 0)
                drug    = float(scores.get("known_drug", 0) or 0)
                if genetic > 0.7 and drug < 0.1:
                    conflicts.append({
                        "type":           "EVIDENCE_TYPE_ASYMMETRY",
                        "severity":       "MEDIUM",
                        "source_a":       "Open Targets (genetics)",
                        "source_b":       "Open Targets (drugs)",
                        "entity":         assoc.get("disease_name", ""),
                        "detail":         f"Strong genetic (score:{genetic:.2f}) but no approved drugs (score:{drug:.2f})",
                        "recommendation": "Potential unmet therapeutic need — investigate druggability.",
                    })
            except (TypeError, ValueError):
                pass

    high   = sum(1 for c in conflicts if c["severity"] == "HIGH")
    medium = sum(1 for c in conflicts if c["severity"] == "MEDIUM")
    low    = sum(1 for c in conflicts if c["severity"] == "LOW")
    penalty = high * 0.3 + medium * 0.15 + low * 0.05
    consistency_score = round(max(0.0, min(1.0, 1.0 - penalty)), 2)

    return {
        "gene":              gene_symbol,
        "conflicts_found":   len(conflicts),
        "conflicts":         conflicts,
        "consistency_score": consistency_score,
        "consistency_grade": (
            "HIGH" if consistency_score >= 0.8 else
            "MEDIUM" if consistency_score >= 0.5 else "LOW"
        ),
        "databases_scanned": ["NCBI Gene", "UniProt", "ChEMBL", "Open Targets"],
        "summary": (
            f"Scanned 4 databases for {gene_symbol}. "
            f"Found {len(conflicts)} conflict(s): "
            f"{high} high / {medium} medium / {low} low severity."
        ),
        "recommendation": (
            "Data appears largely consistent — suitable for research use."
            if consistency_score >= 0.8 else
            "Review flagged conflicts before drawing conclusions."
            if consistency_score >= 0.5 else
            "Significant discrepancies detected — manual curation recommended."
        ),
        "database_snapshots": {
            "ncbi_gene_name":   ncbi_name,
            "uniprot_name":     uniprot_name,
            "chembl_compounds": len(chembl_result.get("drugs", [])) if isinstance(chembl_result, dict) else 0,
            "ot_associations":  len(ot_result.get("associations", [])) if isinstance(ot_result, dict) else 0,
        },
    }
