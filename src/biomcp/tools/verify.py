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

_CHEMBL_ASSAY_TYPE_LABELS = {
    "A": "ADMET assay",
    "B": "binding assay",
    "F": "functional assay",
    "P": "physicochemical assay",
    "T": "toxicity assay",
    "U": "unclassified assay",
}

_PUBMED_SUPPORT_PATTERNS = (
    "confirm",
    "demonstrat",
    "reveal",
    "support",
    "oncogenic",
    "driver mutation",
    "drives",
    "promotes",
    "required for",
    "essential for",
    "dependency",
    "activating mutation",
)

_PUBMED_CONTRADICTION_PATTERNS = (
    "no evidence",
    "not associated",
    "no association",
    "did not observe",
    "did not support",
    "failed to demonstrate",
    "failed to show",
    "not required",
    "dispensable",
    "does not drive",
    "independent of",
)

_PUBMED_RESISTANCE_PATTERNS = (
    "mechanism of resistance",
    "mechanisms of resistance",
    "acquired resistance",
    "drug resistance",
    "resistance to",
    "resistant to",
    "bypass",
    "escape",
    "adaptive resistance",
    "feedback activation",
)

_CLAIM_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "their",
    "drives", "drive", "progression", "cancer", "disease", "disorder", "syndrome",
    "tumor", "tumour", "gene", "mutation", "variant", "pathway", "protein",
}

_CLAIM_LANGUAGE_ALIASES = {
    "nsclc": "non small cell lung cancer",
    "tnbc": "triple negative breast cancer",
    "gbm": "glioblastoma",
}

_PROTEIN_VARIANT_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "TER": "*",
}


def _describe_assay_type(assay_type: str) -> str:
    if not assay_type:
        return "unspecified assay"
    return _CHEMBL_ASSAY_TYPE_LABELS.get(assay_type.upper(), assay_type)


def _claim_focus_terms(claim_lower: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", claim_lower)
    return {
        token
        for token in tokens
        if len(token) > 3 and token not in _CLAIM_STOPWORDS
    }


def _normalize_claim_language(claim: str) -> str:
    normalized = claim.lower()
    for alias, replacement in _CLAIM_LANGUAGE_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(alias)}\b", replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_protein_variant(variant: str) -> str:
    cleaned = variant.strip().replace(" ", "")
    if not cleaned:
        return ""
    if cleaned.lower().startswith("p."):
        cleaned = cleaned[2:]

    rsid_match = re.fullmatch(r"(rs\d+)", cleaned, flags=re.IGNORECASE)
    if rsid_match:
        return rsid_match.group(1).lower()

    three_letter = re.fullmatch(r"([A-Za-z]{3})(\d+)([A-Za-z]{3}|Ter)", cleaned)
    if three_letter:
        ref = _PROTEIN_VARIANT_THREE_TO_ONE.get(three_letter.group(1).upper(), "")
        alt = _PROTEIN_VARIANT_THREE_TO_ONE.get(three_letter.group(3).upper(), "")
        if ref and alt:
            return f"{ref}{three_letter.group(2)}{alt}"

    one_letter = re.fullmatch(r"([A-Za-z*])(\d+)([A-Za-z*])", cleaned, flags=re.IGNORECASE)
    if one_letter:
        return (
            f"{one_letter.group(1).upper()}"
            f"{one_letter.group(2)}"
            f"{one_letter.group(3).upper()}"
        )

    return cleaned.upper()


def _extract_variant_from_claim(claim: str) -> str:
    patterns = (
        r"\b(rs\d+)\b",
        r"\bp\.([A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|Ter))\b",
        r"\b([A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|Ter))\b",
        r"\b([A-Z]\d+[A-Z*])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, claim, flags=re.IGNORECASE)
        if match:
            return _normalize_protein_variant(match.group(1))
    return ""


def _extract_disease_focus(claim_lower: str) -> tuple[str, list[str]]:
    disease_suffixes = (
        "cancer",
        "disease",
        "disorder",
        "syndrome",
        "carcinoma",
        "lymphoma",
        "leukemia",
        "melanoma",
        "glioblastoma",
        "sarcoma",
        "metaplasia",
        "myeloma",
        "tumor",
        "tumour",
    )

    focus = ""
    contextual_matches = re.findall(
        rf"(?:in|for|with|associated with|linked to|risk of|drives|driver in)\s+"
        rf"([a-z0-9\-\s]{{3,60}}?(?:{'|'.join(disease_suffixes)}))",
        claim_lower,
    )
    if contextual_matches:
        focus = max((match.strip() for match in contextual_matches), key=len, default="")
    if not focus:
        generic_matches = re.findall(
            rf"([a-z0-9\-\s]{{0,40}}(?:{'|'.join(disease_suffixes)}))",
            claim_lower,
        )
        focus = max((match.strip() for match in generic_matches), key=len, default="")

    terms = [
        token
        for token in _claim_focus_terms(focus)
        if token not in {"small", "cell", "negative"}
    ]
    return focus, terms


def _classify_claim_relation(claim_lower: str) -> str:
    if any(token in claim_lower for token in ("overexpress", "upregulat", "downregulat", "expression")):
        return "expression"
    if any(token in claim_lower for token in ("resistance", "resistant", "sensitive to", "response", "survival")):
        return "therapeutic_response"
    if any(token in claim_lower for token in ("pathogenic", "benign", "deleterious", "disease-causing")):
        return "pathogenicity"
    if any(token in claim_lower for token in ("oncogenic", "driver", "tumorigen", "tumourigen", "transformation", "drives")):
        return "oncogenicity"
    if any(token in claim_lower for token in ("associated with", "linked to", "risk of", "predispos", "causes")):
        return "disease_association"
    return "biological_association"


def _decompose_biological_claim(claim: str, context_gene: str = "") -> dict[str, Any]:
    normalized_claim = _normalize_claim_language(claim)
    gene_hits = re.findall(r"\b([A-Z][A-Z0-9]{1,9})\b", claim)
    gene = context_gene.upper() if context_gene else (gene_hits[0] if gene_hits else "")
    variant = _extract_variant_from_claim(claim)
    disease_focus, disease_terms = _extract_disease_focus(normalized_claim)
    relation_type = _classify_claim_relation(normalized_claim)
    return {
        "claim": claim,
        "normalized_claim": normalized_claim,
        "gene": gene,
        "variant": variant,
        "disease_focus": disease_focus,
        "disease_terms": disease_terms,
        "relation_type": relation_type,
        "focus_terms": sorted(_claim_focus_terms(normalized_claim)),
    }


def _variant_context_matches(text: str, variant: str) -> bool:
    if not variant:
        return True
    normalized_variant = _normalize_protein_variant(variant)
    if not normalized_variant:
        return True
    candidates = {
        normalized_variant.lower(),
        f"p.{normalized_variant.lower()}",
    }
    if normalized_variant.endswith("*"):
        candidates.add(normalized_variant[:-1].lower() + "ter")
    return any(candidate in text for candidate in candidates)


def _evidence_strength_label(score: float) -> str:
    if score >= 0.8:
        return "strong"
    if score >= 0.55:
        return "moderate"
    return "weak"


def _build_evidence_item(
    *,
    source: str,
    evidence: str,
    stance: str,
    score: float,
    rationale: str,
    relation_type: str,
    **extra: Any,
) -> dict[str, Any]:
    item = {
        "source": source,
        "evidence": evidence,
        "strength": _evidence_strength_label(score),
        "score": round(max(0.0, min(1.0, score)), 3),
        "stance": stance,
        "rationale": rationale,
        "relation_type": relation_type,
    }
    item.update(extra)
    return item


def _pubmed_relation_patterns(relation_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    support_patterns = {
        "oncogenicity": (
            "oncogenic",
            "driver mutation",
            "drives",
            "tumor growth",
            "tumour growth",
            "tumorigen",
            "tumourigen",
            "promotes proliferation",
            "activating mutation",
        ),
        "pathogenicity": (
            "pathogenic",
            "likely pathogenic",
            "disease-causing",
            "deleterious",
            "causal variant",
        ),
        "disease_association": (
            "associated with",
            "association with",
            "linked to",
            "risk of",
            "predisposes",
        ),
        "expression": (
            "overexpress",
            "upregulat",
            "downregulat",
            "differential expression",
        ),
        "therapeutic_response": (
            "responds to",
            "sensitive to",
            "progression-free survival",
            "objective response",
            "resistance to",
        ),
        "biological_association": _PUBMED_SUPPORT_PATTERNS,
    }
    contradiction_patterns = {
        "oncogenicity": (
            "not oncogenic",
            "passenger mutation",
            "does not drive",
            "dispensable",
        ),
        "pathogenicity": (
            "benign",
            "likely benign",
            "uncertain significance",
            "no pathogenicity",
        ),
        "disease_association": (
            "no association",
            "not associated",
            "no increased risk",
        ),
        "expression": (
            "not differentially expressed",
            "no differential expression",
            "unchanged expression",
        ),
        "therapeutic_response": (
            "did not improve",
            "no response",
            "lacked response",
        ),
        "biological_association": _PUBMED_CONTRADICTION_PATTERNS,
    }
    support = tuple(dict.fromkeys((*support_patterns.get(relation_type, ()), *_PUBMED_SUPPORT_PATTERNS)))
    contradiction = tuple(
        dict.fromkeys((*contradiction_patterns.get(relation_type, ()), *_PUBMED_CONTRADICTION_PATTERNS))
    )
    return support, contradiction


def _classify_pubmed_claim_evidence(
    *,
    decomposition: dict[str, Any],
    article: dict[str, Any],
) -> tuple[str, float, str]:
    text = " ".join(
        segment.strip().lower()
        for segment in (
            article.get("title", "") or "",
            article.get("abstract", "") or "",
        )
        if segment
    ).strip()

    if not text:
        return "unresolved", 0.25, "Article text is unavailable for structured scoring."

    gene = str(decomposition.get("gene", ""))
    variant = str(decomposition.get("variant", ""))
    disease_terms = decomposition.get("disease_terms", [])
    relation_type = str(decomposition.get("relation_type", "biological_association"))

    has_gene_context = not gene or gene.lower() in text
    has_variant_context = _variant_context_matches(text, variant)
    matched_disease_terms = [term for term in disease_terms if term in text]
    has_disease_context = not disease_terms or bool(matched_disease_terms)
    focus_terms = decomposition.get("focus_terms", [])
    matched_focus_terms = [term for term in focus_terms if term in text]

    if any(pattern in text for pattern in _PUBMED_RESISTANCE_PATTERNS) and relation_type != "therapeutic_response":
        return (
            "unresolved",
            0.35,
            "Resistance or escape-mechanism context does not directly negate the decomposed claim.",
        )

    if not has_gene_context:
        return "unresolved", 0.2, "The article does not mention the claim gene explicitly."
    if variant and not has_variant_context:
        return "unresolved", 0.22, "The article mentions the gene but not the specific variant."

    support_patterns, contradiction_patterns = _pubmed_relation_patterns(relation_type)
    support_hits = [pattern for pattern in support_patterns if pattern in text]
    contradiction_hits = [pattern for pattern in contradiction_patterns if pattern in text]

    if contradiction_hits and (
        has_disease_context
        or any(hit in {"no evidence", "not associated", "no association"} for hit in contradiction_hits)
    ):
        return (
            "contradicting",
            0.78 if len(contradiction_hits) >= 2 else 0.66,
            f"Relation-specific contradiction cues detected: {', '.join(contradiction_hits[:2])}.",
        )

    if support_hits and has_disease_context:
        rationale = f"Relation-specific support cues detected: {', '.join(support_hits[:2])}."
        if matched_disease_terms:
            rationale += f" Disease context matched ({', '.join(matched_disease_terms[:3])})."
        return (
            "supporting",
            0.84 if len(support_hits) >= 2 else 0.72,
            rationale,
        )

    if matched_focus_terms or matched_disease_terms:
        return (
            "unresolved",
            0.38,
            "The paper overlaps the claim entities but does not make the target relation explicit.",
        )

    return (
        "unresolved",
        0.25,
        "The article does not provide explicit evidence for the decomposed claim relation.",
    )


def _text_matches_disease_terms(text: str, disease_terms: list[str]) -> bool:
    return not disease_terms or any(term in text for term in disease_terms)


def synthesize_conflicting_evidence(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Explain why multiple records disagree rather than only flagging a discrepancy.

    The function accepts a list of structured result fragments for a single conflict.
    It returns a compact reasoning payload that can be attached to conflict reports.
    """
    if not tool_results:
        return {
            "summary": "No evidence records were provided for synthesis.",
            "likely_causes": [],
            "reasoning_steps": [],
            "confidence": "low",
        }

    if all("activity_value" in result for result in tool_results):
        numeric_values: list[float] = []
        for result in tool_results:
            try:
                value = float(result.get("activity_value", 0) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                numeric_values.append(value)
        assay_types = sorted({
            _describe_assay_type(str(result.get("assay_type", "")))
            for result in tool_results
            if result.get("assay_type")
        })
        activity_types = sorted({
            str(result.get("activity_type", "IC50"))
            for result in tool_results
            if result.get("activity_type")
        })
        relations = sorted({
            str(result.get("activity_relation", ""))
            for result in tool_results
            if result.get("activity_relation")
        })
        units = sorted({
            str(result.get("activity_units", ""))
            for result in tool_results
            if result.get("activity_units")
        })
        years = sorted({
            int(str(result.get("document_year", "")))
            for result in tool_results
            if str(result.get("document_year", "")).isdigit()
        })

        value_min = min(numeric_values) if numeric_values else 0.0
        value_max = max(numeric_values) if numeric_values else 0.0
        ratio = value_max / max(value_min, 1e-9) if numeric_values else 1.0
        likely_causes: list[str] = []
        reasoning_steps: list[dict[str, str]] = []

        if assay_types:
            reasoning_steps.append({
                "dimension": "assay_type",
                "observation": ", ".join(assay_types),
                "implication": (
                    "Different assay modalities often produce materially different potency values."
                    if len(assay_types) > 1
                    else "All records share the same assay modality, so other factors likely drive the spread."
                ),
            })
            if len(assay_types) > 1:
                likely_causes.append(
                    f"Assay modality differs across records ({', '.join(assay_types)})."
                )

        if numeric_values:
            range_units = units[0] if len(units) == 1 else "mixed units"
            reasoning_steps.append({
                "dimension": "concentration_range",
                "observation": f"{value_min:g} to {value_max:g} {range_units}",
                "implication": (
                    f"Potency spans roughly {ratio:.0f}x, which is large enough to reflect context-specific assay behavior."
                    if ratio >= 100
                    else "Potency spread is present but modest."
                ),
            })
            if ratio >= 100:
                likely_causes.append(
                    f"Reported {activity_types[0] if activity_types else 'activity'} values span roughly {ratio:.0f}x."
                )

        if relations:
            reasoning_steps.append({
                "dimension": "activity_relation",
                "observation": ", ".join(relations),
                "implication": (
                    "Some results are bounded ('>' or '<') rather than exact measurements."
                    if any(rel in relations for rel in (">", "<", ">=", "<="))
                    else "Measurements are exact comparisons."
                ),
            })
            if any(rel in relations for rel in (">", "<", ">=", "<=")):
                likely_causes.append(
                    "At least one potency value is a bound rather than an exact endpoint."
                )

        if years:
            reasoning_steps.append({
                "dimension": "study_vintage",
                "observation": ", ".join(str(year) for year in years),
                "implication": (
                    "Protocol drift over time can change assay sensitivity and reported potency."
                    if len(years) > 1
                    else "All records come from the same study vintage."
                ),
            })
            if len(years) > 1:
                likely_causes.append("Measurements come from different publication years and likely different protocols.")

        likely_causes.append(
            "Cell-line context is not exposed in the current cached ChEMBL summary; inspect raw assay records if you need per-cell-line attribution."
        )

        confidence = "high" if len(reasoning_steps) >= 3 else "moderate"
        return {
            "summary": (
                f"Conflicting potency values are most likely driven by assay context rather than a true contradiction. "
                f"The records cover {len(tool_results)} assay measurements"
                + (f" across {len(assay_types)} assay modalities." if assay_types else ".")
            ),
            "likely_causes": likely_causes,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
        }

    record_type = str(tool_results[0].get("record_type", "generic"))
    if record_type == "name_alignment":
        values = [
            f"{item.get('source', 'source')}: {item.get('value', '')}"
            for item in tool_results
            if item.get("value")
        ]
        return {
            "summary": "Gene and protein resources often prefer different naming conventions and curation labels.",
            "likely_causes": [
                "NCBI and UniProt may emphasize different synonyms or full-name conventions.",
                "This usually reflects nomenclature drift rather than a biological contradiction.",
            ],
            "reasoning_steps": [
                {
                    "dimension": "source_labels",
                    "observation": "; ".join(values),
                    "implication": "Cross-check HGNC-approved names to normalize the label set.",
                }
            ],
            "confidence": "moderate",
        }

    if record_type == "evidence_asymmetry":
        genetics = next((item for item in tool_results if item.get("channel") == "genetic_association"), {})
        drugs = next((item for item in tool_results if item.get("channel") == "known_drug"), {})
        return {
            "summary": "This is more likely a translational gap than a contradiction: genetics supports the target, but drug evidence lags.",
            "likely_causes": [
                "Human genetic evidence can accumulate before tractable compounds or approved drugs exist.",
                "The target may be biologically validated but still hard to drug.",
            ],
            "reasoning_steps": [
                {
                    "dimension": "genetic_association",
                    "observation": str(genetics.get("score", "")),
                    "implication": "Strong human genetics increases confidence that the disease link is real.",
                },
                {
                    "dimension": "known_drug",
                    "observation": str(drugs.get("score", "")),
                    "implication": "Weak drug evidence points to a therapeutic-development gap, not necessarily conflicting biology.",
                },
            ],
            "confidence": "moderate",
        }

    return {
        "summary": "The records disagree, but the current metadata is too thin to attribute the discrepancy precisely.",
        "likely_causes": [
            "Source-specific curation choices may differ.",
            "Additional assay-level metadata is required to explain the discrepancy confidently.",
        ],
        "reasoning_steps": [],
        "confidence": "low",
    }


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
    from biomcp.tools.advanced import search_gene_expression
    from biomcp.tools.innovations import get_cancer_hotspots
    from biomcp.tools.ncbi import search_pubmed
    from biomcp.tools.pathways import get_gene_disease_associations
    from biomcp.tools.proteins import search_proteins
    from biomcp.tools.variant_interpreter import (
        classify_variant,
        get_population_frequency,
        lookup_clinvar_variant,
    )

    max_evidence_sources = BioValidator.clamp_int(max_evidence_sources, 1, 8, "max_evidence_sources")
    decomposition = _decompose_biological_claim(claim, context_gene)
    gene = str(decomposition.get("gene", ""))
    variant = str(decomposition.get("variant", ""))
    relation_type = str(decomposition.get("relation_type", "biological_association"))
    disease_focus = str(decomposition.get("disease_focus", ""))
    disease_terms = [
        str(term)
        for term in decomposition.get("disease_terms", [])
        if isinstance(term, str) and term
    ]

    queryable_population_variant = bool(
        variant and (
            variant.startswith("rs")
            or bool(re.fullmatch(r"\d+[-:][A-Za-z*]+[-:][A-Za-z*]+", variant))
        )
    )

    provider_requests: list[tuple[str, Any, str]] = []
    pubmed_query = (
        " ".join(part for part in [gene, variant, disease_focus] if part).strip()
        or claim[:120]
    )
    provider_requests.append((
        "PubMed",
        search_pubmed(f"{pubmed_query} evidence", max_results=8, sort="relevance"),
        "Primary literature remains the broadest source of mechanistic evidence.",
    ))

    if gene:
        provider_requests.append((
            "UniProt Swiss-Prot",
            search_proteins(gene, max_results=1, reviewed_only=True),
            "Reviewed protein records confirm the exact gene/protein identity.",
        ))

    if gene and relation_type in {"oncogenicity", "pathogenicity", "disease_association", "biological_association"}:
        provider_requests.append((
            "Open Targets",
            get_gene_disease_associations(gene, max_results=5),
            "Genetic and disease-association evidence is the most direct structured readout for disease claims.",
        ))

    if gene and relation_type == "expression":
        provider_requests.append((
            "NCBI GEO",
            search_gene_expression(gene, max_datasets=5),
            "Expression claims should be grounded in actual transcriptomics datasets.",
        ))

    if gene and variant:
        provider_requests.append((
            "ClinVar",
            lookup_clinvar_variant(gene_symbol=gene, variant=variant, max_results=3),
            "ClinVar provides curated clinical significance for named variants.",
        ))

    if gene and variant and relation_type in {"oncogenicity", "pathogenicity", "disease_association"}:
        provider_requests.append((
            "Cancer Hotspots",
            get_cancer_hotspots(gene, cancer_type=disease_focus, min_samples=3),
            "Cancer hotspot recurrence helps validate oncogenic or disease-linked variants.",
        ))

    if queryable_population_variant:
        provider_requests.append((
            "gnomAD",
            get_population_frequency(variant),
            "Population frequency constrains whether a claimed pathogenic or oncogenic variant is plausibly rare.",
        ))

    if gene and variant and relation_type in {"pathogenicity", "disease_association"}:
        provider_requests.append((
            "ACMG Variant Classification",
            classify_variant(gene, variant),
            "ACMG-style interpretation adds a relation-aware pathogenicity summary.",
        ))

    selected_requests = provider_requests[:max_evidence_sources]
    raw_results = await asyncio.gather(
        *(awaitable for _, awaitable, _ in selected_requests),
        return_exceptions=True,
    )
    results_by_source = dict(
        zip((name for name, _, _ in selected_requests), raw_results, strict=False)
    )

    supporting: list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    evidence_by_source: dict[str, dict[str, Any]] = {
        name: {
            "status": "pending",
            "reason": reason,
            "supporting": 0,
            "contradicting": 0,
            "unresolved": 0,
        }
        for name, _, reason in selected_requests
    }

    def _record(item: dict[str, Any]) -> None:
        stance = str(item.get("stance", "unresolved"))
        source = str(item.get("source", "unknown"))
        summary = evidence_by_source.setdefault(
            source,
            {"status": "ok", "supporting": 0, "contradicting": 0, "unresolved": 0},
        )
        summary["status"] = "ok"
        if stance == "supporting":
            supporting.append(item)
            summary["supporting"] += 1
            return
        if stance == "contradicting":
            contradicting.append(item)
            summary["contradicting"] += 1
            return
        unresolved.append(item)
        summary["unresolved"] += 1

    for source_name, result in results_by_source.items():
        if isinstance(result, Exception):
            evidence_by_source[source_name]["status"] = "failed"
            evidence_by_source[source_name]["error"] = str(result)
        else:
            evidence_by_source[source_name]["status"] = "ok"

    pubmed_result = results_by_source.get("PubMed", {})
    if isinstance(pubmed_result, dict):
        for article in pubmed_result.get("articles", []):
            try:
                classification, score, rationale = _classify_pubmed_claim_evidence(
                    decomposition=decomposition,
                    article=article,
                )
            except Exception as exc:
                logger.debug(f"[verify] Article scoring failed: {exc}")
                continue
            _record(_build_evidence_item(
                source="PubMed",
                evidence=article.get("title", ""),
                stance=classification,
                score=score,
                rationale=rationale,
                relation_type=relation_type,
                pmid=article.get("pmid", ""),
                url=article.get("url", ""),
            ))

    uniprot_result = results_by_source.get("UniProt Swiss-Prot", {})
    if isinstance(uniprot_result, dict):
        proteins = uniprot_result.get("proteins", [])
        if proteins and gene:
            genes = [str(item).upper() for item in (proteins[0].get("genes") or [])]
            if gene.upper() in genes:
                _record(_build_evidence_item(
                    source="UniProt Swiss-Prot",
                    evidence=f"Reviewed UniProt entry confirms {gene}.",
                    stance="supporting",
                    score=0.24,
                    rationale="Identity confirmation anchors the claim to the correct curated protein record.",
                    relation_type=relation_type,
                ))

    ot_result = results_by_source.get("Open Targets", {})
    if isinstance(ot_result, dict):
        matched_any = False
        for assoc in ot_result.get("associations", [])[:3]:
            disease_name = str(assoc.get("disease_name", ""))
            disease_text = " ".join(
                [
                    disease_name,
                    str(assoc.get("description", "")),
                    " ".join(str(area) for area in assoc.get("therapeutic_areas", [])),
                ]
            ).lower()
            if not _text_matches_disease_terms(disease_text, disease_terms):
                continue
            matched_any = True
            overall = float(assoc.get("overall_score", 0) or 0)
            datatype_scores = assoc.get("evidence_by_datatype", {})
            genetics = float(datatype_scores.get("genetic_association", 0) or 0)
            somatic = float(datatype_scores.get("somatic_mutation", 0) or 0)
            best_score = max(overall, genetics, somatic)
            rationale = (
                f"Matched disease context '{disease_name}' with overall score {overall:.2f}; "
                f"genetic_association {genetics:.2f}; somatic_mutation {somatic:.2f}."
            )
            if best_score >= 0.55 or (relation_type == "oncogenicity" and max(genetics, somatic) >= 0.35):
                _record(_build_evidence_item(
                    source="Open Targets",
                    evidence=f"{gene}–{disease_name} structured association",
                    stance="supporting",
                    score=0.84 if best_score >= 0.75 else 0.7,
                    rationale=rationale,
                    relation_type=relation_type,
                    disease_name=disease_name,
                    overall_score=round(overall, 3),
                    genetics_score=round(genetics, 3),
                    somatic_score=round(somatic, 3),
                ))
            elif best_score <= 0.12:
                _record(_build_evidence_item(
                    source="Open Targets",
                    evidence=f"{gene}–{disease_name} structured association",
                    stance="contradicting",
                    score=0.66,
                    rationale=rationale,
                    relation_type=relation_type,
                    disease_name=disease_name,
                    overall_score=round(overall, 3),
                ))
        if ot_result.get("associations") and disease_terms and not matched_any:
            _record(_build_evidence_item(
                source="Open Targets",
                evidence=f"No Open Targets association matched disease context '{disease_focus}'.",
                stance="unresolved",
                score=0.35,
                rationale="Open Targets returned disease links, but none aligned with the disease terms extracted from the claim.",
                relation_type=relation_type,
            ))

    clinvar_result = results_by_source.get("ClinVar", {})
    if isinstance(clinvar_result, dict):
        for entry in clinvar_result.get("variants", [])[:3]:
            clin_sig = str(entry.get("clinical_significance", "")).lower()
            phenotype_text = " ".join(
                [str(entry.get("title", ""))] + [str(item) for item in entry.get("phenotypes", [])]
            ).lower()
            if disease_terms and not _text_matches_disease_terms(phenotype_text, disease_terms):
                continue
            evidence = str(entry.get("title", "")) or f"{gene} {variant}"
            if "pathogenic" in clin_sig and "benign" not in clin_sig:
                _record(_build_evidence_item(
                    source="ClinVar",
                    evidence=evidence,
                    stance="supporting",
                    score=0.88,
                    rationale=f"ClinVar classifies the variant as {entry.get('clinical_significance', '')}.",
                    relation_type=relation_type,
                    clinvar_id=entry.get("variation_id", ""),
                ))
            elif "benign" in clin_sig:
                _record(_build_evidence_item(
                    source="ClinVar",
                    evidence=evidence,
                    stance="contradicting",
                    score=0.84,
                    rationale=f"ClinVar classifies the variant as {entry.get('clinical_significance', '')}.",
                    relation_type=relation_type,
                    clinvar_id=entry.get("variation_id", ""),
                ))
            elif clin_sig:
                _record(_build_evidence_item(
                    source="ClinVar",
                    evidence=evidence,
                    stance="unresolved",
                    score=0.42,
                    rationale=f"ClinVar returned a non-binary interpretation: {entry.get('clinical_significance', '')}.",
                    relation_type=relation_type,
                    clinvar_id=entry.get("variation_id", ""),
                ))

    hotspots_result = results_by_source.get("Cancer Hotspots", {})
    if isinstance(hotspots_result, dict) and variant:
        matched_hotspots = [
            hotspot
            for hotspot in hotspots_result.get("hotspots", [])
            if _normalize_protein_variant(str(hotspot.get("amino_acid_change", ""))) == variant
        ]
        if matched_hotspots:
            top_hotspot = matched_hotspots[0]
            _record(_build_evidence_item(
                source="Cancer Hotspots",
                evidence=f"{variant} recurrent hotspot in {gene}",
                stance="supporting",
                score=0.82,
                rationale=(
                    f"The variant recurs as a hotspot in {top_hotspot.get('count', 0)} samples, "
                    "which supports a non-random cancer-driving role."
                ),
                relation_type=relation_type,
                hotspot_count=top_hotspot.get("count", 0),
            ))
        elif hotspots_result.get("hotspots"):
            _record(_build_evidence_item(
                source="Cancer Hotspots",
                evidence=f"No exact hotspot match for {variant} in {gene}",
                stance="unresolved",
                score=0.36,
                rationale="Hotspot data exists for the gene, but the exact claimed variant was not observed in the returned hotspot set.",
                relation_type=relation_type,
            ))

    gnomad_result = results_by_source.get("gnomAD", {})
    if isinstance(gnomad_result, dict):
        global_af = gnomad_result.get("global_af")
        if isinstance(global_af, (int, float)):
            af = float(global_af)
            if af <= 0.0001:
                _record(_build_evidence_item(
                    source="gnomAD",
                    evidence=f"Global AF {af:.6f}",
                    stance="supporting",
                    score=0.72,
                    rationale="Very low population frequency is consistent with a rare pathogenic or oncogenic claim.",
                    relation_type=relation_type,
                ))
            elif af >= 0.01:
                _record(_build_evidence_item(
                    source="gnomAD",
                    evidence=f"Global AF {af:.6f}",
                    stance="contradicting",
                    score=0.8,
                    rationale="Common population frequency weakens a strong pathogenic or oncogenic interpretation.",
                    relation_type=relation_type,
                ))
            else:
                _record(_build_evidence_item(
                    source="gnomAD",
                    evidence=f"Global AF {af:.6f}",
                    stance="unresolved",
                    score=0.4,
                    rationale="Population frequency is neither absent nor common enough to decide the claim alone.",
                    relation_type=relation_type,
                ))

    variant_classification = results_by_source.get("ACMG Variant Classification", {})
    if isinstance(variant_classification, dict):
        classification = str(
            variant_classification.get("classification")
            or variant_classification.get("acmg_class", "")
        ).lower()
        if "pathogenic" in classification and "benign" not in classification:
            _record(_build_evidence_item(
                source="ACMG Variant Classification",
                evidence=f"{gene} {variant}",
                stance="supporting",
                score=0.8,
                rationale=f"ACMG-style interpretation classified the variant as {variant_classification.get('classification', classification)}.",
                relation_type=relation_type,
            ))
        elif "benign" in classification:
            _record(_build_evidence_item(
                source="ACMG Variant Classification",
                evidence=f"{gene} {variant}",
                stance="contradicting",
                score=0.8,
                rationale=f"ACMG-style interpretation classified the variant as {variant_classification.get('classification', classification)}.",
                relation_type=relation_type,
            ))

    geo_result = results_by_source.get("NCBI GEO", {})
    if isinstance(geo_result, dict) and int(geo_result.get("total_found", 0) or 0) > 0:
        _record(_build_evidence_item(
            source="NCBI GEO",
            evidence=f"{geo_result.get('total_found', 0)} gene expression datasets mention {gene}.",
            stance="supporting" if relation_type == "expression" else "unresolved",
            score=0.72 if relation_type == "expression" else 0.38,
            rationale=(
                "Expression datasets directly support the expression-oriented claim."
                if relation_type == "expression"
                else "Expression datasets are relevant context but do not alone prove the target relation."
            ),
            relation_type=relation_type,
        ))

    support_weight = sum(float(item.get("score", 0) or 0) for item in supporting)
    contra_weight = sum(float(item.get("score", 0) or 0) for item in contradicting)
    n_sources_ok = sum(1 for result in raw_results if not isinstance(result, Exception))
    support_sources = {item["source"] for item in supporting}
    consensus = (
        support_weight / (support_weight + contra_weight)
        if (support_weight + contra_weight) > 0
        else 0.5
    )
    coverage_bonus = min(0.12, 0.03 * n_sources_ok)
    final_score = max(
        0.0,
        min(1.0, 0.45 + 0.23 * (support_weight - contra_weight) + 0.22 * consensus + coverage_bonus),
    )

    if final_score >= 0.88 and len(support_sources) >= 3 and contra_weight == 0:
        grade, verdict = "A", "VERIFIED"
    elif final_score >= 0.8 and len(support_sources) >= 3:
        grade, verdict = "B", "VERIFIED"
    elif final_score >= 0.68 and len(support_sources) >= 2:
        grade, verdict = "B", "LIKELY TRUE"
    elif final_score >= 0.55:
        grade, verdict = "C", "PARTIALLY SUPPORTED"
    elif final_score >= 0.4:
        grade, verdict = "D", "WEAK EVIDENCE"
    else:
        grade, verdict = "F", "CONTRADICTED"

    recommendation = {
        "A": "High confidence structured evidence supports the claim across independent sources.",
        "B": "Evidence is directionally strong, but still verify with the primary literature before citation.",
        "C": "The claim has partial support; inspect unresolved evidence before relying on it.",
        "D": "Evidence is weak or incomplete; narrow the claim or gather orthogonal validation.",
        "F": "Structured databases or literature contradict the claim; revise it before use.",
    }[grade]

    return {
        "claim": claim,
        "gene_context": gene,
        "claim_decomposition": decomposition,
        "verdict": verdict,
        "confidence_score": round(final_score, 3),
        "confidence_grade": grade,
        "supporting_evidence": sorted(supporting, key=lambda item: item["score"], reverse=True)[:10],
        "contradicting_evidence": sorted(contradicting, key=lambda item: item["score"], reverse=True)[:10],
        "unresolved": sorted(unresolved, key=lambda item: item["score"], reverse=True)[:10],
        "databases_queried": [name for name, _, _ in selected_requests],
        "evidence_strategy": [
            {"source": name, "reason": reason}
            for name, _, reason in selected_requests
        ],
        "evidence_by_source": evidence_by_source,
        "evidence_counts": {
            "supporting": len(supporting),
            "contradicting": len(contradicting),
            "unresolved": len(unresolved),
            "supporting_weight": round(support_weight, 3),
            "contradicting_weight": round(contra_weight, 3),
            "total_articles": len(pubmed_result.get("articles", [])) if isinstance(pubmed_result, dict) else 0,
        },
        "recommendation": recommendation,
        "methodology": (
            "Claim verification uses structured decomposition of the input claim into gene, variant, "
            "disease focus, and relation type. Evidence is then gathered from relation-specific "
            "sources such as Open Targets, ClinVar, gnomAD, cancer hotspot recurrence, GEO, and "
            "primary literature. PubMed articles are scored only when they mention the relevant "
            "entities and relation context, so resistance or review papers do not automatically "
            "become contradictions. This is still an automated research aid, not a substitute for "
            "expert curation."
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

    gathered_results: list[Any] = list(await asyncio.gather(
        get_gene_info(gene_symbol),
        search_proteins(gene_symbol, max_results=1),
        get_drug_targets(gene_symbol, max_results=20),
        get_gene_disease_associations(gene_symbol, max_results=10),
        return_exceptions=True,
    ))
    ncbi_result, uniprot_result, chembl_result, ot_result = gathered_results

    conflicts: list[dict[str, Any]] = []

    # Conflict 1: ChEMBL activity value consistency
    if isinstance(chembl_result, dict) and "drugs" in chembl_result:
        activity_values: dict[str, list[dict[str, Any]]] = {}
        for drug in chembl_result["drugs"]:
            try:
                mol_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
                act_type = drug.get("activity_type", "IC50")
                val = float(drug.get("activity_value", 0) or 0)
                if val > 0:
                    key = f"{mol_name}:{act_type}"
                    activity_values.setdefault(key, []).append({
                        "molecule_name": mol_name,
                        "activity_type": act_type,
                        "activity_value": val,
                        "activity_units": drug.get("activity_units", ""),
                        "activity_relation": drug.get("activity_relation", ""),
                        "assay_type": drug.get("assay_type", ""),
                        "document_year": drug.get("document_year", ""),
                    })
            except (ValueError, TypeError):
                pass

        for key, observations in activity_values.items():
            if len(observations) >= 2:
                values = [item["activity_value"] for item in observations]
                ratio = max(values) / max(min(values), 1e-9)
                if ratio > 100:
                    synthesis = synthesize_conflicting_evidence(observations)
                    conflicts.append({
                        "type":           "ACTIVITY_VALUE_DISCREPANCY",
                        "severity":       "HIGH",
                        "source_a":       "ChEMBL (assay 1)",
                        "source_b":       "ChEMBL (assay 2)",
                        "entity":         key.split(":")[0],
                        "detail":         f"IC50 values differ by {ratio:.0f}x across assays",
                        "values":         values,
                        "assay_observations": observations[:5],
                        "synthesis":      synthesis,
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
            synthesis = synthesize_conflicting_evidence([
                {"record_type": "name_alignment", "source": "NCBI Gene", "value": ncbi_name},
                {"record_type": "name_alignment", "source": "UniProt", "value": uniprot_name},
            ])
            conflicts.append({
                "type":           "GENE_NAME_MISMATCH",
                "severity":       "LOW",
                "source_a":       "NCBI Gene",
                "source_b":       "UniProt",
                "detail":         f"NCBI: '{ncbi_name}' vs UniProt: '{uniprot_name}'",
                "synthesis":      synthesis,
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
                    synthesis = synthesize_conflicting_evidence([
                        {"record_type": "evidence_asymmetry", "channel": "genetic_association", "score": genetic},
                        {"record_type": "evidence_asymmetry", "channel": "known_drug", "score": drug},
                    ])
                    conflicts.append({
                        "type":           "EVIDENCE_TYPE_ASYMMETRY",
                        "severity":       "MEDIUM",
                        "source_a":       "Open Targets (genetics)",
                        "source_b":       "Open Targets (drugs)",
                        "entity":         assoc.get("disease_name", ""),
                        "detail":         f"Strong genetic (score:{genetic:.2f}) but no approved drugs (score:{drug:.2f})",
                        "synthesis":      synthesis,
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
        "conflict_synthesis": [
            {
                "type": conflict["type"],
                "entity": conflict.get("entity", gene_symbol),
                "summary": conflict.get("synthesis", {}).get("summary", ""),
                "likely_causes": conflict.get("synthesis", {}).get("likely_causes", []),
            }
            for conflict in conflicts
        ],
    }
