"""
Strategy-driven public tool surface for Heuris-BioMCP.

This module consolidates low-level capabilities into review-friendly workflows
and adds the higher-value tools from the product strategy.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any
from urllib.parse import quote

from loguru import logger

from biomcp.utils import BioValidator, cached, get_http_client, rate_limited, with_retry

OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
ENSEMBL_BASE = "https://rest.ensembl.org"
ENSEMBL_GRCH37_BASE = "https://grch37.rest.ensembl.org"
STRING_API_BASE = "https://string-db.org/api/json"
OLS_SEARCH_BASE = "https://www.ebi.ac.uk/ols4/api/search"

_GENE_TOKEN_STOPWORDS = {
    "DNA",
    "RNA",
    "CELL",
    "CELLS",
    "GENE",
    "GENES",
    "PANEL",
    "PANELS",
    "NGS",
    "PCR",
    "RNASEQ",
    "MRNA",
    "PATHWAY",
    "REVIEW",
    "CASE",
    "REPORT",
    "TRIAL",
    "TUMOR",
    "CANCER",
    "DISEASE",
    "PATIENT",
    "PATIENTS",
    "HUMAN",
    "HUMANS",
    "COVID",
    "SARS",
    "SIGNATURE",
    "BIOMARKER",
    "BIOMARKERS",
}

_CELL_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "T_cell": ("CD3D", "CD3E", "TRBC1", "IL7R", "LTB"),
    "B_cell": ("MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA"),
    "NK_cell": ("NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A"),
    "Myeloid": ("LYZ", "S100A8", "S100A9", "FCN1", "CTSS"),
    "Dendritic": ("FCER1A", "CST3", "CLEC10A", "HLA-DPA1", "HLA-DPB1"),
    "Endothelial": ("PECAM1", "VWF", "KDR", "EMCN", "ESAM"),
    "Fibroblast": ("COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"),
    "Epithelial": ("EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"),
    "Hepatocyte": ("ALB", "APOA1", "TF", "CYP3A4", "HP"),
    "Neuron": ("RBFOX3", "MAP2", "SNAP25", "SYT1", "GAP43"),
    "Astrocyte": ("GFAP", "AQP4", "SLC1A3", "ALDH1L1", "S100B"),
}

_CPIC_GUIDELINES = {
    "warfarin": {
        "genes": ["CYP2C9", "VKORC1", "CYP4F2"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-warfarin-and-cyp2c9-vkorc1/",
    },
    "clopidogrel": {
        "genes": ["CYP2C19"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-clopidogrel-and-cyp2c19/",
    },
    "simvastatin": {
        "genes": ["SLCO1B1"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-simvastatin-and-slco1b1/",
    },
    "thiopurine": {
        "genes": ["TPMT", "NUDT15"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
    },
    "azathioprine": {
        "genes": ["TPMT", "NUDT15"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
    },
    "mercaptopurine": {
        "genes": ["TPMT", "NUDT15"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
    },
    "allopurinol": {
        "genes": ["HLA-B"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-allopurinol-and-hla-b/",
    },
    "abacavir": {
        "genes": ["HLA-B"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-abacavir-and-hla-b/",
    },
    "codeine": {
        "genes": ["CYP2D6"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-codeine-and-cyp2d6/",
    },
    "tramadol": {
        "genes": ["CYP2D6"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-opioids-and-cyp2d6/",
    },
    "tacrolimus": {
        "genes": ["CYP3A5"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-tacrolimus-and-cyp3a5/",
    },
    "carbamazepine": {
        "genes": ["HLA-B", "HLA-A"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-carbamazepine-and-hla-b/",
    },
    "irinotecan": {
        "genes": ["UGT1A1"],
        "guideline_url": "https://cpicpgx.org/guidelines/guideline-for-irinotecan-and-ugt1a1/",
    },
}

_PGX_GENE_PANEL = {
    "ABCG2",
    "CACNA1S",
    "CFTR",
    "CYP2B6",
    "CYP2C9",
    "CYP2C19",
    "CYP2D6",
    "CYP3A4",
    "CYP3A5",
    "DPYD",
    "G6PD",
    "HLA-A",
    "HLA-B",
    "IFNL3",
    "NUDT15",
    "RYR1",
    "SLCO1B1",
    "TPMT",
    "UGT1A1",
    "VKORC1",
}


def _extract_gene_tokens(text: str) -> list[str]:
    counts: Counter[str] = Counter()
    for token in re.findall(r"\b[A-Z0-9-]{3,10}\b", text.upper()):
        if token in _GENE_TOKEN_STOPWORDS or token.isdigit():
            continue
        counts[token] += 1
    return [gene for gene, _ in counts.most_common()]


def _safe_preview(text: str, limit: int = 350) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _token_set(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-z0-9]+", value.lower()):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _extract_named_genes(text: str, allowed_genes: set[str] | None = None) -> list[str]:
    if not text:
        return []
    haystack = text.upper()
    allowed = sorted(allowed_genes or _PGX_GENE_PANEL)
    matches: list[str] = []
    for gene in allowed:
        if re.search(rf"(?<![A-Z0-9]){re.escape(gene)}(?![A-Z0-9])", haystack):
            matches.append(gene)
    return matches


def _merge_gene_candidates(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            gene = str(candidate.get("gene", "")).strip().upper()
            if not gene:
                continue

            evidence_breakdown = list(candidate.get("evidence_breakdown", []) or [])
            source = str(candidate.get("source", "")).strip()

            if gene not in merged:
                merged[gene] = {
                    **candidate,
                    "gene": gene,
                    "score": round(float(candidate.get("score", 0.0)), 3),
                    "evidence_breakdown": evidence_breakdown[:10],
                    "sources": [source] if source else [],
                }
                order.append(gene)
                continue

            existing = merged[gene]
            existing["score"] = round(
                max(float(existing.get("score", 0.0)), float(candidate.get("score", 0.0))),
                3,
            )
            if candidate.get("gene_name") and not existing.get("gene_name"):
                existing["gene_name"] = candidate["gene_name"]
            if source and source not in existing["sources"]:
                existing["sources"].append(source)
            existing["evidence_breakdown"] = (
                list(existing.get("evidence_breakdown", [])) + evidence_breakdown
            )[:10]

    combined: list[dict[str, Any]] = []
    for gene in order:
        candidate = merged[gene]
        sources = candidate.pop("sources", [])
        candidate["source"] = " + ".join(sources) if sources else candidate.get("source", "")
        combined.append(candidate)
    return combined


async def _fetch_ensembl_region_features(region_string: str, api_base: str) -> dict[str, Any]:
    client = await get_http_client()

    async def _fetch(feature: str) -> list[dict[str, Any]]:
        resp = await client.get(
            f"{api_base}/overlap/region/human/{region_string}",
            params={"feature": feature},
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    genes, variations, regulatory = await asyncio.gather(
        _fetch("gene"),
        _fetch("variation"),
        _fetch("regulatory"),
        return_exceptions=True,
    )

    gene_rows = genes if isinstance(genes, list) else []
    variation_rows = variations if isinstance(variations, list) else []
    regulatory_rows = regulatory if isinstance(regulatory, list) else []

    return {
        "nearby_genes": [
            {
                "gene_id": row.get("id", ""),
                "symbol": row.get("external_name", ""),
                "biotype": row.get("biotype", ""),
                "start": row.get("start"),
                "end": row.get("end"),
                "strand": row.get("strand"),
            }
            for row in gene_rows[:8]
        ],
        "notable_variants": [
            {
                "variant_id": row.get("id", ""),
                "start": row.get("start"),
                "end": row.get("end"),
                "strand": row.get("strand"),
                "consequence_type": row.get("consequence_type", []),
            }
            for row in variation_rows[:8]
        ],
        "regulatory_features": [
            {
                "feature_id": row.get("id", ""),
                "feature_type": row.get("feature_type", ""),
                "description": row.get("description", ""),
                "start": row.get("start"),
                "end": row.get("end"),
            }
            for row in regulatory_rows[:8]
        ],
        "counts": {
            "genes": len(gene_rows),
            "variants": len(variation_rows),
            "regulatory_features": len(regulatory_rows),
        },
    }


async def _resolve_primary_accession(query: str) -> str:
    from biomcp.tools.proteins import search_proteins

    result = await search_proteins(query, max_results=1, reviewed_only=True)
    proteins = result.get("proteins", [])
    if not proteins:
        raise LookupError(f"No reviewed UniProt protein found for '{query}'.")
    accession = proteins[0].get("accession", "")
    if not accession:
        raise LookupError(f"Protein result for '{query}' is missing a UniProt accession.")
    return accession


async def _search_open_targets_disease(disease: str, panel_size: int) -> list[dict[str, Any]]:
    client = await get_http_client()
    search_resp = await client.post(
        OPENTARGETS_GQL,
        json={
            "query": """
                query SearchDisease($query: String!) {
                  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: 1}) {
                    hits {
                      id
                      name
                    }
                  }
                }
            """,
            "variables": {"query": disease},
        },
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    search_resp.raise_for_status()
    hits = ((search_resp.json().get("data") or {}).get("search") or {}).get("hits") or []
    if not hits:
        return []

    efo_id = hits[0].get("id", "")
    if not efo_id:
        return []

    assoc_resp = await client.post(
        OPENTARGETS_GQL,
        json={
            "query": """
                query DiseaseAssociations($efoId: String!, $size: Int!) {
                  disease(efoId: $efoId) {
                    associatedTargets(page: {index: 0, size: $size}) {
                      rows {
                        score
                        datatypeScores {
                          id
                          score
                        }
                        target {
                          approvedSymbol
                          approvedName
                        }
                      }
                    }
                  }
                }
            """,
            "variables": {"efoId": efo_id, "size": panel_size},
        },
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    assoc_resp.raise_for_status()
    rows = (
        ((assoc_resp.json().get("data") or {}).get("disease") or {})
        .get("associatedTargets", {})
        .get("rows", [])
    )
    candidates: list[dict[str, Any]] = []
    for row in rows:
        target = row.get("target") or {}
        symbol = target.get("approvedSymbol", "")
        if not symbol:
            continue
        candidates.append(
            {
                "gene": symbol,
                "gene_name": target.get("approvedName", ""),
                "score": round(float(row.get("score", 0.0)), 3),
                "evidence_breakdown": row.get("datatypeScores", [])[:5],
                "source": "Open Targets",
            }
        )
    return candidates


async def _heuristic_biomarker_candidates(disease: str, panel_size: int) -> list[dict[str, Any]]:
    from biomcp.tools.ncbi import get_gene_info, search_pubmed

    papers = await search_pubmed(
        f'{disease} ("biomarker" OR "gene signature" OR "diagnostic panel")',
        max_results=min(max(panel_size * 2, 10), 20),
    )
    counts: Counter[str] = Counter()
    for article in papers.get("articles", []):
        counts.update(
            _extract_gene_tokens(f"{article.get('title', '')} {article.get('abstract', '')}")
        )

    gene_candidates = [gene for gene, _ in counts.most_common(panel_size * 3)]
    info_results = await asyncio.gather(
        *[get_gene_info(gene) for gene in gene_candidates],
        return_exceptions=True,
    )

    selected: list[dict[str, Any]] = []
    for gene, info in zip(gene_candidates, info_results, strict=False):
        if isinstance(info, Exception) or info.get("error"):
            continue
        selected.append(
            {
                "gene": gene,
                "gene_name": info.get("description", ""),
                "score": round(min(1.0, counts[gene] / max(counts[gene_candidates[0]], 1)), 3),
                "evidence_breakdown": [{"id": "pubmed_frequency", "score": counts[gene]}],
                "source": "PubMed heuristic",
            }
        )
        if len(selected) >= panel_size:
            break
    return selected


@cached("uniprot")
@rate_limited("default")
async def find_protein(
    query: str = "",
    source: str = "auto",
    accession: str = "",
    organism: str = "homo sapiens",
    reviewed_only: bool = True,
    max_results: int = 10,
) -> dict[str, Any]:
    from biomcp.tools.proteins import get_protein_info, search_pdb_structures, search_proteins

    source = source.lower()
    max_results = BioValidator.clamp_int(max_results, 1, 25, "max_results")
    if source not in {"auto", "both", "uniprot", "pdb"}:
        raise ValueError("source must be one of: auto, both, uniprot, pdb.")

    if accession:
        protein = await get_protein_info(accession)
        return {
            "mode": "record_lookup",
            "source": "uniprot",
            "accession": accession,
            "result": protein,
        }

    if not query:
        raise ValueError("Provide either accession or query.")

    do_uniprot = source in {"auto", "both", "uniprot"}
    do_pdb = source in {"auto", "both", "pdb"}
    tasks: list[Any] = []
    if do_uniprot:
        tasks.append(search_proteins(query, organism=organism, max_results=max_results, reviewed_only=reviewed_only))
    if do_pdb:
        tasks.append(search_pdb_structures(query, max_results=max_results))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    payload: dict[str, Any] = {
        "query": query,
        "source": source,
        "organism": organism,
        "reviewed_only": reviewed_only,
    }

    idx = 0
    if do_uniprot:
        uniprot_result = results[idx]
        payload["uniprot_results"] = (
            uniprot_result if isinstance(uniprot_result, dict) else {"error": str(uniprot_result)}
        )
        idx += 1
    if do_pdb:
        pdb_result = results[idx]
        payload["pdb_results"] = pdb_result if isinstance(pdb_result, dict) else {"error": str(pdb_result)}

    return payload


@cached("reactome")
@rate_limited("default")
async def pathway_analysis(
    action: str = "auto",
    db: str = "auto",
    query: str = "",
    gene_symbol: str = "",
    pathway_id: str = "",
    organism: str = "hsa",
) -> dict[str, Any]:
    from biomcp.tools.pathways import get_pathway_genes, get_reactome_pathways, search_pathways

    action = action.lower()
    db = db.lower()
    if db not in {"auto", "kegg", "reactome"}:
        raise ValueError("db must be one of: auto, kegg, reactome.")

    if action == "auto":
        if pathway_id:
            action = "genes"
        elif gene_symbol:
            action = "gene_context"
        elif query:
            action = "search"
        else:
            raise ValueError("Provide query, gene_symbol, or pathway_id.")

    if action == "genes":
        if not pathway_id:
            raise ValueError("pathway_id is required when action='genes'.")
        genes = await get_pathway_genes(pathway_id)
        return {"action": action, "db": "kegg", "pathway": genes}

    if action == "search":
        search_term = query or gene_symbol
        if not search_term:
            raise ValueError("query or gene_symbol is required when action='search'.")
        kegg_result = None
        reactome_result = None
        if db in {"auto", "kegg"}:
            kegg_result = await search_pathways(search_term, organism=organism)
        if db in {"auto", "reactome"} and gene_symbol:
            reactome_result = await get_reactome_pathways(gene_symbol)
        return {
            "action": action,
            "db": db,
            "query": search_term,
            "kegg": kegg_result,
            "reactome": reactome_result,
        }

    if action == "gene_context":
        if not gene_symbol:
            raise ValueError("gene_symbol is required when action='gene_context'.")
        results = await asyncio.gather(
            get_reactome_pathways(gene_symbol),
            search_pathways(gene_symbol, organism=organism),
            return_exceptions=True,
        )
        return {
            "action": action,
            "gene": gene_symbol,
            "reactome": results[0] if isinstance(results[0], dict) else {"error": str(results[0])},
            "kegg": results[1] if isinstance(results[1], dict) else {"error": str(results[1])},
        }

    raise ValueError("Unsupported action. Use auto, search, gene_context, or genes.")


@rate_limited("default")
async def crispr_analysis(
    action: str,
    gene_symbol: str = "",
    guide_sequence: str = "",
    target_mutation: str = "",
    repair_template: str = "",
    cas_variant: str = "SpCas9",
    target_region: str = "early_exons",
    n_guides: int = 5,
    min_score: float = 40.0,
    mismatches: int = 3,
    genome: str = "hg38",
    cell_line: str = "generic",
    use_blast: bool = False,
) -> dict[str, Any]:
    from biomcp.tools.crispr_tools import (
        design_base_editor_guides,
        design_crispr_guides,
        get_crispr_repair_outcomes,
        predict_off_target_sites,
        score_guide_efficiency,
    )

    action = action.lower()
    if action == "design":
        if not gene_symbol:
            raise ValueError("gene_symbol is required for CRISPR guide design.")
        return await design_crispr_guides(
            gene_symbol=gene_symbol,
            target_region=target_region,
            cas_variant=cas_variant,
            n_guides=n_guides,
            min_score=min_score,
        )
    if action == "score":
        if not guide_sequence:
            raise ValueError("guide_sequence is required for guide scoring.")
        return await score_guide_efficiency(
            guide_sequence=guide_sequence,
            cas_variant=cas_variant,
        )
    if action == "off_target":
        if not guide_sequence:
            raise ValueError("guide_sequence is required for off-target analysis.")
        return await predict_off_target_sites(
            guide_sequence=guide_sequence,
            cas_variant=cas_variant,
            mismatches=mismatches,
            genome=genome,
            use_blast=use_blast,
        )
    if action == "base_edit":
        if not gene_symbol or not target_mutation:
            raise ValueError("gene_symbol and target_mutation are required for base editing.")
        return await design_base_editor_guides(
            gene_symbol=gene_symbol,
            target_mutation=target_mutation,
        )
    if action == "repair":
        if not gene_symbol or not guide_sequence:
            raise ValueError("gene_symbol and guide_sequence are required for repair outcome analysis.")
        return await get_crispr_repair_outcomes(
            gene_symbol=gene_symbol,
            guide_sequence=guide_sequence,
            repair_template=repair_template,
            cell_line=cell_line,
        )

    raise ValueError("Unsupported action. Use design, score, off_target, base_edit, or repair.")


@rate_limited("default")
async def drug_safety(
    action: str,
    drug_name: str,
    comparator_drug: str = "",
    event_type: str = "all",
    serious_only: bool = False,
    event_terms: list[str] | None = None,
    max_results: int = 50,
    patient_sex: str = "",
    age_group: str = "",
) -> dict[str, Any]:
    from biomcp.tools.drug_safety import (
        analyze_safety_signals,
        compare_drug_safety,
        get_drug_label_warnings,
        query_adverse_events,
    )

    action = action.lower()
    if action == "events":
        return await query_adverse_events(
            drug_name=drug_name,
            event_type=event_type,
            serious_only=serious_only,
            max_results=max_results,
            patient_sex=patient_sex,
            age_group=age_group,
        )
    if action == "signals":
        return await analyze_safety_signals(drug_name=drug_name, event_terms=event_terms)
    if action == "label":
        return await get_drug_label_warnings(drug_name=drug_name)
    if action == "compare":
        if not comparator_drug:
            raise ValueError("comparator_drug is required when action='compare'.")
        return await compare_drug_safety(drugs=[drug_name, comparator_drug], event_category=event_type)

    raise ValueError("Unsupported action. Use events, signals, label, or compare.")


@rate_limited("default")
async def variant_analysis(
    action: str,
    gene_symbol: str = "",
    variant: str = "",
    inheritance: str = "unknown",
    consequence: str = "",
    proband_phenotype: str = "",
    populations: list[str] | None = None,
) -> dict[str, Any]:
    from biomcp.tools.innovations import predict_splice_impact
    from biomcp.tools.variant_interpreter import (
        classify_variant,
        get_population_frequency,
        lookup_clinvar_variant,
    )

    action = action.lower()
    if action == "classify":
        if not gene_symbol or not variant:
            raise ValueError("gene_symbol and variant are required for classification.")
        return await classify_variant(
            gene_symbol=gene_symbol,
            variant=variant,
            inheritance=inheritance,
            consequence=consequence,
            proband_phenotype=proband_phenotype,
        )
    if action == "population_frequency":
        if not variant:
            raise ValueError("variant is required for population frequency lookup.")
        return await get_population_frequency(variant_id=variant, populations=populations)
    if action == "clinvar":
        if not variant and not gene_symbol:
            raise ValueError("gene_symbol or variant is required for ClinVar lookup.")
        return await lookup_clinvar_variant(gene_symbol=gene_symbol, variant=variant)
    if action == "splice":
        if not gene_symbol or not variant:
            raise ValueError("gene_symbol and variant are required for splice analysis.")
        return await predict_splice_impact(gene_symbol=gene_symbol, variant=variant)
    if action == "full_report":
        if not gene_symbol or not variant:
            raise ValueError("gene_symbol and variant are required for a full variant report.")
        results = await asyncio.gather(
            classify_variant(
                gene_symbol=gene_symbol,
                variant=variant,
                inheritance=inheritance,
                consequence=consequence,
                proband_phenotype=proband_phenotype,
            ),
            get_population_frequency(variant_id=variant, populations=populations),
            lookup_clinvar_variant(gene_symbol=gene_symbol, variant=variant),
            return_exceptions=True,
        )
        report: dict[str, Any] = {
            "gene": gene_symbol,
            "variant": variant,
            "classification": results[0] if isinstance(results[0], dict) else {"error": str(results[0])},
            "population_frequency": results[1] if isinstance(results[1], dict) else {"error": str(results[1])},
            "clinvar": results[2] if isinstance(results[2], dict) else {"error": str(results[2])},
        }
        if re.search(r"(?:\+|-)\d|splice", variant, flags=re.IGNORECASE):
            try:
                report["splice_impact"] = await predict_splice_impact(gene_symbol=gene_symbol, variant=variant)
            except Exception as exc:
                report["splice_impact"] = {"error": str(exc)}
        return report

    raise ValueError("Unsupported action. Use classify, population_frequency, clinvar, splice, or full_report.")


@rate_limited("default")
async def boltz2_workflow(
    mode: str = "structure",
    protein_sequences: list[str] | None = None,
    ligand_smiles: list[str] | None = None,
    dna_sequences: list[str] | None = None,
    rna_sequences: list[str] | None = None,
    uniprot_accession: str = "",
    predict_affinity: bool = False,
    method_conditioning: str | None = None,
    pocket_residues: list[dict[str, Any]] | None = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
) -> dict[str, Any]:
    from biomcp.tools.nvidia_nim import design_protein_ligand, predict_structure_boltz2

    mode = mode.lower()
    if mode not in {"structure", "protein_ligand"}:
        raise ValueError("Unsupported mode. Use structure or protein_ligand.")
    if mode == "protein_ligand":
        if not uniprot_accession:
            raise ValueError("uniprot_accession is required when mode='protein_ligand'.")
        if not ligand_smiles or not ligand_smiles[0]:
            raise ValueError("At least one ligand_smiles value is required when mode='protein_ligand'.")
        return await design_protein_ligand(
            uniprot_accession=uniprot_accession,
            ligand_smiles=ligand_smiles[0],
            predict_affinity=predict_affinity,
            method_conditioning=method_conditioning,
        )

    return await predict_structure_boltz2(
        protein_sequences=protein_sequences or [],
        ligand_smiles=ligand_smiles,
        dna_sequences=dna_sequences,
        rna_sequences=rna_sequences,
        predict_affinity=predict_affinity,
        method_conditioning=method_conditioning,
        pocket_residues=pocket_residues,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
    )


@rate_limited("default")
async def evo2_workflow(
    mode: str = "generate",
    sequence: str = "",
    num_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 4,
    top_p: float = 1.0,
    enable_logits: bool = False,
    num_generations: int = 1,
    wildtype_sequence: str = "",
    variant_sequence: str = "",
) -> dict[str, Any]:
    from biomcp.tools.nvidia_nim import generate_dna_evo2, score_sequence_evo2

    mode = mode.lower()
    if mode not in {"generate", "score"}:
        raise ValueError("Unsupported mode. Use generate or score.")
    if mode == "score":
        if not wildtype_sequence or not variant_sequence:
            raise ValueError("wildtype_sequence and variant_sequence are required when mode='score'.")
        return await score_sequence_evo2(
            wildtype_sequence=wildtype_sequence,
            variant_sequence=variant_sequence,
        )

    return await generate_dna_evo2(
        sequence=sequence,
        num_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_logits=enable_logits,
        num_generations=num_generations,
    )


@cached("fda")
@rate_limited("default")
async def drug_interaction_checker(
    drug_name: str,
    co_medications: list[str] | None = None,
) -> dict[str, Any]:
    from biomcp.tools.drug_safety import get_drug_label_warnings

    primary = await get_drug_label_warnings(drug_name)
    if primary.get("error"):
        return primary

    interaction_text = " ".join(
        [
            primary.get("drug_interactions", ""),
            primary.get("boxed_warning", ""),
            primary.get("warnings_and_cautions", ""),
            primary.get("contraindications", ""),
        ]
    ).lower()

    co_medications = co_medications or []
    interactions: list[dict[str, Any]] = []
    for co_med in co_medications:
        secondary = await get_drug_label_warnings(co_med)
        secondary_text = " ".join(
            [
                secondary.get("drug_interactions", ""),
                secondary.get("boxed_warning", ""),
                secondary.get("warnings_and_cautions", ""),
                secondary.get("contraindications", ""),
            ]
        ).lower()
        detected_in_primary = co_med.lower() in interaction_text
        detected_in_secondary = drug_name.lower() in secondary_text
        name_hits = detected_in_primary or detected_in_secondary
        severity = (
            "high"
            if name_hits and (
                co_med.lower() in primary.get("boxed_warning", "").lower()
                or co_med.lower() in primary.get("contraindications", "").lower()
                or drug_name.lower() in secondary.get("boxed_warning", "").lower()
                or drug_name.lower() in secondary.get("contraindications", "").lower()
            )
            else "moderate"
            if name_hits
            else "informational"
        )
        interactions.append(
            {
                "co_medication": co_med,
                "detected_in_primary_label": detected_in_primary,
                "detected_in_secondary_label": detected_in_secondary,
                "severity": severity,
                "primary_interaction_snippet": _safe_preview(primary.get("drug_interactions", "")),
                "secondary_warning_snippet": _safe_preview(
                    secondary.get("boxed_warning", "") or secondary.get("warnings_and_cautions", "")
                ),
                "supporting_urls": [
                    primary.get("full_label_url", ""),
                    secondary.get("full_label_url", ""),
                ],
            }
        )

    return {
        "drug": drug_name,
        "label_based_interaction_summary": _safe_preview(primary.get("drug_interactions", "")),
        "has_black_box_warning": primary.get("has_black_box_warning", False),
        "interactions": interactions,
        "note": "FDA label-based interaction screen. Confirm with current prescribing information before clinical use.",
    }


@cached("uniprot")
@rate_limited("default")
async def protein_binding_pocket(
    accession: str = "",
    query: str = "",
    max_sites: int = 10,
) -> dict[str, Any]:
    from biomcp.tools.proteins import get_alphafold_structure, get_protein_info

    max_sites = BioValidator.clamp_int(max_sites, 1, 20, "max_sites")
    if not accession and not query:
        raise ValueError("Provide accession or query.")
    resolved_accession = accession or await _resolve_primary_accession(query)
    protein = await get_protein_info(resolved_accession)
    structure = await get_alphafold_structure(resolved_accession)

    candidate_sites: list[dict[str, Any]] = []
    site_counts: Counter[str] = Counter()
    for feature in protein.get("features", []):
        ftype = (feature.get("type") or "").lower()
        if not any(keyword in ftype for keyword in ("binding", "site", "active", "region", "motif", "domain")):
            continue
        site_class = (
            "binding_site"
            if "binding" in ftype
            else "active_site"
            if "active" in ftype
            else "motif_or_region"
        )
        site_counts[site_class] += 1
        candidate_sites.append(
            {
                "feature_type": feature.get("type", ""),
                "site_class": site_class,
                "description": feature.get("description", ""),
                "start": feature.get("start"),
                "end": feature.get("end"),
                "evidence_source": "UniProt annotation",
            }
        )

    return {
        "accession": resolved_accession,
        "protein_name": protein.get("full_name", ""),
        "candidate_sites": candidate_sites[:max_sites],
        "site_type_summary": dict(site_counts),
        "alphafold_confidence": structure.get("plddt_summary", {}),
        "druggability_note": (
            "Candidate sites are derived from curated UniProt features and AlphaFold context. "
            "Use pocket-specific tools such as fpocket or DoGSiteScorer for docking-grade pocket geometry."
        ),
    }


@cached("default")
@rate_limited("default")
async def biomarker_panel_design(
    disease: str,
    panel_size: int = 10,
    context: str = "oncology",
) -> dict[str, Any]:
    panel_size = BioValidator.clamp_int(panel_size, 3, 25, "panel_size")

    open_targets_candidates: list[dict[str, Any]] = []
    try:
        open_targets_candidates = await _search_open_targets_disease(disease, panel_size)
    except Exception as exc:
        logger.warning(f"[BiomarkerPanel] Open Targets search failed for '{disease}': {exc}")
    heuristic_candidates: list[dict[str, Any]] = []
    if len(open_targets_candidates) < panel_size:
        heuristic_candidates = await _heuristic_biomarker_candidates(disease, panel_size)

    candidates = _merge_gene_candidates(open_targets_candidates, heuristic_candidates)
    source = (
        "Open Targets + PubMed heuristic"
        if open_targets_candidates and heuristic_candidates
        else "Open Targets"
        if open_targets_candidates
        else "PubMed heuristic"
    )

    selected = candidates[:panel_size]
    return {
        "disease": disease,
        "context": context,
        "panel_size": panel_size,
        "panel": selected,
        "panel_coverage": {
            "requested": panel_size,
            "returned": len(selected),
            "open_targets_candidates": len(open_targets_candidates),
            "heuristic_candidates": len(heuristic_candidates),
        },
        "design_principles": [
            "Prioritize genes with replicated disease association evidence.",
            "Prefer markers with orthogonal support from literature, genetics, or cancer cohorts.",
            "Treat this as a research panel draft and validate analytically before clinical deployment.",
        ],
        "evidence_source": source,
    }


@cached("pharmgkb")
@rate_limited("default")
async def pharmacogenomics_report(
    drug_name: str,
    gene_symbol: str = "",
    max_annotations: int = 10,
) -> dict[str, Any]:
    from biomcp.tools.databases import get_pharmgkb_variants
    from biomcp.tools.drug_safety import get_drug_label_warnings

    max_annotations = BioValidator.clamp_int(max_annotations, 1, 25, "max_annotations")
    normalized_drug = drug_name.lower()
    matched_cpic = None
    for key, guideline in _CPIC_GUIDELINES.items():
        if key in normalized_drug:
            matched_cpic = guideline
            break

    label = await get_drug_label_warnings(drug_name)
    label_context = " ".join(
        [
            label.get("drug_interactions", ""),
            label.get("warnings_and_cautions", ""),
            label.get("use_in_specific_populations", ""),
            label.get("boxed_warning", ""),
            label.get("pharmacogenomics", ""),
        ]
    )
    label_gene_mentions = _extract_named_genes(label_context, _PGX_GENE_PANEL)
    suggested_genes = []
    if matched_cpic:
        suggested_genes.extend(matched_cpic["genes"])
    if gene_symbol:
        suggested_genes.append(BioValidator.validate_gene_symbol(gene_symbol))
    suggested_genes.extend(label_gene_mentions)
    genes_to_query = _dedupe_preserve_order(suggested_genes)
    pgx_results = await asyncio.gather(
        *[get_pharmgkb_variants(gene, max_results=max_annotations) for gene in genes_to_query],
        return_exceptions=True,
    )

    annotations: list[dict[str, Any]] = []
    for gene, result in zip(genes_to_query, pgx_results, strict=False):
        if isinstance(result, Exception):
            annotations.append({"gene": gene, "error": str(result)})
        else:
            annotations.append({"gene": gene, **result})

    testing_recommendations: list[str] = []
    if matched_cpic:
        testing_recommendations.append(
            f"CPIC guidance is available for {drug_name} via {matched_cpic['guideline_url']}"
        )
    for gene in label_gene_mentions:
        testing_recommendations.append(
            f"Drug label text references {gene}; review whether pre-treatment genotyping is appropriate."
        )

    return {
        "drug": drug_name,
        "genes_considered": genes_to_query,
        "cpic_guideline": matched_cpic,
        "label_gene_mentions": label_gene_mentions,
        "pharmacogenomic_evidence": annotations,
        "label_pharmacogenomics_context": _safe_preview(
            label_context
        ),
        "testing_recommendations": _dedupe_preserve_order(testing_recommendations),
        "note": "This report combines CPIC-style gene selection with PharmGKB-style evidence retrieval when available.",
    }


@cached("uniprot")
@rate_limited("default")
async def protein_family_analysis(
    accession: str = "",
    query: str = "",
) -> dict[str, Any]:
    from biomcp.tools.innovations import get_protein_domain_structure
    from biomcp.tools.proteins import get_protein_info

    if not accession and not query:
        raise ValueError("Provide accession or query.")
    resolved_accession = accession or await _resolve_primary_accession(query)
    protein_result, interpro_result = await asyncio.gather(
        get_protein_info(resolved_accession),
        get_protein_domain_structure(resolved_accession),
        return_exceptions=True,
    )

    protein = protein_result if isinstance(protein_result, dict) else {"error": str(protein_result)}
    interpro = interpro_result if isinstance(interpro_result, dict) else {"error": str(interpro_result)}
    domain_features = [
        feature
        for feature in protein.get("features", [])
        if (feature.get("type") or "").lower() in {"domain", "region", "repeat", "motif"}
    ]
    interpro_domains = interpro.get("domains", [])

    keywords = []
    full_name = protein.get("full_name", "")
    for token in re.findall(r"[A-Za-z0-9-]+", full_name):
        if len(token) >= 4:
            keywords.append(token)
    for domain in interpro_domains[:10]:
        name = str(domain.get("name", ""))
        for token in re.findall(r"[A-Za-z0-9-]+", name):
            if len(token) >= 4:
                keywords.append(token)

    return {
        "accession": resolved_accession,
        "protein_name": full_name,
        "gene_names": protein.get("gene_names", []),
        "domain_annotations": domain_features,
        "domain_architecture": interpro_domains[:10],
        "domain_coverage_pct": interpro.get("domain_coverage_pct"),
        "domain_diagram": interpro.get("domain_diagram", ""),
        "putative_family_keywords": _dedupe_preserve_order(keywords)[:10],
        "cross_reference_links": {
            "interpro": f"https://www.ebi.ac.uk/interpro/search/text/{quote(resolved_accession)}",
            "pfam": f"https://pfam.xfam.org/protein/{quote(resolved_accession)}",
        },
        "summary": (
            "Family context is derived from curated UniProt annotations plus InterPro domain architecture."
        ),
    }


@cached("reactome")
@rate_limited("default")
async def network_enrichment(
    gene_list: list[str],
    min_string_score: int = 700,
    max_results: int = 10,
) -> dict[str, Any]:
    from biomcp.tools.databases import get_string_interactions
    from biomcp.tools.innovations import compute_pathway_enrichment
    from biomcp.tools.pathways import get_reactome_pathways

    if not gene_list:
        raise ValueError("gene_list is required.")
    normalized_genes = [BioValidator.validate_gene_symbol(gene) for gene in gene_list[:20]]
    max_results = BioValidator.clamp_int(max_results, 3, 25, "max_results")

    reactome_results = await asyncio.gather(
        *[get_reactome_pathways(gene) for gene in normalized_genes],
        return_exceptions=True,
    )
    string_results = await asyncio.gather(
        *[
            get_string_interactions(gene, min_score=min_string_score, max_results=max_results)
            for gene in normalized_genes
        ],
        return_exceptions=True,
    )
    enrichment_result: dict[str, Any] | None = None
    if len(normalized_genes) >= 2:
        try:
            enrichment_result = await compute_pathway_enrichment(
                normalized_genes,
                database="both",
                min_genes=2,
            )
        except Exception as exc:
            enrichment_result = {"error": str(exc)}

    pathway_counts: Counter[str] = Counter()
    pathway_records: dict[str, dict[str, Any]] = {}
    for result in reactome_results:
        if not isinstance(result, dict):
            continue
        for pathway in result.get("pathways", []):
            name = pathway.get("name", "")
            if not name:
                continue
            pathway_counts[name] += 1
            pathway_records.setdefault(name, pathway)

    partner_counts: Counter[str] = Counter()
    input_gene_set = set(normalized_genes)
    edge_records: dict[tuple[str, str], dict[str, Any]] = {}
    for result in string_results:
        if not isinstance(result, dict):
            continue
        for interaction in result.get("interactions", []):
            partner = interaction.get("partner", "")
            if partner:
                partner_counts[partner] += 1
                if partner in input_gene_set:
                    pair = tuple(sorted((result.get("gene", ""), partner)))
                    if all(pair):
                        current = edge_records.get(pair)
                        score = float(interaction.get("combined_score", 0.0))
                        if current is None or score > current["combined_score"]:
                            edge_records[pair] = {
                                "gene_a": pair[0],
                                "gene_b": pair[1],
                                "combined_score": score,
                            }

    return {
        "input_genes": normalized_genes,
        "enriched_pathways": [
            {
                "pathway_name": name,
                "hit_count": count,
                "pathway": pathway_records.get(name, {}),
            }
            for name, count in pathway_counts.most_common(max_results)
        ],
        "network_hubs": [
            {"gene": gene, "supporting_inputs": count}
            for gene, count in partner_counts.most_common(max_results)
        ],
        "input_gene_edges": sorted(
            edge_records.values(),
            key=lambda item: item["combined_score"],
            reverse=True,
        )[:max_results],
        "pathway_enrichment": (
            enrichment_result.get("enriched_pathways", [])[:max_results]
            if isinstance(enrichment_result, dict)
            else []
        ),
        "pathway_enrichment_summary": (
            {
                "pathways_tested": enrichment_result.get("pathways_tested", 0),
                "significant_pathways": enrichment_result.get("significant_pathways", 0),
            }
            if isinstance(enrichment_result, dict) and "error" not in enrichment_result
            else enrichment_result
            if isinstance(enrichment_result, dict)
            else {}
        ),
        "note": "This enrichment summary combines Reactome recurrence, approximate enrichment scoring, and STRING network aggregation.",
    }


@rate_limited("default")
async def rnaseq_deconvolution(
    expression_profile: dict[str, float] | None = None,
    ranked_genes: list[str] | None = None,
    max_cell_types: int = 5,
) -> dict[str, Any]:
    expression_profile = expression_profile or {}
    ranked_genes = ranked_genes or []
    max_cell_types = BioValidator.clamp_int(max_cell_types, 2, 10, "max_cell_types")

    upper_expression = {gene.upper(): float(value) for gene, value in expression_profile.items()}
    ranked_upper = {gene.upper() for gene in ranked_genes}

    scores: list[dict[str, Any]] = []
    for cell_type, markers in _CELL_TYPE_MARKERS.items():
        marker_values = [upper_expression.get(marker, 0.0) for marker in markers]
        ranked_hits = sum(1 for marker in markers if marker in ranked_upper)
        raw_score = sum(marker_values) + ranked_hits * 2.0
        scores.append(
            {
                "cell_type": cell_type,
                "raw_score": round(raw_score, 3),
                "supporting_markers": [
                    marker
                    for marker in markers
                    if upper_expression.get(marker, 0.0) > 0 or marker in ranked_upper
                ],
            }
        )

    scores.sort(key=lambda item: item["raw_score"], reverse=True)
    total = sum(item["raw_score"] for item in scores) or 1.0
    for item in scores:
        item["estimated_fraction"] = round(item["raw_score"] / total, 3)

    return {
        "method": "marker_based_heuristic",
        "top_cell_types": scores[:max_cell_types],
        "limitations": [
            "This is a heuristic marker-based estimate, not a substitute for CIBERSORTx or reference-based deconvolution.",
            "Results depend on the quality and scale of the submitted expression profile.",
        ],
    }


@cached("default")
@rate_limited("default")
@with_retry(max_attempts=3)
async def structural_similarity(
    query: str = "",
    smiles: str = "",
    threshold: int = 90,
    max_results: int = 10,
) -> dict[str, Any]:
    client = await get_http_client()
    max_results = BioValidator.clamp_int(max_results, 1, 25, "max_results")
    threshold = BioValidator.clamp_int(threshold, 70, 99, "threshold")

    if not query and not smiles:
        raise ValueError("Provide either query or smiles.")

    cid = ""
    if query:
        cid_resp = await client.get(f"{PUBCHEM_BASE}/compound/name/{quote(query)}/cids/JSON")
        if cid_resp.status_code == 200:
            cid = str((cid_resp.json().get("IdentifierList") or {}).get("CID", [""])[0])

    if smiles and not cid:
        cid_resp = await client.get(f"{PUBCHEM_BASE}/compound/smiles/{quote(smiles)}/cids/JSON")
        if cid_resp.status_code == 200:
            cid = str((cid_resp.json().get("IdentifierList") or {}).get("CID", [""])[0])

    if not cid and query and not smiles:
        return {
            "query": query,
            "threshold": threshold,
            "matches": [],
            "note": "No PubChem compound could be resolved from the provided query.",
        }

    similarity_path = (
        f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/cid/{cid}/cids/JSON"
        if cid
        else f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/smiles/{quote(smiles)}/cids/JSON"
    )
    sim_resp = await client.get(
        similarity_path,
        params={"Threshold": threshold, "MaxRecords": max_results},
    )
    sim_resp.raise_for_status()
    cids = ((sim_resp.json().get("IdentifierList") or {}).get("CID")) or []
    if not cids:
        return {"query": query or smiles, "matches": []}

    prop_resp = await client.get(
        f"{PUBCHEM_BASE}/compound/cid/{','.join(str(cid_value) for cid_value in cids[:max_results])}"
        "/property/Title,CanonicalSMILES,MolecularFormula,MolecularWeight,XLogP,TPSA/JSON"
    )
    prop_resp.raise_for_status()
    props = (prop_resp.json().get("PropertyTable") or {}).get("Properties", [])

    return {
        "query": query or smiles,
        "threshold": threshold,
        "matches": [
            {
                "cid": prop.get("CID"),
                "title": prop.get("Title", ""),
                "canonical_smiles": prop.get("CanonicalSMILES", ""),
                "molecular_formula": prop.get("MolecularFormula", ""),
                "molecular_weight": prop.get("MolecularWeight"),
                "xlogp": prop.get("XLogP"),
                "tpsa": prop.get("TPSA"),
                "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{prop.get('CID')}",
            }
            for prop in props
        ],
    }


@cached("omim")
@rate_limited("default")
async def rare_disease_diagnosis(
    phenotype_terms: list[str] | None = None,
    gene_symbol: str = "",
    max_results: int = 10,
) -> dict[str, Any]:
    from biomcp.tools.databases import get_omim_gene_diseases
    from biomcp.tools.pathways import get_gene_disease_associations

    phenotype_terms = phenotype_terms or []
    max_results = BioValidator.clamp_int(max_results, 1, 20, "max_results")

    normalized_terms = [term.strip() for term in phenotype_terms if term.strip()]
    if not normalized_terms and not gene_symbol:
        raise ValueError("Provide phenotype_terms or gene_symbol.")
    phenotype_tokens = _token_set(normalized_terms)
    matched_hpo_terms = [
        {
            "query": term,
            "hpo_search_url": f"https://hpo.jax.org/app/browse/search?q={quote(term)}",
        }
        for term in normalized_terms
    ]

    differentials: list[dict[str, Any]] = []
    if gene_symbol:
        omim_result, open_targets_result = await asyncio.gather(
            get_omim_gene_diseases(gene_symbol),
            get_gene_disease_associations(gene_symbol, max_results=max_results),
            return_exceptions=True,
        )

        omim = omim_result if isinstance(omim_result, dict) else {"diseases": []}
        for disease in omim.get("diseases", []):
            disease_tokens = _token_set([disease.get("phenotype", "")])
            overlap = len(phenotype_tokens & disease_tokens)
            differentials.append(
                {
                    "disease_name": disease.get("phenotype", ""),
                    "source": "OMIM",
                    "omim_id": disease.get("omim_id", ""),
                    "inheritance_pattern": disease.get("inheritance_pattern", ""),
                    "support_url": disease.get("omim_url", ""),
                    "evidence_score": 1.0,
                    "phenotype_overlap_score": overlap,
                }
            )

        open_targets = open_targets_result if isinstance(open_targets_result, dict) else {"associations": []}
        for association in open_targets.get("associations", []):
            disease_name = association.get("disease_name", "")
            if not disease_name:
                continue
            disease_tokens = _token_set([disease_name, association.get("description", "")])
            overlap = len(phenotype_tokens & disease_tokens)
            differentials.append(
                {
                    "disease_name": disease_name,
                    "source": "Open Targets",
                    "therapeutic_areas": association.get("therapeutic_areas", []),
                    "support_url": association.get("url", ""),
                    "evidence_score": association.get("overall_score", 0.0),
                    "phenotype_overlap_score": overlap,
                }
            )

        differentials.sort(
            key=lambda item: (
                item.get("phenotype_overlap_score", 0),
                item.get("evidence_score", 0.0),
            ),
            reverse=True,
        )

    return {
        "gene": gene_symbol or None,
        "phenotype_terms": normalized_terms,
        "matched_hpo_terms": matched_hpo_terms,
        "differential_diagnosis": differentials[:max_results],
        "recommendation": (
            "Use the phenotype-term matches to normalize the case with HPO, then review OMIM and ClinVar evidence "
            "for the highest-ranked gene-disease pairs."
        ),
    }


@cached("ensembl")
@rate_limited("ensembl")
@with_retry(max_attempts=3)
async def genome_browser_snapshot(
    gene_symbol: str = "",
    region: str = "",
    flank_bp: int = 25000,
    assembly: str = "GRCh38",
) -> dict[str, Any]:
    client = await get_http_client()
    flank_bp = BioValidator.clamp_int(flank_bp, 1000, 500000, "flank_bp")
    api_base = ENSEMBL_GRCH37_BASE if assembly.upper() == "GRCH37" else ENSEMBL_BASE

    chromosome = ""
    start = 0
    end = 0
    label = gene_symbol or region
    if gene_symbol:
        lookup = await client.get(
            f"{api_base}/lookup/symbol/homo_sapiens/{gene_symbol}",
            params={"expand": 0},
            headers={"Accept": "application/json"},
        )
        lookup.raise_for_status()
        data = lookup.json()
        chromosome = str(data.get("seq_region_name", ""))
        start = int(data.get("start", 0))
        end = int(data.get("end", 0))
    elif region:
        match = re.match(r"^(?:chr)?(?P<chrom>[A-Za-z0-9_]+):(?P<start>\d+)-(?P<end>\d+)$", region)
        if not match:
            raise ValueError("region must look like 'chr17:43044295-43170245'.")
        chromosome = match.group("chrom")
        start = int(match.group("start"))
        end = int(match.group("end"))
    else:
        raise ValueError("Provide gene_symbol or region.")

    view_start = max(1, start - flank_bp)
    view_end = end + flank_bp
    region_string = f"{chromosome}:{view_start}-{view_end}"
    ucsc_db = "hg38" if assembly.upper() == "GRCH38" else "hg19"
    region_features = await _fetch_ensembl_region_features(region_string, api_base)

    return {
        "label": label,
        "assembly": assembly,
        "region": region_string,
        "browser_links": {
            "ensembl": f"https://www.ensembl.org/Homo_sapiens/Location/View?r={quote(region_string)}",
            "ucsc": f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={ucsc_db}&position={quote(region_string)}",
        },
        "suggested_tracks": [
            "Reference genes",
            "ClinVar variants",
            "gnomAD population variation",
            "Conservation",
            "Regulatory annotations",
        ],
        "region_feature_summary": region_features,
        "snapshot_note": "Use the browser links to generate a visual locus snapshot for reports or reviews.",
    }
