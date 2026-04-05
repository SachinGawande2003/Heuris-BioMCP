from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_verify_biological_claim_parses_pubmed_sentiment():
    pubmed_result = {
        "articles": [
            {
                "title": "Study confirms EGFR signaling in lung cancer",
                "abstract": "This study demonstrates and confirms strong EGFR-dependent tumor growth.",
                "pmid": "1001",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1001/",
            },
            {
                "title": "Unexpected EGFR finding",
                "abstract": "However, no evidence of the claimed association was observed in this cohort.",
                "pmid": "1002",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1002/",
            },
        ]
    }
    protein_result = {"proteins": [{"genes": ["EGFR"]}]}
    association_result = {
        "associations": [
            {"disease_name": "lung cancer", "overall_score": 0.81},
        ]
    }

    with (
        patch("biomcp.tools.ncbi.search_pubmed", new=AsyncMock(return_value=pubmed_result)),
        patch("biomcp.tools.proteins.search_proteins", new=AsyncMock(return_value=protein_result)),
        patch(
            "biomcp.tools.pathways.get_gene_disease_associations",
            new=AsyncMock(return_value=association_result),
        ),
    ):
        from biomcp.tools.verify import verify_biological_claim

        result = await verify_biological_claim(
            "EGFR drives lung cancer progression",
            context_gene="EGFR",
        )

    assert result["gene_context"] == "EGFR"
    assert result["verdict"] == "VERIFIED"
    assert result["evidence_counts"]["supporting"] == 3
    assert result["evidence_counts"]["contradicting"] == 1
    assert result["supporting_evidence"][0]["source"] == "PubMed"
    assert any(item["source"] == "UniProt Swiss-Prot" for item in result["supporting_evidence"])
    assert any(item["source"] == "Open Targets" for item in result["supporting_evidence"])
    assert result["contradicting_evidence"][0]["source"] == "PubMed"
