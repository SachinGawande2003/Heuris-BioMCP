"""
Tests — NCBI Tools (Mocked HTTP)
==================================
Unit tests for PubMed search, Gene info, and BLAST.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest


# ── PubMed Search ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_pubmed_parses_results(mock_http_client, mock_http_response):
    """search_pubmed should parse esearch + efetch XML into structured articles."""
    # Mock esearch response (returns IDs)
    esearch_resp = mock_http_response(
        json_data={"esearchresult": {"idlist": ["39000001", "39000002"], "count": "2"}}
    )
    # Mock efetch response (returns XML)
    efetch_resp = mock_http_response(text="""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>39000001</PMID>
      <Article>
        <ArticleTitle>CRISPR TP53 Correction</ArticleTitle>
        <Abstract><AbstractText>A breakthrough study.</AbstractText></Abstract>
        <Journal><Title>Nature</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        <AuthorList><Author><LastName>Smith</LastName><ForeName>J</ForeName></Author></AuthorList>
        <ArticleIdList><ArticleId IdType="doi">10.1038/test</ArticleId></ArticleIdList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>""")

    mock_http_client.get = AsyncMock(side_effect=[esearch_resp, efetch_resp])

    with patch("biomcp.tools.ncbi.get_http_client", return_value=mock_http_client):
        from biomcp.tools.ncbi import search_pubmed
        # Clear any cached result
        result = await search_pubmed.__wrapped__.__wrapped__.__wrapped__("TP53 CRISPR", max_results=5)

    assert result["total_found"] == 2
    assert len(result["articles"]) >= 1
    assert result["articles"][0]["pmid"] == "39000001"


# ── Gene Info ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_gene_info_parses_response(mock_http_client, mock_http_response):
    """get_gene_info should return structured gene data from NCBI Gene."""
    esearch_resp = mock_http_response(
        json_data={"esearchresult": {"idlist": ["7157"], "count": "1"}}
    )
    esummary_resp = mock_http_response(
        json_data={
            "result": {
                "uids": ["7157"],
                "7157": {
                    "uid": "7157",
                    "name": "TP53",
                    "description": "tumor protein p53",
                    "organism": {"scientificname": "Homo sapiens"},
                    "chromosome": "17",
                    "maplocation": "17p13.1",
                    "summary": "This gene encodes a tumor suppressor protein.",
                    "otheraliases": "p53, LFS1",
                    "genomicinfo": [{"chrloc": "17", "chrstart": 7668421, "chrstop": 7687490}],
                },
            }
        }
    )

    mock_http_client.get = AsyncMock(side_effect=[esearch_resp, esummary_resp])

    with patch("biomcp.tools.ncbi.get_http_client", return_value=mock_http_client):
        from biomcp.tools.ncbi import get_gene_info
        result = await get_gene_info.__wrapped__.__wrapped__.__wrapped__("TP53")

    assert result["symbol"] == "TP53"
    assert result["chromosome"] == "17"
    assert "tumor" in result.get("summary", "").lower()


# ── Integration (Network Required) ──────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pubmed_search_live():
    from biomcp.tools.ncbi import search_pubmed
    result = await search_pubmed("TP53 tumor suppressor", max_results=3)
    assert result["total_found"] > 0
    assert len(result["articles"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_info_live():
    from biomcp.tools.ncbi import get_gene_info
    result = await get_gene_info("TP53")
    assert result.get("symbol", "").upper() in ("TP53", "P53")
