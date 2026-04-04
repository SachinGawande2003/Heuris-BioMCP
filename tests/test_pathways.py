"""
Tests — Pathway & Drug Tools (Mocked HTTP)
============================================
Unit tests for KEGG, Reactome, ChEMBL, and Open Targets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from tenacity import wait_none


@pytest.mark.asyncio
async def test_search_pathways_parses_kegg(mock_http_client, mock_http_response):
    """search_pathways should parse KEGG tab-separated response."""
    kegg_resp = mock_http_response(
        text=("path:map04210\tApoptosis\npath:map04215\tApoptosis - multiple species\n")
    )

    mock_http_client.get = AsyncMock(return_value=kegg_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import search_pathways

        result = await search_pathways.__wrapped__.__wrapped__.__wrapped__("apoptosis")

    assert result["total"] == 2
    assert result["pathways"][0]["pathway_id"] == "map04210"
    assert "Apoptosis" in result["pathways"][0]["description"]
    assert "kegg.jp" in result["pathways"][0]["viewer_url"]


@pytest.mark.asyncio
async def test_get_kegg_gene_pathways_resolves_gene_membership(mock_http_client, mock_http_response):
    conv_resp = mock_http_response(text="ncbi-geneid:1956\thsa:1956\n")
    link_resp = mock_http_response(text="hsa:1956\tpath:hsa04010\nhsa:1956\tpath:hsa04012\n")
    detail_resp = mock_http_response(
        text=(
            "ENTRY       hsa04010                    Pathway\n"
            "NAME        MAPK signaling pathway - Homo sapiens (human)\n"
            "DESCRIPTION MAPK cascade.\n"
            "CLASS       Environmental Information Processing; Signal transduction\n"
            "///\n"
            "ENTRY       hsa04012                    Pathway\n"
            "NAME        ErbB signaling pathway - Homo sapiens (human)\n"
            "DESCRIPTION EGFR family signaling.\n"
            "CLASS       Environmental Information Processing; Signal transduction\n"
            "///\n"
        )
    )
    mock_http_client.get = AsyncMock(side_effect=[conv_resp, link_resp, detail_resp])

    async def fake_gene_info(gene_symbol: str, organism: str = "homo sapiens"):
        return {"gene_id": "1956", "symbol": gene_symbol, "organism": organism}

    with (
        patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.ncbi.get_gene_info", side_effect=fake_gene_info),
    ):
        from biomcp.tools.pathways import get_kegg_gene_pathways

        result = await get_kegg_gene_pathways.__wrapped__.__wrapped__.__wrapped__("EGFR")

    assert result["ncbi_gene_id"] == "1956"
    assert result["kegg_gene_ids"] == ["hsa:1956"]
    assert result["total"] == 2
    assert result["pathways"][0]["pathway_id"] == "hsa04010"
    assert result["pathways"][1]["description"].startswith("ErbB signaling pathway")


@pytest.mark.asyncio
async def test_get_drug_targets_no_target(mock_http_client, mock_http_response):
    """get_drug_targets should handle empty ChEMBL target search gracefully."""
    empty_resp = mock_http_response(json_data={"targets": []})
    mock_http_client.get = AsyncMock(return_value=empty_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_drug_targets

        result = await get_drug_targets.__wrapped__.__wrapped__.__wrapped__("FAKEGENE")

    assert result["drugs"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_reactome_pathways_handles_transport_error(mock_http_client):
    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_reactome_pathways

        mock_http_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timed out"))
        retrying_call = get_reactome_pathways.__wrapped__.__wrapped__.retry_with(wait=wait_none())
        with pytest.raises(httpx.ConnectTimeout):
            await retrying_call("EGFR")

    assert mock_http_client.post.await_count == 3


@pytest.mark.asyncio
async def test_get_reactome_pathways_parses_analysis_service(mock_http_client, mock_http_response):
    analysis_resp = mock_http_response(
        json_data={
            "identifiersNotFound": 0,
            "pathways": [
                {
                    "stId": "R-HSA-177929",
                    "name": "Signaling by EGFR",
                    "species": {"taxId": "9606", "name": "Homo sapiens"},
                    "entities": {"found": 1, "total": 64, "pValue": 0.0118, "fdr": 0.0125},
                    "inDisease": False,
                },
                {
                    "stId": "R-MMU-177929",
                    "name": "Mouse EGFR signaling",
                    "species": {"taxId": "10090", "name": "Mus musculus"},
                    "entities": {"found": 1, "total": 40, "pValue": 0.02, "fdr": 0.03},
                    "inDisease": False,
                },
            ],
        }
    )
    mock_http_client.post = AsyncMock(return_value=analysis_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_reactome_pathways

        result = await get_reactome_pathways.__wrapped__.__wrapped__.__wrapped__("EGFR")

    assert result["gene"] == "EGFR"
    assert result["total"] == 1
    assert result["pathways"][0]["reactome_id"] == "R-HSA-177929"
    assert result["pathways"][0]["species"] == "Homo sapiens"
    assert result["pathways"][0]["found_entities"] == 1
    assert result["pathways"][0]["p_value"] == 0.0118
    assert result["pathways"][0]["fdr"] == 0.0125


@pytest.mark.asyncio
async def test_get_drug_targets_handles_chembl_http_error(mock_http_client, mock_http_response):
    error_resp = mock_http_response(status_code=500)
    mock_http_client.get = AsyncMock(return_value=error_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_drug_targets

        retrying_call = get_drug_targets.__wrapped__.__wrapped__.retry_with(wait=wait_none())
        with pytest.raises(httpx.HTTPStatusError):
            await retrying_call("EGFR")

    assert mock_http_client.get.await_count == 3


@pytest.mark.asyncio
async def test_get_compound_info_not_found(mock_http_client, mock_http_response):
    """get_compound_info should handle 404 gracefully."""
    not_found = mock_http_response(status_code=404)
    not_found.raise_for_status = lambda: None
    mock_http_client.get = AsyncMock(return_value=not_found)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_compound_info

        result = await get_compound_info("CHEMBL123456")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_compound_info_parses_molecule(mock_http_client, mock_http_response):
    """get_compound_info should parse ChEMBL molecule JSON."""
    mol_resp = mock_http_response(
        json_data={
            "molecule_chembl_id": "CHEMBL25",
            "pref_name": "ASPIRIN",
            "molecule_type": "Small molecule",
            "max_phase": 4,
            "molecule_properties": {
                "full_molformula": "C9H8O4",
                "full_mwt": "180.16",
                "alogp": "1.31",
                "hbd": "1",
                "hba": "3",
                "psa": "63.60",
                "rtb": "3",
                "num_ro5_violations": "0",
                "qed_weighted": "0.56",
            },
            "molecule_structures": {
                "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "standard_inchi": "InChI=1S/C9H8O4/...",
                "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            },
            "drug_indications": [],
        }
    )

    mock_http_client.get = AsyncMock(return_value=mol_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_compound_info

        result = await get_compound_info.__wrapped__.__wrapped__.__wrapped__("CHEMBL25")

    assert result["chembl_id"] == "CHEMBL25"
    assert result["name"] == "ASPIRIN"
    assert result["drug_approved"] is True
    assert result["molecular_formula"] == "C9H8O4"


# ── Integration ──────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kegg_pathways_live():
    from biomcp.tools.pathways import search_pathways

    result = await search_pathways("apoptosis")
    assert result["total"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kegg_gene_pathways_live():
    from biomcp.tools.pathways import get_kegg_gene_pathways

    result = await get_kegg_gene_pathways("EGFR")
    assert result["total"] > 1
    assert any(pathway["pathway_id"] == "hsa04010" for pathway in result["pathways"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reactome_pathways_live():
    from biomcp.tools.pathways import get_reactome_pathways

    result = await get_reactome_pathways("EGFR")
    assert result["total"] > 0
    assert result["pathways"][0]["url"].startswith("https://reactome.org/content/detail/")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reactome_pathways_live_brca1():
    from biomcp.tools.pathways import get_reactome_pathways

    result = await get_reactome_pathways("BRCA1")
    assert result["total"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_drug_targets_live():
    from biomcp.tools.pathways import get_drug_targets

    result = await get_drug_targets("EGFR", max_results=5)
    assert "drugs" in result
