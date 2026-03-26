"""
Tests — Pathway & Drug Tools (Mocked HTTP)
============================================
Unit tests for KEGG, Reactome, ChEMBL, and Open Targets.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_pathways_parses_kegg(mock_http_client, mock_http_response):
    """search_pathways should parse KEGG tab-separated response."""
    kegg_resp = mock_http_response(text=(
        "path:map04210\tApoptosis\n"
        "path:map04215\tApoptosis - multiple species\n"
    ))

    mock_http_client.get = AsyncMock(return_value=kegg_resp)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import search_pathways
        result = await search_pathways.__wrapped__.__wrapped__.__wrapped__("apoptosis")

    assert result["total"] == 2
    assert result["pathways"][0]["pathway_id"] == "map04210"
    assert "Apoptosis" in result["pathways"][0]["description"]
    assert "kegg.jp" in result["pathways"][0]["viewer_url"]


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
async def test_get_compound_info_not_found(mock_http_client, mock_http_response):
    """get_compound_info should handle 404 gracefully."""
    not_found = mock_http_response(status_code=404)
    not_found.raise_for_status = lambda: None
    mock_http_client.get = AsyncMock(return_value=not_found)

    with patch("biomcp.tools.pathways.get_http_client", return_value=mock_http_client):
        from biomcp.tools.pathways import get_compound_info
        result = await get_compound_info.__wrapped__.__wrapped__.__wrapped__("CHEMBL_FAKE")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_compound_info_parses_molecule(mock_http_client, mock_http_response):
    """get_compound_info should parse ChEMBL molecule JSON."""
    mol_resp = mock_http_response(json_data={
        "molecule_chembl_id": "CHEMBL25",
        "pref_name": "ASPIRIN",
        "molecule_type": "Small molecule",
        "max_phase": 4,
        "molecule_properties": {
            "full_molformula": "C9H8O4",
            "full_mwt": "180.16",
            "alogp": "1.31",
            "hbd": "1", "hba": "3", "psa": "63.60", "rtb": "3",
            "num_ro5_violations": "0", "qed_weighted": "0.56",
        },
        "molecule_structures": {
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "standard_inchi": "InChI=1S/C9H8O4/...",
            "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        },
        "drug_indications": [],
    })

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
async def test_drug_targets_live():
    from biomcp.tools.pathways import get_drug_targets
    result = await get_drug_targets("EGFR", max_results=5)
    assert "drugs" in result
