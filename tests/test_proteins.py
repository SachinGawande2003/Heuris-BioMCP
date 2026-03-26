"""
Tests — Protein Tools (Mocked HTTP)
=====================================
Unit tests for UniProt, AlphaFold, and PDB search.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_protein_info_parses_uniprot(mock_http_client, mock_http_response):
    """get_protein_info should parse UniProt JSON into structured protein data."""
    uniprot_resp = mock_http_response(json_data={
        "primaryAccession": "P04637",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Cellular tumor antigen p53"},
                "shortNames": [{"value": "p53"}],
            }
        },
        "genes": [{"geneName": {"value": "TP53"}}],
        "organism": {"scientificName": "Homo sapiens"},
        "sequence": {"length": 393, "molWeight": 43653, "value": "MEEPQ..."},
        "comments": [
            {"commentType": "FUNCTION", "texts": [{"value": "Acts as tumor suppressor."}]},
            {"commentType": "SUBCELLULAR LOCATION",
             "subcellularLocations": [{"location": {"value": "Nucleus"}}]},
        ],
        "features": [
            {"type": "Domain", "description": "Transactivation",
             "location": {"start": {"value": 1}, "end": {"value": 61}}},
        ],
        "uniProtKBCrossReferences": [
            {"database": "GO", "id": "GO:0005634",
             "properties": [{"key": "GoTerm", "value": "C:nucleus"}, {"key": "GoEvidenceType", "value": "IDA"}]},
        ],
    })

    mock_http_client.get = AsyncMock(return_value=uniprot_resp)

    with patch("biomcp.tools.proteins.get_http_client", return_value=mock_http_client):
        from biomcp.tools.proteins import get_protein_info
        result = await get_protein_info.__wrapped__.__wrapped__.__wrapped__("P04637")

    assert result["accession"] == "P04637"
    assert result["full_name"] == "Cellular tumor antigen p53"
    assert "TP53" in result["gene_names"]
    assert result["length"] == 393
    assert result["reviewed"] is True
    assert len(result["go_terms"]) >= 1


@pytest.mark.asyncio
async def test_get_alphafold_structure_not_found(mock_http_client, mock_http_response):
    """AlphaFold should return graceful error for unknown accession."""
    not_found = mock_http_response(status_code=404)
    not_found.raise_for_status = lambda: None  # override — we handle 404 explicitly

    mock_http_client.get = AsyncMock(return_value=not_found)

    with patch("biomcp.tools.proteins.get_http_client", return_value=mock_http_client):
        from biomcp.tools.proteins import get_alphafold_structure
        result = await get_alphafold_structure.__wrapped__.__wrapped__.__wrapped__("XXXXXX")

    assert "error" in result


@pytest.mark.asyncio
async def test_search_pdb_structures_parses_rcsb(mock_http_client, mock_http_response):
    """search_pdb_structures should query RCSB and fetch entry details."""
    search_resp = mock_http_response(json_data={
        "result_set": [{"identifier": "1TUP"}, {"identifier": "2OCJ"}],
        "total_count": 2,
    })
    detail_resp = mock_http_response(json_data={
        "struct": {"title": "P53 core domain"},
        "exptl": [{"method": "X-RAY DIFFRACTION"}],
        "refine": [{"ls_d_res_high": 2.2}],
        "rcsb_accession_info": {"deposit_date": "2020-01-01", "initial_release_date": "2020-03-01"},
        "rcsb_entry_info": {"organism_name": "Homo sapiens", "deposited_polymer_entity_instance_count": 4},
    })

    mock_http_client.post = AsyncMock(return_value=search_resp)
    mock_http_client.get = AsyncMock(return_value=detail_resp)

    with patch("biomcp.tools.proteins.get_http_client", return_value=mock_http_client):
        from biomcp.tools.proteins import search_pdb_structures
        result = await search_pdb_structures.__wrapped__.__wrapped__.__wrapped__("TP53", max_results=5)

    assert result["total_found"] == 2
    assert len(result["structures"]) >= 1
    assert result["structures"][0]["pdb_id"] == "1TUP"


# ── Integration ──────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_protein_info_live():
    from biomcp.tools.proteins import get_protein_info
    result = await get_protein_info("P04637")
    assert result["accession"] == "P04637"
    assert result["length"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alphafold_structure_live():
    from biomcp.tools.proteins import get_alphafold_structure
    result = await get_alphafold_structure("P04637")
    assert "error" in result or "plddt_summary" in result
