"""
Tests — Advanced Tools (Mocked HTTP)
======================================
Unit tests for ClinicalTrials.gov, GEO, Ensembl, scRNA-seq, neuroimaging.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_clinical_trials_parses_response(mock_http_client, mock_http_response):
    """search_clinical_trials should parse ClinicalTrials.gov v2 JSON."""
    ct_resp = mock_http_response(json_data={
        "totalCount": 1,
        "studies": [{
            "protocolSection": {
                "identificationModule": {"nctId": "NCT04280705", "briefTitle": "KRAS Lung Trial"},
                "statusModule": {"overallStatus": "RECRUITING",
                                 "startDateStruct": {"date": "2024-01-01"},
                                 "primaryCompletionDateStruct": {"date": "2026-12-31"}},
                "descriptionModule": {"briefSummary": "Phase 2 study of KRAS inhibitor."},
                "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL",
                                 "enrollmentInfo": {"count": 120}},
                "conditionsModule": {"conditions": ["Non-Small Cell Lung Cancer"]},
                "armsInterventionsModule": {
                    "interventions": [{"interventionName": "KRASi-001", "interventionType": "DRUG"}]
                },
                "eligibilityModule": {"eligibilityCriteria": "Age >= 18"},
                "contactsLocationsModule": {"locations": [{"city": "Boston", "country": "United States"}]},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "NIH"}},
            }
        }]
    })

    mock_http_client.get = AsyncMock(return_value=ct_resp)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import search_clinical_trials
        result = await search_clinical_trials.__wrapped__.__wrapped__.__wrapped__(
            "KRAS lung cancer", max_results=5
        )

    assert result["total_found"] == 1
    assert result["studies"][0]["nct_id"] == "NCT04280705"
    assert result["studies"][0]["status"] == "RECRUITING"
    assert "KRAS" in result["studies"][0]["title"]


@pytest.mark.asyncio
async def test_get_trial_details_not_found(mock_http_client, mock_http_response):
    """get_trial_details should handle 404 gracefully."""
    not_found = mock_http_response(status_code=404)
    not_found.raise_for_status = lambda: None
    mock_http_client.get = AsyncMock(return_value=not_found)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import get_trial_details
        result = await get_trial_details.__wrapped__.__wrapped__.__wrapped__("NCT00000000")

    assert "error" in result



@pytest.mark.asyncio
async def test_search_gene_expression_empty(mock_http_client, mock_http_response):
    """search_gene_expression should handle empty GEO results."""
    empty_resp = mock_http_response(
        json_data={"esearchresult": {"idlist": [], "count": "0"}}
    )
    mock_http_client.get = AsyncMock(return_value=empty_resp)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import search_gene_expression
        result = await search_gene_expression.__wrapped__.__wrapped__.__wrapped__("FAKEGENE123")

    assert result["datasets"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_get_gene_variants_not_found(mock_http_client, mock_http_response):
    """get_gene_variants should handle missing gene in Ensembl gracefully."""
    empty_lookup = mock_http_response(json_data=[])
    mock_http_client.get = AsyncMock(return_value=empty_lookup)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import get_gene_variants
        result = await get_gene_variants.__wrapped__.__wrapped__.__wrapped__("FAKEGENE")

    assert result["variants"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_query_neuroimaging_openneuro_failure(mock_http_client, mock_http_response):
    """Neuroimaging search should handle OpenNeuro/NeuroVault failures gracefully."""
    error_resp = mock_http_response(status_code=500)
    mock_http_client.post = AsyncMock(side_effect=Exception("Connection error"))
    mock_http_client.get = AsyncMock(side_effect=Exception("Connection error"))

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import query_neuroimaging_datasets
        result = await query_neuroimaging_datasets("hippocampus")

    assert result["total_datasets"] == 0
    assert result["datasets"] == []
    assert "recommended_tools" in result


# ── Integration ──────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_clinical_trials_live():
    from biomcp.tools.advanced import search_clinical_trials
    result = await search_clinical_trials("EGFR lung cancer", max_results=5)
    assert "studies" in result
    assert isinstance(result["studies"], list)
