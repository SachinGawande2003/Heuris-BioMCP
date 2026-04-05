"""
Tests — Advanced Tools (Mocked HTTP)
=====================================
Unit tests for ClinicalTrials.gov, GEO, Ensembl, scRNA-seq, neuroimaging.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_clinical_trials_parses_response(mock_http_client, mock_http_response):
    ct_resp = mock_http_response(json_data={
        "totalCount": 1,
        "studies": [{
            "protocolSection": {
                "identificationModule": {"nctId": "NCT04280705", "briefTitle": "KRAS Lung Trial"},
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-01-01"},
                    "primaryCompletionDateStruct": {"date": "2026-12-31"},
                },
                "descriptionModule": {"briefSummary": "Phase 2 study of KRAS inhibitor."},
                "designModule": {
                    "phases": ["PHASE2"], "studyType": "INTERVENTIONAL",
                    "enrollmentInfo": {"count": 120},
                },
                "conditionsModule": {"conditions": ["Non-Small Cell Lung Cancer"]},
                "armsInterventionsModule": {
                    "interventions": [{"interventionName": "KRASi-001", "interventionType": "DRUG"}]
                },
                "eligibilityModule": {"eligibilityCriteria": "Age >= 18"},
                "contactsLocationsModule": {"locations": [{"city": "Boston", "country": "United States"}]},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "NIH"}},
            }
        }],
    })

    mock_http_client.get = AsyncMock(return_value=ct_resp)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import search_clinical_trials
        result = await search_clinical_trials.__wrapped__.__wrapped__.__wrapped__(
            "KRAS lung cancer", max_results=5
        )

    assert result["total_found"] == 1   # FIX: was result["total"]
    assert result["studies"][0]["nct_id"] == "NCT04280705"
    assert result["studies"][0]["status"] == "RECRUITING"
    assert "KRAS" in result["studies"][0]["title"]


@pytest.mark.asyncio
async def test_search_clinical_trials_retries_after_403(mock_http_client, mock_http_response):
    rate_limited = mock_http_response(status_code=403)
    success = mock_http_response(json_data={"totalCount": 0, "studies": []})
    mock_http_client.get = AsyncMock(side_effect=[rate_limited, success])

    with (
        patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.advanced.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        from biomcp.tools.advanced import search_clinical_trials

        result = await search_clinical_trials.__wrapped__.__wrapped__.__wrapped__(
            "EGFR lung cancer", max_results=5
        )

    assert result["total_found"] == 0
    assert result["studies"] == []
    assert mock_http_client.get.await_count == 2
    sleep_mock.assert_awaited_once_with(60)


@pytest.mark.asyncio
async def test_get_trial_details_not_found(mock_http_client, mock_http_response):
    not_found = mock_http_response(status_code=404)
    not_found.raise_for_status = lambda: None
    mock_http_client.get = AsyncMock(return_value=not_found)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import get_trial_details
        result = await get_trial_details.__wrapped__.__wrapped__.__wrapped__("NCT00000000")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_trial_details_retries_after_403(mock_http_client, mock_http_response):
    rate_limited = mock_http_response(status_code=403)
    success = mock_http_response(
        json_data={
            "protocolSection": {
                "outcomesModule": {"primaryOutcomes": [{"measure": "ORR", "timeFrame": "12 months"}]},
                "armsInterventionsModule": {"armGroups": [{"armGroupLabel": "Arm A", "armGroupType": "EXPERIMENTAL"}]},
                "eligibilityModule": {"eligibilityCriteria": "Adults only"},
            }
        }
    )
    mock_http_client.get = AsyncMock(side_effect=[rate_limited, success])

    with (
        patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.advanced.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        from biomcp.tools.advanced import get_trial_details

        result = await get_trial_details.__wrapped__.__wrapped__.__wrapped__("NCT04280705")

    assert result["nct_id"] == "NCT04280705"
    assert result["primary_outcomes"][0]["measure"] == "ORR"
    assert mock_http_client.get.await_count == 2
    sleep_mock.assert_awaited_once_with(60)


@pytest.mark.asyncio
async def test_search_gene_expression_empty(mock_http_client, mock_http_response):
    empty_resp = mock_http_response(
        json_data={"esearchresult": {"idlist": [], "count": "0"}}
    )
    mock_http_client.get = AsyncMock(return_value=empty_resp)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import search_gene_expression
        result = await search_gene_expression.__wrapped__.__wrapped__.__wrapped__("FAKEGENE123")

    assert result["datasets"] == []
    assert result["total_found"] == 0   # FIX: was result["total"]


@pytest.mark.asyncio
async def test_get_gene_variants_not_found(mock_http_client, mock_http_response):
    empty_lookup = mock_http_response(json_data=[])
    mock_http_client.get = AsyncMock(return_value=empty_lookup)

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import get_gene_variants
        result = await get_gene_variants.__wrapped__.__wrapped__.__wrapped__("FAKEGENE")

    assert result["variants"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_gene_variants_handles_ensembl_overlap_error(mock_http_client, mock_http_response):
    xref_resp = mock_http_response(json_data=[{"id": "ENSG00000141510", "type": "gene"}])
    lookup_resp = mock_http_response(
        json_data={"seq_region_name": "17", "start": 7661779, "end": 7687546}
    )
    overlap_error = mock_http_response(status_code=404)
    mock_http_client.get = AsyncMock(side_effect=[xref_resp, lookup_resp, overlap_error])

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import get_gene_variants

        result = await get_gene_variants.__wrapped__.__wrapped__("TP53")

    assert result["gene"] == "TP53"
    assert result["variants"] == []
    assert result["returned"] == 0
    assert "note" in result
    assert mock_http_client.get.await_count == 3


@pytest.mark.asyncio
async def test_query_neuroimaging_handles_failure_gracefully(mock_http_client, mock_http_response):
    mock_http_client.post = AsyncMock(side_effect=Exception("Connection error"))
    mock_http_client.get  = AsyncMock(side_effect=Exception("Connection error"))

    with patch("biomcp.tools.advanced.get_http_client", return_value=mock_http_client):
        from biomcp.tools.advanced import query_neuroimaging_datasets
        result = await query_neuroimaging_datasets("hippocampus")

    assert result["total_found"] == 0
    assert result["datasets"] == []
    assert "recommended_tools" in result


@pytest.mark.asyncio
async def test_multi_omics_gene_report_handles_partial_layer_failures():
    literature = {
        "total_found": 2,
        "articles": [
            {
                "pmid": "4001",
                "title": "MYC review",
                "year": 2025,
                "journal": "Cell",
            }
        ],
    }

    with (
        patch("biomcp.tools.ncbi.get_gene_info", new=AsyncMock(return_value={"symbol": "MYC"})),
        patch("biomcp.tools.ncbi.search_pubmed", new=AsyncMock(return_value=literature)),
        patch(
            "biomcp.tools.pathways.get_reactome_pathways",
            new=AsyncMock(side_effect=RuntimeError("Reactome timeout")),
        ),
        patch(
            "biomcp.tools.pathways.get_drug_targets",
            new=AsyncMock(return_value={"drugs": [{"name": "DrugX"}]}),
        ),
        patch(
            "biomcp.tools.pathways.get_gene_disease_associations",
            new=AsyncMock(return_value={"associations": [{"disease_name": "Lymphoma"}]}),
        ),
        patch(
            "biomcp.tools.advanced.search_gene_expression",
            new=AsyncMock(side_effect=ValueError("GEO offline")),
        ),
        patch(
            "biomcp.tools.advanced.search_clinical_trials",
            new=AsyncMock(return_value={"studies": [{"nct_id": "NCT0001"}]}),
        ),
    ):
        from biomcp.tools.advanced import multi_omics_gene_report

        result = await multi_omics_gene_report.__wrapped__.__wrapped__("MYC")

    assert result["gene"] == "MYC"
    assert result["layers"]["genomics"]["symbol"] == "MYC"
    assert result["layers"]["literature"]["total_publications"] == 2
    assert result["layers"]["literature"]["recent_papers"][0]["pmid"] == "4001"
    assert result["layers"]["reactome"]["status"] == "failed"
    assert result["layers"]["reactome"]["error"] == "Reactome timeout"
    assert result["layers"]["expression"]["status"] == "failed"
    assert result["layers"]["clinical_trials"]["studies"][0]["nct_id"] == "NCT0001"
    assert "status: failed" in result["note"]


# ── Integration ──────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_clinical_trials_live():
    from biomcp.tools.advanced import search_clinical_trials
    result = await search_clinical_trials("EGFR lung cancer", max_results=5)
    assert "studies" in result
    assert isinstance(result["studies"], list)
