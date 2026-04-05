from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_query_adverse_events_builds_correct_fda_query(
    mock_http_client,
    mock_http_response,
):
    events_resp = mock_http_response(
        json_data={
            "results": [
                {"term": "Myocardial infarction", "count": 12},
                {"term": "Atrial fibrillation", "count": 7},
            ]
        }
    )
    count_resp = mock_http_response(json_data={"meta": {"results": {"total": 100}}})
    outcome_responses = [
        mock_http_response(json_data={"meta": {"results": {"total": total}}})
        for total in (4, 3, 10, 2, 1, 6)
    ]
    sex_responses = [
        mock_http_response(json_data={"meta": {"results": {"total": 30}}}),
        mock_http_response(json_data={"meta": {"results": {"total": 70}}}),
    ]
    yearly_resp = mock_http_response(
        json_data={"results": [{"time": "20250101", "count": 20}]}
    )

    mock_http_client.get = AsyncMock(
        side_effect=[events_resp, count_resp, *outcome_responses, *sex_responses, yearly_resp]
    )

    with patch("biomcp.tools.drug_safety.get_http_client", return_value=mock_http_client):
        from biomcp.tools.drug_safety import query_adverse_events

        result = await query_adverse_events.__wrapped__.__wrapped__.__wrapped__(
            "aspirin",
            event_type="cardiac",
            serious_only=True,
            max_results=25,
            patient_sex="female",
            age_group="elderly",
        )

    first_call = mock_http_client.get.await_args_list[0]
    params = first_call.kwargs["params"]
    search_q = params["search"]

    assert "patient.drug.medicinalproduct:\"aspirin\"" in search_q
    assert 'serious:"1"' in search_q
    assert 'patient.patientsex:"2"' in search_q
    assert "patient.patientonsetage:[65 TO 120]" in search_q
    assert search_q.count("patient.reaction.reactionmeddrapt:") == 5
    assert 'patient.reaction.reactionmeddrapt:"myocardial infarction"' in search_q
    assert 'patient.reaction.reactionmeddrapt:"atrial fibrillation"' in search_q
    assert params["limit"] == 25

    assert result["total_reports"] == 100
    assert result["filters"]["serious_only"] is True
    assert result["filters"]["sex"] == "female"
    assert result["filters"]["age_group"] == "elderly"
    assert result["outcomes"]["deaths"] == 4
    assert result["sex_distribution"]["female"] == 70
    assert result["top_reactions"][0]["reaction"] == "Myocardial infarction"
    assert result["top_reactions"][0]["pct_of_total"] == 12.0
