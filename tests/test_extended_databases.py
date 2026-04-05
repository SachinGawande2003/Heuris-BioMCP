from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_biogrid_interactions_requires_api_key(
    mock_http_client,
):
    mock_http_client.get = AsyncMock()

    with (
        patch("biomcp.tools.extended_databases.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.extended_databases.os.getenv", return_value=""),
    ):
        from biomcp.tools.extended_databases import get_biogrid_interactions

        result = await get_biogrid_interactions.__wrapped__.__wrapped__.__wrapped__("TP53")

    assert result["gene"] == "TP53"
    assert result["interactions"] == []
    assert "BIOGRID_API_KEY" in result["error"]
    mock_http_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_biogrid_interactions_uses_configured_api_key(
    mock_http_client,
    mock_http_response,
):
    response = mock_http_response(
        json_data={
            "12345": {
                "OFFICIAL_SYMBOL_A": "TP53",
                "OFFICIAL_SYMBOL_B": "MDM2",
                "EXPERIMENTAL_SYSTEM": "Two-hybrid",
                "EXPERIMENTAL_SYSTEM_TYPE": "physical",
                "PUBMED_ID": "12345678|23456789",
            }
        }
    )
    mock_http_client.get = AsyncMock(return_value=response)

    with (
        patch("biomcp.tools.extended_databases.get_http_client", return_value=mock_http_client),
        patch("biomcp.tools.extended_databases.os.getenv", return_value="test-biogrid-key"),
    ):
        from biomcp.tools.extended_databases import get_biogrid_interactions

        result = await get_biogrid_interactions.__wrapped__.__wrapped__.__wrapped__("TP53")

    assert result["returned"] == 1
    assert result["interactions"][0]["partner_gene"] == "MDM2"
    _, kwargs = mock_http_client.get.await_args
    assert kwargs["params"]["accesskey"] == "test-biogrid-key"
