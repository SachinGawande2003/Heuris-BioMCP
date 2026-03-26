"""
BioMCP Test Configuration
=========================
Shared fixtures and pytest configuration.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_http_response():
    """Factory fixture to create mock httpx responses."""

    def _make(status_code: int = 200, json_data: dict | None = None, text: str = ""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = text
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import httpx
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"{status_code}", request=MagicMock(), response=resp
            )
        return resp

    return _make


@pytest.fixture
def mock_http_client(mock_http_response):
    """Mock the shared HTTP client returned by get_http_client()."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_http_response())
    client.post = AsyncMock(return_value=mock_http_response())
    return client


# ── Markers ──────────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that call real external APIs (network required)")
