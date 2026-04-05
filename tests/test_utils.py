"""
Tests — Utility Layer
======================
Covers BioValidator, TTLCache, RateLimiter, and make_cache_key.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from uuid import uuid4

import pytest

import biomcp.utils as utils
from biomcp.utils import (
    CACHE_TTLS,
    BioValidator,
    attach_response_meta,
    get_cache,
    make_cache_key,
    strip_cache_metadata,
)

# ── BioValidator ─────────────────────────────────────────────────────────────

class TestBioValidator:
    # PubMed IDs
    def test_validate_pubmed_id_plain(self):
        assert BioValidator.validate_pubmed_id("12345678") == "12345678"

    def test_validate_pubmed_id_prefixed(self):
        assert BioValidator.validate_pubmed_id("PMID:12345678") == "12345678"

    def test_validate_pubmed_id_invalid(self):
        with pytest.raises(ValueError, match="PubMed"):
            BioValidator.validate_pubmed_id("not_a_number")

    # UniProt accessions
    def test_validate_uniprot_accession(self):
        assert BioValidator.validate_uniprot_accession("P04637") == "P04637"

    def test_validate_uniprot_accession_lowercase(self):
        assert BioValidator.validate_uniprot_accession("p04637") == "P04637"

    def test_validate_uniprot_accession_invalid(self):
        with pytest.raises(ValueError, match="UniProt"):
            BioValidator.validate_uniprot_accession("INVALID!")

    # Gene symbols
    def test_validate_gene_symbol_case(self):
        assert BioValidator.validate_gene_symbol("tp53") == "TP53"

    def test_validate_gene_symbol_strip(self):
        assert BioValidator.validate_gene_symbol("  BRCA1  ") == "BRCA1"

    def test_validate_gene_symbol_empty(self):
        with pytest.raises(ValueError, match="[Gg]ene"):
            BioValidator.validate_gene_symbol("")

    # NCT IDs
    def test_validate_nct_id_valid(self):
        assert BioValidator.validate_nct_id("NCT04280705") == "NCT04280705"

    def test_validate_nct_id_invalid(self):
        with pytest.raises(ValueError, match="NCT"):
            BioValidator.validate_nct_id("12345678")

    # Sequences
    def test_validate_sequence_protein(self):
        seq = BioValidator.validate_sequence("MTEYKLVVVGAGGVGKSALTIQLIQNHFV", "protein")
        assert seq == "MTEYKLVVVGAGGVGKSALTIQLIQNHFV"

    def test_validate_sequence_dna(self):
        seq = BioValidator.validate_sequence("ATCGATCG", "dna")
        assert seq == "ATCGATCG"

    def test_validate_sequence_invalid_protein(self):
        with pytest.raises(ValueError, match="[Ss]equence"):
            BioValidator.validate_sequence("MTEYK123LVV", "protein")

    # Clamping
    def test_clamp_int_in_range(self):
        assert BioValidator.clamp_int(50, 1, 100, "test") == 50

    def test_clamp_int_at_bounds(self):
        assert BioValidator.clamp_int(1, 1, 100, "test") == 1
        assert BioValidator.clamp_int(100, 1, 100, "test") == 100

    def test_clamp_int_out_of_range(self):
        with pytest.raises(ValueError):
            BioValidator.clamp_int(0, 1, 100, "test")


# ── Cache ────────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_key_deterministic(self):
        key1 = make_cache_key("arg1", foo="bar")
        key2 = make_cache_key("arg1", foo="bar")
        assert key1 == key2

    def test_cache_key_different_args(self):
        key1 = make_cache_key("arg1")
        key2 = make_cache_key("arg2")
        assert key1 != key2

    def test_cache_key_different_kwargs(self):
        key1 = make_cache_key("arg1", a=1)
        key2 = make_cache_key("arg1", a=2)
        assert key1 != key2

    def test_get_cache_namespaced(self):
        cache1 = get_cache("pubmed")
        cache2 = get_cache("uniprot")
        assert cache1 is not cache2

    def test_get_cache_same_namespace_singleton(self):
        cache1 = get_cache("pubmed")
        cache2 = get_cache("pubmed")
        assert cache1 is cache2

    def test_all_cached_namespaces_have_explicit_ttls(self):
        used_namespaces: set[str] = set()
        pattern = re.compile(r'@cached\("([^"]+)"\)')
        for path in Path("src/biomcp").rglob("*.py"):
            used_namespaces.update(pattern.findall(path.read_text(encoding="utf-8")))

        missing = sorted(namespace for namespace in used_namespaces if namespace not in CACHE_TTLS)
        assert missing == []

    def test_cache_ttl_anchors_for_reviewed_sources(self):
        assert CACHE_TTLS["fda"] == 3_600
        assert CACHE_TTLS["chembl_registry"] == 604_800
        assert CACHE_TTLS["clinical_trials"] == 1_800
        assert CACHE_TTLS["multi_omics"] == 3_600
        assert CACHE_TTLS["variant"] == 86_400
        assert CACHE_TTLS["gtex"] == 604_800
        assert CACHE_TTLS["tcga"] == 604_800

    @pytest.mark.asyncio
    async def test_cached_decorator_adds_fingerprint_and_status(self):
        namespace = f"test-cache-{uuid4().hex}"

        @utils.cached(namespace)
        async def load_value(value: int) -> dict[str, int]:
            return {"value": value}

        try:
            first = await load_value(7)
            second = await load_value(7)
        finally:
            utils._CACHES.pop(namespace, None)

        assert first["value"] == 7
        assert first["_cache"]["status"] == "fresh"
        assert first["_cache"]["namespace"] == namespace
        assert first["_cache"]["ttl_s"] == CACHE_TTLS["default"]

        fingerprint = first["_cache"]["fingerprint"]
        first["value"] = 999

        assert second["value"] == 7
        assert second["_cache"]["status"] == "cached"
        assert second["_cache"]["fingerprint"] == fingerprint
        assert second["_cache"]["cache_key"] == first["_cache"]["cache_key"]
        assert second["_cache"]["cached_at"] == first["_cache"]["cached_at"]
        assert second["_cache"]["age_s"] >= 0.0
        assert second["_cache"]["expires_in_s"] <= CACHE_TTLS["default"]

    def test_strip_cache_metadata_removes_nested_cache_fields(self):
        payload = {
            "gene": "EGFR",
            "_cache": {"status": "fresh"},
            "layers": {
                "genomics": {"symbol": "EGFR", "_cache": {"status": "cached"}},
                "literature": [
                    {"pmid": "1", "_cache": {"status": "cached"}},
                    {"pmid": "2"},
                ],
            },
        }

        stripped = strip_cache_metadata(payload)

        assert "_cache" not in stripped
        assert "_cache" not in stripped["layers"]["genomics"]
        assert all("_cache" not in article for article in stripped["layers"]["literature"])

    def test_attach_response_meta_adds_confidence_block(self):
        payload = attach_response_meta(
            "verify_biological_claim",
            {
                "confidence_score": 0.81,
                "evidence_counts": {"supporting": 3, "contradicting": 0},
                "databases_queried": ["PubMed", "Open Targets", "ClinVar"],
            },
        )

        assert payload["_meta"]["response_scope"] == "tool_output"
        assert payload["_meta"]["confidence"] >= 0.75
        assert "confidence_factors" in payload["_meta"]


@pytest.mark.asyncio
async def test_get_http_client_recreates_client_for_new_event_loop(monkeypatch: pytest.MonkeyPatch):
    class FakeClient:
        def __init__(self):
            self.is_closed = False

        async def aclose(self):
            self.is_closed = True

    created: list[FakeClient] = []

    def fake_async_client(**kwargs):
        client = FakeClient()
        created.append(client)
        return client

    stale_client = FakeClient()

    monkeypatch.setattr(utils.httpx, "AsyncClient", fake_async_client)
    monkeypatch.setattr(utils, "_HTTP_CLIENT", stale_client)
    monkeypatch.setattr(utils, "_HTTP_CLIENT_LOOP_ID", -1)
    monkeypatch.setattr(utils, "_HTTP_LOCK", None)

    client = await utils.get_http_client()
    await asyncio.sleep(0)

    assert client is created[0]
    assert client is not stale_client
    assert stale_client.is_closed is True

    await utils.close_http_client()
