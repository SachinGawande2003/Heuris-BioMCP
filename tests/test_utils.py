"""
Tests — Utility Layer
======================
Covers BioValidator, TTLCache, RateLimiter, and make_cache_key.
"""

from __future__ import annotations

import pytest

from biomcp.utils import BioValidator, get_cache, make_cache_key


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
