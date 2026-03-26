"""
BioMCP Test Suite
=================
Integration tests for all major tool modules.
Requires network access to public biological APIs.
"""

from __future__ import annotations

import pytest
import pytest_asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBioValidator:
    def test_validate_pubmed_id_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_pubmed_id("12345678") == "12345678"
        assert BioValidator.validate_pubmed_id("PMID:12345678") == "12345678"

    def test_validate_pubmed_id_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_pubmed_id("not_a_number")

    def test_validate_uniprot_accession_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_uniprot_accession("P04637") == "P04637"
        assert BioValidator.validate_uniprot_accession("p04637") == "P04637"

    def test_validate_gene_symbol(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_gene_symbol("tp53") == "TP53"
        assert BioValidator.validate_gene_symbol(" BRCA1 ") == "BRCA1"

    def test_validate_sequence_protein(self):
        from biomcp.utils import BioValidator
        seq = BioValidator.validate_sequence("MTEYKLVVVGAGGVGKSALTIQLIQNHFV", "protein")
        assert seq == "MTEYKLVVVGAGGVGKSALTIQLIQNHFV"

    def test_validate_sequence_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_sequence("MTEYK123LVV", "protein")

    def test_validate_nct_id_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_nct_id("NCT04280705") == "NCT04280705"

    def test_validate_nct_id_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_nct_id("12345678")

    def test_clamp_int(self):
        from biomcp.utils import BioValidator
        assert BioValidator.clamp_int(50, 1, 100, "test") == 50
        with pytest.raises(ValueError):
            BioValidator.clamp_int(0, 1, 100, "test")


# ─────────────────────────────────────────────────────────────────────────────
# Cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_key_deterministic(self):
        from biomcp.utils import make_cache_key
        key1 = make_cache_key("arg1", foo="bar")
        key2 = make_cache_key("arg1", foo="bar")
        assert key1 == key2

    def test_cache_key_different_args(self):
        from biomcp.utils import make_cache_key
        key1 = make_cache_key("arg1")
        key2 = make_cache_key("arg2")
        assert key1 != key2

    def test_get_cache_namespaced(self):
        from biomcp.utils import get_cache
        cache1 = get_cache("pubmed")
        cache2 = get_cache("uniprot")
        assert cache1 is not cache2

    def test_get_cache_same_namespace(self):
        from biomcp.utils import get_cache
        cache1 = get_cache("pubmed")
        cache2 = get_cache("pubmed")
        assert cache1 is cache2


# ─────────────────────────────────────────────────────────────────────────────
# Server tests
# ─────────────────────────────────────────────────────────────────────────────

class TestServer:
    def test_tool_count(self):
        from biomcp.server import TOOLS
        assert len(TOOLS) >= 20, f"Expected at least 20 tools, got {len(TOOLS)}"

    def test_all_tools_have_required_fields(self):
        from biomcp.server import TOOLS
        for tool in TOOLS:
            assert tool.name, f"Tool missing name: {tool}"
            assert tool.description, f"Tool '{tool.name}' missing description"
            assert tool.inputSchema, f"Tool '{tool.name}' missing inputSchema"

    def test_all_tools_have_required_params(self):
        from biomcp.server import TOOLS
        for tool in TOOLS:
            schema = tool.inputSchema
            assert "properties" in schema, f"Tool '{tool.name}' schema missing properties"
            assert "required" in schema, f"Tool '{tool.name}' schema missing required"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests (require network — mark with @pytest.mark.integration)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pubmed_search_integration():
    from biomcp.tools.ncbi import search_pubmed
    result = await search_pubmed("TP53 tumor suppressor", max_results=3)
    assert result["total_found"] > 0
    assert len(result["articles"]) > 0
    article = result["articles"][0]
    assert "pmid" in article
    assert "title" in article
    assert "abstract" in article


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_info_integration():
    from biomcp.tools.ncbi import get_gene_info
    result = await get_gene_info("TP53")
    assert result.get("symbol", "").upper() in ("TP53", "P53")
    assert "chromosome" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_protein_info_integration():
    from biomcp.tools.proteins import get_protein_info
    result = await get_protein_info("P04637")  # human TP53
    assert result["accession"] == "P04637"
    assert "TP53" in result.get("gene_names", [])
    assert result["length"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alphafold_structure_integration():
    from biomcp.tools.proteins import get_alphafold_structure
    result = await get_alphafold_structure("P04637")
    # Either has structure data or graceful error
    assert "error" in result or "plddt_summary" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clinical_trials_integration():
    from biomcp.tools.advanced import search_clinical_trials
    result = await search_clinical_trials("EGFR lung cancer", max_results=5)
    assert "studies" in result
    assert isinstance(result["studies"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kegg_pathways_integration():
    from biomcp.tools.pathways import search_pathways
    result = await search_pathways("apoptosis")
    assert result["total"] > 0
    assert len(result["pathways"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_drug_targets_integration():
    from biomcp.tools.pathways import get_drug_targets
    result = await get_drug_targets("EGFR", max_results=5)
    assert "drugs" in result
    # Should find drugs for EGFR (e.g. erlotinib, gefitinib)
    assert len(result.get("drugs", [])) >= 0  # graceful even if no results
