"""
BioMCP Core Test Suite
=======================
Unit tests (no network) and integration tests (live API, marked @integration).

Run unit tests:
    pytest tests/ -v -m "not integration"

Run integration tests (live API):
    pytest tests/ -v -m integration
"""

from __future__ import annotations

import pytest


class TestBioValidator:
    def test_pubmed_id_plain(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_pubmed_id("12345678") == "12345678"

    def test_pubmed_id_prefixed(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_pubmed_id("PMID:12345678") == "12345678"

    def test_pubmed_id_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_pubmed_id("not_a_number")

    def test_uniprot_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_uniprot_accession("P04637") == "P04637"

    def test_uniprot_lowercase_normalised(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_uniprot_accession("p04637") == "P04637"

    def test_uniprot_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_uniprot_accession("NOTANID")

    def test_gene_symbol_uppercase(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_gene_symbol("tp53") == "TP53"

    def test_gene_symbol_strips_whitespace(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_gene_symbol("  BRCA1  ") == "BRCA1"

    def test_gene_symbol_empty_raises(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError):
            BioValidator.validate_gene_symbol("")

    def test_protein_sequence_valid(self):
        from biomcp.utils import BioValidator
        seq = BioValidator.validate_sequence("MTEYKLVVVGAGGVGKSALT", "protein")
        assert seq == "MTEYKLVVVGAGGVGKSALT"

    def test_nucleotide_sequence_valid(self):
        from biomcp.utils import BioValidator
        seq = BioValidator.validate_sequence("ATGCGATCGA", "nucleotide")
        assert seq == "ATGCGATCGA"

    def test_sequence_invalid_chars(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="invalid"):
            BioValidator.validate_sequence("MTEYK123", "protein")

    def test_sequence_too_short(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="too short"):
            BioValidator.validate_sequence("MTEK", "protein")

    def test_nct_id_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_nct_id("NCT04280705") == "NCT04280705"

    def test_nct_id_lowercase_normalised(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_nct_id("nct04280705") == "NCT04280705"

    def test_nct_id_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="Invalid NCT"):
            BioValidator.validate_nct_id("12345678")

    def test_chembl_id_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_chembl_id("CHEMBL25") == "CHEMBL25"

    def test_chembl_id_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="Invalid ChEMBL"):
            BioValidator.validate_chembl_id("DRUG001")

    def test_kegg_pathway_valid(self):
        from biomcp.utils import BioValidator
        assert BioValidator.validate_kegg_pathway_id("hsa05200") == "hsa05200"

    def test_kegg_pathway_invalid(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="Invalid KEGG"):
            BioValidator.validate_kegg_pathway_id("PATHWAY1")

    def test_clamp_int_within_range(self):
        from biomcp.utils import BioValidator
        assert BioValidator.clamp_int(50, 1, 100, "x") == 50

    def test_clamp_int_below_min_raises(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="between"):
            BioValidator.clamp_int(0, 1, 100, "x")

    def test_clamp_int_above_max_raises(self):
        from biomcp.utils import BioValidator
        with pytest.raises(ValueError, match="between"):
            BioValidator.clamp_int(101, 1, 100, "x")

    def test_clamp_int_wrong_type_raises(self):
        from biomcp.utils import BioValidator
        with pytest.raises(TypeError, match="integer"):
            BioValidator.clamp_int("ten", 1, 100, "x")  # type: ignore


class TestCache:
    def test_cache_key_deterministic(self):
        from biomcp.utils import make_cache_key
        assert make_cache_key("a", b=1) == make_cache_key("a", b=1)

    def test_cache_key_different_args_differ(self):
        from biomcp.utils import make_cache_key
        assert make_cache_key("a") != make_cache_key("b")

    def test_namespace_isolation(self):
        from biomcp.utils import get_cache
        assert get_cache("pubmed") is not get_cache("uniprot")

    def test_same_namespace_same_object(self):
        from biomcp.utils import get_cache
        assert get_cache("pubmed") is get_cache("pubmed")

    def test_custom_ttl_applied(self):
        from biomcp.utils import get_cache, CACHE_TTLS
        cache = get_cache("alphafold")
        assert cache.ttl == CACHE_TTLS["alphafold"]


class TestFormatters:
    def test_format_success_has_status(self):
        import json
        from biomcp.utils import format_success
        out = json.loads(format_success("my_tool", {"key": "value"}))
        assert out["status"] == "success"
        assert out["tool"]   == "my_tool"
        assert out["data"]   == {"key": "value"}

    def test_format_error_has_status(self):
        import json
        from biomcp.utils import format_error
        out = json.loads(format_error("my_tool", ValueError("bad input")))
        assert out["status"]     == "error"
        assert out["tool"]       == "my_tool"
        assert out["error_type"] == "ValueError"
        # Validation errors omit traceback
        assert "traceback" not in out

    def test_format_error_includes_traceback_for_unexpected(self):
        import json
        from biomcp.utils import format_error
        out = json.loads(format_error("my_tool", RuntimeError("oops")))
        assert "traceback" in out


class TestServer:
    def test_tool_count(self):
        from biomcp.server import TOOLS
        assert len(TOOLS) >= 21, f"Expected >=21 tools, got {len(TOOLS)}"

    def test_all_tools_have_name(self):
        from biomcp.server import TOOLS
        for t in TOOLS:
            assert t.name

    def test_all_tools_have_description(self):
        from biomcp.server import TOOLS
        for t in TOOLS:
            assert len(t.description) >= 20, f"Tool '{t.name}' has too-short description"

    def test_all_tools_have_schema(self):
        from biomcp.server import TOOLS
        for t in TOOLS:
            assert "properties" in t.inputSchema, f"'{t.name}' missing properties"
            assert "required"   in t.inputSchema, f"'{t.name}' missing required"

    def test_tool_names_are_unique(self):
        from biomcp.server import TOOLS
        names = [t.name for t in TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_nvidia_tools_registered(self):
        from biomcp.server import TOOLS
        names = {t.name for t in TOOLS}
        assert "predict_structure_boltz2" in names
        assert "generate_dna_evo2"        in names
        assert "score_sequence_evo2"      in names
        assert "design_protein_ligand"    in names

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_returns_error(self):
        import json
        from biomcp.server import _dispatch
        result = json.loads(await _dispatch("nonexistent_tool", {}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_dispatch_validation_error_returns_structured_json(self):
        import json
        from biomcp.server import _dispatch
        # max_results=9999 exceeds limit — should return structured error, not raise
        result = json.loads(await _dispatch("search_pubmed", {"query": "test", "max_results": 9999}))
        assert result["status"] == "error"


# ── Integration tests — require live network ──────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pubmed_search_live():
    from biomcp.tools.ncbi import search_pubmed
    result = await search_pubmed("TP53 tumor suppressor mechanism", max_results=3)
    assert result["total_found"] > 0
    assert len(result["articles"]) == 3
    art = result["articles"][0]
    assert art["pmid"]
    assert art["title"]
    assert "pubmed.ncbi.nlm.nih.gov" in art["url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_info_live():
    from biomcp.tools.ncbi import get_gene_info
    result = await get_gene_info("EGFR")
    assert "error" not in result
    assert result["symbol"].upper() == "EGFR"
    assert result["chromosome"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_protein_info_live():
    from biomcp.tools.proteins import get_protein_info
    result = await get_protein_info("P00533")   # EGFR
    assert "error" not in result
    assert result["accession"]         == "P00533"
    assert result["sequence_length"]    > 0        # FIX: was result["length"]
    assert result["reviewed"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alphafold_live():
    from biomcp.tools.proteins import get_alphafold_structure
    result = await get_alphafold_structure("P04637")
    assert "plddt_summary" in result or "error" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_proteins_live():
    from biomcp.tools.proteins import search_proteins
    result = await search_proteins("kinase", max_results=5)
    assert result["total_results"] > 0
    assert len(result["proteins"]) == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kegg_pathways_live():
    from biomcp.tools.pathways import search_pathways
    result = await search_pathways("apoptosis")
    assert result["total"] > 0
    assert "viewer_url" in result["pathways"][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reactome_pathways_live():
    from biomcp.tools.pathways import get_reactome_pathways
    result = await get_reactome_pathways("EGFR")
    assert "pathways" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_drug_targets_live():
    from biomcp.tools.pathways import get_drug_targets
    result = await get_drug_targets("EGFR", max_results=5)
    assert "drugs" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_disease_associations_live():
    from biomcp.tools.pathways import get_gene_disease_associations
    result = await get_gene_disease_associations("BRCA1", max_results=5)
    assert "associations" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clinical_trials_live():
    from biomcp.tools.advanced import search_clinical_trials
    result = await search_clinical_trials("EGFR lung cancer", max_results=5)
    assert "studies" in result
    assert isinstance(result["studies"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_expression_live():
    from biomcp.tools.advanced import search_gene_expression
    result = await search_gene_expression("BRCA1", "breast cancer", max_datasets=3)
    assert "datasets" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gene_variants_live():
    from biomcp.tools.advanced import get_gene_variants
    result = await get_gene_variants("TP53", max_results=5)
    assert "variants" in result
