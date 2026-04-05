"""
Tests - MCP server public surface.
"""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

import biomcp.server as server_module
from biomcp import __version__
from biomcp.server import (
    _PUBLIC_TOOL_EXAMPLES,
    TOOLS,
    _build_health_report,
    _build_readiness_report,
    _build_root_report,
    _build_tool_health_report,
    _dispatch,
    _list_resource_definitions,
    _read_resource_contents,
    _resolve_logo_path,
    create_server,
)

TOOL_IMPLEMENTATIONS = {
    "search_pubmed": ("biomcp.tools.ncbi", "search_pubmed"),
    "get_gene_info": ("biomcp.tools.ncbi", "get_gene_info"),
    "run_blast": ("biomcp.tools.ncbi", "run_blast"),
    "get_protein_info": ("biomcp.tools.proteins", "get_protein_info"),
    "find_protein": ("biomcp.tools.strategy_surface", "find_protein"),
    "get_alphafold_structure": ("biomcp.tools.proteins", "get_alphafold_structure"),
    "pathway_analysis": ("biomcp.tools.strategy_surface", "pathway_analysis"),
    "get_drug_targets": ("biomcp.tools.pathways", "get_drug_targets"),
    "get_gene_disease_associations": ("biomcp.tools.pathways", "get_gene_disease_associations"),
    "search_clinical_trials": ("biomcp.tools.advanced", "search_clinical_trials"),
    "multi_omics_gene_report": ("biomcp.tools.advanced", "multi_omics_gene_report"),
    "predict_structure_boltz2": ("biomcp.tools.strategy_surface", "boltz2_workflow"),
    "generate_dna_evo2": ("biomcp.tools.strategy_surface", "evo2_workflow"),
    "crispr_analysis": ("biomcp.tools.strategy_surface", "crispr_analysis"),
    "drug_safety": ("biomcp.tools.strategy_surface", "drug_safety"),
    "variant_analysis": ("biomcp.tools.strategy_surface", "variant_analysis"),
    "find_repurposing_candidates": ("biomcp.tools.intelligence", "find_repurposing_candidates"),
    "verify_biological_claim": ("biomcp.tools.verify", "verify_biological_claim"),
    "search_cbio_mutations": ("biomcp.tools.databases", "search_cbio_mutations"),
    "search_gwas_catalog": ("biomcp.tools.databases", "search_gwas_catalog"),
    "session": ("biomcp.server", "_session_workflow"),
    "drug_interaction_checker": ("biomcp.tools.strategy_surface", "drug_interaction_checker"),
    "protein_binding_pocket": ("biomcp.tools.strategy_surface", "protein_binding_pocket"),
    "biomarker_panel_design": ("biomcp.tools.strategy_surface", "biomarker_panel_design"),
    "pharmacogenomics_report": ("biomcp.tools.strategy_surface", "pharmacogenomics_report"),
    "protein_family_analysis": ("biomcp.tools.strategy_surface", "protein_family_analysis"),
    "network_enrichment": ("biomcp.tools.strategy_surface", "network_enrichment"),
    "rnaseq_deconvolution": ("biomcp.tools.strategy_surface", "rnaseq_deconvolution"),
    "structural_similarity": ("biomcp.tools.strategy_surface", "structural_similarity"),
    "rare_disease_diagnosis": ("biomcp.tools.strategy_surface", "rare_disease_diagnosis"),
    "genome_browser_snapshot": ("biomcp.tools.strategy_surface", "genome_browser_snapshot"),
}


def _assert_value_matches_schema(prop_def: dict, value: object) -> None:
    prop_type = prop_def.get("type")
    if prop_type == "string":
        assert isinstance(value, str)
    elif prop_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool)
    elif prop_type == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
    elif prop_type == "boolean":
        assert isinstance(value, bool)
    elif prop_type == "array":
        assert isinstance(value, list)
    elif prop_type == "object":
        assert isinstance(value, dict)
    else:
        raise AssertionError(f"Unsupported schema type: {prop_type}")

    enum = prop_def.get("enum")
    if enum is not None:
        assert value in enum


class TestToolRegistry:
    def test_minimum_tool_count(self):
        assert len(TOOLS) >= 20, f"Expected >=20 tools, got {len(TOOLS)}"

    def test_curated_tool_count_target(self):
        assert 25 <= len(TOOLS) <= 40, f"Expected curated surface in 25-40 range, got {len(TOOLS)}"

    def test_all_tools_have_name(self):
        for tool in TOOLS:
            assert tool.name, f"Tool missing name: {tool}"

    def test_all_tools_have_description(self):
        for tool in TOOLS:
            assert tool.description, f"Tool '{tool.name}' has empty description"

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert tool.inputSchema, f"Tool '{tool.name}' missing inputSchema"

    def test_schemas_have_required_keys(self):
        for tool in TOOLS:
            schema = tool.inputSchema
            assert "properties" in schema, f"'{tool.name}' schema missing 'properties'"
            assert "required" in schema, f"'{tool.name}' schema missing 'required'"

    def test_no_duplicate_tool_names(self):
        names = [t.name for t in TOOLS]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert len(names) == len(set(names)), f"Duplicate tool names: {duplicates}"

    def test_strategy_tools_present(self):
        names = {t.name for t in TOOLS}
        expected = {
            "search_pubmed",
            "get_gene_info",
            "run_blast",
            "get_protein_info",
            "find_protein",
            "get_alphafold_structure",
            "pathway_analysis",
            "get_drug_targets",
            "get_gene_disease_associations",
            "search_clinical_trials",
            "multi_omics_gene_report",
            "predict_structure_boltz2",
            "generate_dna_evo2",
            "crispr_analysis",
            "drug_safety",
            "variant_analysis",
            "find_repurposing_candidates",
            "verify_biological_claim",
            "search_cbio_mutations",
            "search_gwas_catalog",
            "session",
            "drug_interaction_checker",
            "protein_binding_pocket",
            "biomarker_panel_design",
            "pharmacogenomics_report",
            "protein_family_analysis",
            "network_enrichment",
            "rnaseq_deconvolution",
            "structural_similarity",
            "rare_disease_diagnosis",
            "genome_browser_snapshot",
        }
        missing = expected - names
        assert not missing, f"Missing tools: {missing}"

    def test_removed_tools_not_public(self):
        names = {t.name for t in TOOLS}
        removed = {
            "search_proteins",
            "search_pdb_structures",
            "search_pathways",
            "get_pathway_genes",
            "get_reactome_pathways",
            "design_crispr_guides",
            "score_guide_efficiency",
            "predict_off_target_sites",
            "design_base_editor_guides",
            "get_crispr_repair_outcomes",
            "query_adverse_events",
            "analyze_safety_signals",
            "get_drug_label_warnings",
            "compare_drug_safety",
            "classify_variant",
            "get_population_frequency",
            "lookup_clinvar_variant",
            "query_neuroimaging_datasets",
            "generate_research_hypothesis",
            "estimate_statistical_power",
            "suggest_cell_lines",
            "detect_database_conflicts",
            "find_research_gaps",
            "validate_reasoning_chain",
            "generate_experimental_protocol",
            "bulk_gene_analysis",
            "compute_pathway_enrichment",
            "analyze_coexpression",
            "predict_splice_impact",
            "get_cancer_hotspots",
            "get_biogrid_interactions",
            "search_orphan_diseases",
            "search_cellmarker",
            "search_metabolomics",
            "get_encode_regulatory",
            "get_disgenet_associations",
            "get_pharmgkb_variants",
            "get_ucsc_splice_variants",
        }
        assert not (removed & names), f"Unexpected public legacy tools: {removed & names}"

    def test_schema_types_are_valid(self):
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        for tool in TOOLS:
            for prop_name, prop_def in tool.inputSchema.get("properties", {}).items():
                ptype = prop_def.get("type", "")
                assert ptype in valid_types, (
                    f"Tool '{tool.name}', property '{prop_name}' has invalid type '{ptype}'"
                )

    def test_all_public_tools_have_examples(self):
        for tool in TOOLS:
            examples = tool.inputSchema.get("examples")
            assert examples, f"Tool '{tool.name}' is missing schema examples"

    def test_examples_cover_required_fields_and_match_types(self):
        tool_names = {tool.name for tool in TOOLS}
        assert set(_PUBLIC_TOOL_EXAMPLES) == tool_names

        for tool in TOOLS:
            schema = tool.inputSchema
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            examples = schema.get("examples", [])

            for example in examples:
                assert required.issubset(example), (
                    f"Example for '{tool.name}' does not cover required fields"
                )
                for field_name, value in example.items():
                    assert field_name in properties, (
                        f"Example for '{tool.name}' includes unknown field '{field_name}'"
                    )
                    _assert_value_matches_schema(properties[field_name], value)


class TestMCPResources:
    def test_resource_catalog_is_exposed(self):
        resources = _list_resource_definitions()
        uris = {str(resource.uri) for resource in resources}
        assert "biomcp://server/capabilities" in uris
        assert "biomcp://tools/catalog" in uris

    def test_tool_catalog_resource_contains_examples(self):
        contents = _read_resource_contents("biomcp://tools/catalog")
        assert len(contents) == 1
        assert contents[0].mime_type == "application/json"

        payload = json.loads(contents[0].content)
        assert payload["tool_count"] == len(TOOLS)
        first_tool = payload["tools"][0]
        assert "examples" in first_tool

    def test_capabilities_resource_includes_transport_and_health_endpoints(self):
        contents = _read_resource_contents("biomcp://server/capabilities")
        payload = json.loads(contents[0].content)
        assert payload["service"] == "heuris-biomcp"
        assert payload["transport_modes"] == ["stdio", "http"]
        assert payload["transport_endpoints"]["streamable_http"] == "/mcp"
        assert payload["transport_endpoints"]["sse"] == "/sse"
        assert "/readyz" in payload["health_endpoints"]


class TestDispatchSmoke:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name", sorted(TOOL_IMPLEMENTATIONS))
    async def test_public_tools_dispatch_with_example_payloads(
        self,
        tool_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        module_name, attr_name = TOOL_IMPLEMENTATIONS[tool_name]
        module = importlib.import_module(module_name)

        async def _fake_handler(**kwargs):
            return {"tool": tool_name, "arguments": kwargs}

        monkeypatch.setattr(module, attr_name, _fake_handler)

        payload = _PUBLIC_TOOL_EXAMPLES[tool_name][0]
        result = json.loads(await _dispatch(tool_name, payload))
        assert result["status"] == "success"
        assert result["tool"] == tool_name
        assert result["data"]["arguments"] == payload

    @pytest.mark.asyncio
    async def test_dispatch_wraps_http_errors_as_structured_error(self, monkeypatch: pytest.MonkeyPatch):
        request = httpx.Request("GET", "https://example.org/reactome")

        async def _failing_dispatch(name: str, args: dict[str, object]) -> str:
            raise httpx.ConnectTimeout("timed out", request=request)

        monkeypatch.setattr(server_module, "_raw_dispatch", _failing_dispatch)

        result = json.loads(await _dispatch("get_reactome_pathways", {"gene_symbol": "EGFR"}))

        assert result["status"] == "error"
        assert result["error_type"] == "ConnectTimeout"
        assert result["url"] == "https://example.org/reactome"
        assert "traceback" not in result


class TestDispatchInitialization:
    def test_dispatch_table_is_built_once(self, monkeypatch: pytest.MonkeyPatch):
        build_count = 0
        stub_table = {"search_pubmed": object()}

        def _fake_build_dispatch_table():
            nonlocal build_count
            build_count += 1
            return stub_table

        monkeypatch.setattr(server_module, "_DISPATCH_TABLE", None)
        monkeypatch.setattr(server_module, "_build_dispatch_table", _fake_build_dispatch_table)

        first = server_module._get_dispatch_table()
        second = server_module._get_dispatch_table()

        assert build_count == 1
        assert first is stub_table
        assert second is stub_table

    def test_create_server_warms_dispatch_table(self, monkeypatch: pytest.MonkeyPatch):
        build_count = 0
        stub_table = {"search_pubmed": object()}

        def _fake_build_dispatch_table():
            nonlocal build_count
            build_count += 1
            return stub_table

        monkeypatch.setattr(server_module, "_DISPATCH_TABLE", None)
        monkeypatch.setattr(server_module, "_build_dispatch_table", _fake_build_dispatch_table)

        create_server()

        assert build_count == 1
        assert server_module._DISPATCH_TABLE is stub_table


class TestServerBranding:
    def test_logo_asset_is_resolvable(self):
        logo_path = _resolve_logo_path()
        assert logo_path is not None
        assert logo_path.endswith("LOGO.jpeg")

    def test_initialization_options_include_website_and_icon(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BIOMCP_WEBSITE_URL", "https://example.com/biomcp")
        monkeypatch.setenv("BIOMCP_ICON_URL", "https://example.com/biomcp/logo.jpeg")

        options = create_server().create_initialization_options()

        assert options.website_url == "https://example.com/biomcp"
        assert options.icons is not None
        assert len(options.icons) == 1
        assert options.icons[0].src == "https://example.com/biomcp/logo.jpeg"
        assert options.icons[0].mimeType == "image/jpeg"


class TestOperationalHealth:
    def test_health_report_matches_server_version(self):
        report = _build_health_report("http")
        assert report["service"] == "heuris-biomcp"
        assert report["version"] == __version__
        assert report["tool_count"] == len(TOOLS)
        assert report["transport"]["streamable_http_path"] == "/mcp"
        assert report["transport"]["sse_path"] == "/sse"
        assert report["transport"]["message_path"] == "/messages/"

    def test_readiness_report_is_ready_with_registered_tools(self):
        report = _build_readiness_report("stdio")
        assert report["ready"] is True
        assert report["tool_count"] == len(TOOLS)

    def test_root_report_recommends_streamable_http_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BIOMCP_WEBSITE_URL", "https://example.com/biomcp")

        report = _build_root_report("http")

        assert report["recommended_remote_url"] == "https://example.com/biomcp/mcp"
        assert report["legacy_sse_url"] == "https://example.com/biomcp/sse"
        assert report["health_url"] == "https://example.com/biomcp/healthz"

    def test_tool_health_reflects_missing_optional_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NVIDIA_BOLTZ2_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_EVO2_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)

        report = _build_tool_health_report()
        gated = report["gated_capabilities"]

        assert "nvidia_boltz2" in gated
        assert "nvidia_evo2" in gated
        assert gated["nvidia_boltz2"]["status"] == "degraded"
        assert gated["nvidia_evo2"]["status"] == "degraded"
        assert "predict_structure_boltz2" in gated["nvidia_boltz2"]["tools"]
        assert "generate_dna_evo2" in gated["nvidia_evo2"]["tools"]
