"""
Tests - MCP server public surface.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

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
    "bulk_gene_analysis": ("biomcp.tools.innovations", "bulk_gene_analysis"),
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

    def test_all_tools_have_human_titles(self):
        for tool in TOOLS:
            assert tool.title, f"Tool '{tool.name}' is missing title metadata"

    def test_all_tools_have_safety_annotations(self):
        for tool in TOOLS:
            assert tool.annotations is not None, f"Tool '{tool.name}' is missing annotations"
            annotation_payload = (
                tool.annotations.model_dump(exclude_none=True)
                if hasattr(tool.annotations, "model_dump")
                else tool.annotations
            )
            assert "readOnlyHint" in annotation_payload
            assert "destructiveHint" in annotation_payload
            assert "idempotentHint" in annotation_payload

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
            "bulk_gene_analysis",
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
        assert "biomcp://server/status" in uris
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
        assert "/status" in payload["health_endpoints"]
        assert "/readyz" in payload["health_endpoints"]

    def test_status_resource_includes_http_policy_and_session_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("BIOMCP_TRANSPORT", "http")
        monkeypatch.setenv("BIOMCP_CORS_ALLOW_ORIGINS", "https://claude.ai,https://chatgpt.com")
        monkeypatch.setenv("BIOMCP_SESSION_STORE_DIR", "persistent/sessions")

        contents = _read_resource_contents("biomcp://server/status")
        payload = json.loads(contents[0].content)

        assert payload["transport_mode"] == "http"
        assert payload["http_policy"]["cors_allowed_origins"] == [
            "https://claude.ai",
            "https://chatgpt.com",
        ]
        assert payload["session_storage"]["configured_dir"] == "persistent/sessions"
        assert payload["session_storage"]["ephemeral_warning"] is False

    @pytest.mark.asyncio
    async def test_session_resources_are_listed_after_save(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from biomcp.core import knowledge_graph as knowledge_graph_module

        temp_dir = Path(".codex_test_tmp") / f"server-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("BIOMCP_SESSION_STORE_DIR", str(temp_dir))

        try:
            knowledge_graph_module.reset_skg()

            skg = await knowledge_graph_module.get_skg()
            await skg.upsert_node("EGFR", knowledge_graph_module.NodeType.GENE, source="ncbi")

            saved = await server_module._session_workflow(action="save", label="EGFR session")
            resources = _list_resource_definitions()
            uris = {str(resource.uri) for resource in resources}
            assert saved["resource_uri"] in uris

            payload = json.loads(_read_resource_contents(saved["resource_uri"])[0].content)
            assert payload["session_id"] == saved["session_id"]
            assert payload["graph_snapshot"]["summary"]["total_nodes"] == 1

            knowledge_graph_module.reset_skg()
            restored = await server_module._session_workflow(
                action="restore",
                session_id=saved["session_id"],
            )
            assert restored["restored_session_id"] == saved["session_id"]
            assert restored["graph_stats"]["nodes"] == 1
        finally:
            knowledge_graph_module.reset_skg()
            shutil.rmtree(temp_dir, ignore_errors=True)


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


class TestStartupCacheWarming:
    @pytest.mark.asyncio
    async def test_warm_common_gene_caches_collects_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[tuple[str, str]] = []

        async def _ok(gene: str):
            calls.append(("ok", gene))
            return {"gene": gene}

        async def _fail(gene: str):
            calls.append(("fail", gene))
            raise RuntimeError(f"boom:{gene}")

        monkeypatch.setattr(
            server_module,
            "_build_cache_warmers",
            lambda: [("ok", _ok), ("fail", _fail)],
        )
        monkeypatch.setenv("BIOMCP_CACHE_WARM_CONCURRENCY", "2")

        result = await server_module._warm_common_gene_caches(["TP53", "EGFR"])

        assert result["warming_summary"]["total_calls"] == 4
        assert result["warming_summary"]["successful_calls"] == 2
        assert result["warming_summary"]["failed_calls"] == 2
        assert len(result["failed"]) == 2
        assert {entry["gene"] for entry in result["failed"]} == {"TP53", "EGFR"}
        assert ("ok", "TP53") in calls
        assert ("fail", "EGFR") in calls

    @pytest.mark.asyncio
    async def test_start_cache_warmer_respects_env_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("BIOMCP_CACHE_WARMING", "0")
        task = server_module._start_cache_warmer("http")
        assert task is None

    @pytest.mark.asyncio
    async def test_start_cache_warmer_schedules_background_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        seen: list[list[str]] = []

        async def _fake_warm(genes: list[str]):
            seen.append(list(genes))
            return {"warming_summary": {"total_calls": len(genes)}}

        monkeypatch.setenv("BIOMCP_CACHE_WARMING", "1")
        monkeypatch.setattr(server_module, "_cache_warm_gene_panel", lambda: ["TP53", "EGFR"])
        monkeypatch.setattr(server_module, "_warm_common_gene_caches", _fake_warm)

        task = server_module._start_cache_warmer("http")
        assert task is not None
        await task
        assert seen == [["TP53", "EGFR"]]


class TestStartupWarnings:
    def test_warn_ephemeral_session_store_only_for_http_without_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        warnings: list[str] = []

        monkeypatch.delenv("BIOMCP_SESSION_STORE_DIR", raising=False)
        monkeypatch.setattr(server_module.logger, "warning", lambda message: warnings.append(message))

        assert server_module._warn_ephemeral_session_store("http") is True
        assert len(warnings) == 1
        assert "BIOMCP_SESSION_STORE_DIR" in warnings[0]

    def test_warn_ephemeral_session_store_skips_stdio_or_explicit_directory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        warnings: list[str] = []

        monkeypatch.setattr(server_module.logger, "warning", lambda message: warnings.append(message))

        monkeypatch.setenv("BIOMCP_SESSION_STORE_DIR", ".biomcp_sessions")
        assert server_module._warn_ephemeral_session_store("http") is False

        monkeypatch.delenv("BIOMCP_SESSION_STORE_DIR", raising=False)
        assert server_module._warn_ephemeral_session_store("stdio") is False
        assert warnings == []


class TestStreamingProgress:
    @pytest.mark.asyncio
    async def test_multi_omics_dispatch_streams_progress_notifications(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        session = SimpleNamespace(
            send_progress_notification=AsyncMock(),
            send_log_message=AsyncMock(),
        )
        ctx = SimpleNamespace(
            meta=SimpleNamespace(progressToken="tok-1"),
            session=session,
            request_id="req-1",
        )
        advanced_module = SimpleNamespace()

        async def _fake_impl(
            gene_symbol: str,
            detail_level: str = "compact",
            include_synthesis: bool = True,
            progress_callback=None,
        ):
            for layer_name, payload in [
                ("genomics", {"symbol": gene_symbol}),
                ("literature", {"total_publications": 2, "recent_papers": []}),
                ("reactome", {"pathways": [{"id": "R-HSA-1"}]}),
                ("drug_targets", {"drugs": [{"name": "DrugX"}]}),
                ("disease_associations", {"associations": [{"disease_name": "Lung cancer"}]}),
                ("expression", {"datasets": [{"gse": "GSE1"}]}),
                ("clinical_trials", {"studies": [{"nct_id": "NCT1"}]}),
            ]:
                await progress_callback(layer_name, payload)
            return {"gene": gene_symbol, "detail_level": detail_level, "layers": {}}

        advanced_module._multi_omics_gene_report_impl = _fake_impl
        advanced_module.multi_omics_gene_report = AsyncMock(
            return_value={"gene": "EGFR", "detail_level": "compact", "layers": {}}
        )

        monkeypatch.setattr(server_module, "_SERVER_INSTANCE", SimpleNamespace(request_context=ctx))
        monkeypatch.setattr(server_module, "_get_tool_modules", lambda: {"advanced": advanced_module})
        monkeypatch.setattr(server_module, "get_cache", lambda namespace: {})

        result = await server_module._dispatch_multi_omics_gene_report("EGFR", detail_level="compact")

        assert result["gene"] == "EGFR"
        assert result["detail_level"] == "compact"
        assert session.send_progress_notification.await_count == 7
        first_call = session.send_progress_notification.await_args_list[0]
        assert first_call.args[0] == "tok-1"
        assert first_call.args[1] == 1.0
        assert first_call.kwargs["total"] == 7.0
        assert "genomics ready" in first_call.kwargs["message"]
        assert session.send_log_message.await_count == 8

    @pytest.mark.asyncio
    async def test_plan_and_execute_research_streams_progress_notifications(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        session = SimpleNamespace(
            send_progress_notification=AsyncMock(),
            send_log_message=AsyncMock(),
        )
        ctx = SimpleNamespace(
            meta=SimpleNamespace(progressToken="tok-2"),
            session=session,
            request_id="req-2",
        )

        class FakePlanner:
            def __init__(self, dispatcher):
                self.dispatcher = dispatcher

            def build_plan(self, goal: str, depth: str = "standard", entities=None):
                return SimpleNamespace(goal=goal, strategy="demo", nodes=[object(), object()])

            async def execute(self, plan, timeout_per_tool: float = 60.0, progress_callback=None):
                assert progress_callback is not None
                await progress_callback(
                    "level_started",
                    {"level": 1, "total_levels": 1, "tools": ["get_gene_info", "search_pubmed"]},
                )
                await progress_callback(
                    "node_finished",
                    {
                        "node": {"tool": "get_gene_info", "status": "complete"},
                        "result": {"symbol": "EGFR"},
                    },
                )
                await progress_callback(
                    "node_finished",
                    {
                        "node": {"tool": "search_pubmed", "status": "failed"},
                        "result": {"error": "timed out"},
                    },
                )
                await progress_callback(
                    "plan_completed",
                    {"completed": 1, "failed": 1, "total_steps": 2, "elapsed_s": 3.2},
                )
                return {"goal": plan.goal, "execution_summary": {"completed": 1, "failed": 1, "total_steps": 2}}

        monkeypatch.setattr(server_module, "_SERVER_INSTANCE", SimpleNamespace(request_context=ctx))

        with patch("biomcp.core.query_planner.AdaptiveQueryPlanner", FakePlanner):
            result = await server_module._plan_and_execute_research("Understand EGFR")

        assert result["goal"] == "Understand EGFR"
        assert session.send_progress_notification.await_count == 3
        progress_messages = [
            call.kwargs["message"] for call in session.send_progress_notification.await_args_list
        ]
        assert any("get_gene_info complete" in message for message in progress_messages)
        assert any("search_pubmed failed" in message for message in progress_messages)
        assert progress_messages[-1].endswith("plan complete: 1/2 steps successful")
        assert session.send_log_message.await_count == 5


class TestServerBranding:
    def test_logo_asset_is_resolvable(self):
        logo_path = _resolve_logo_path()
        assert logo_path is not None
        assert logo_path.endswith(("LOGO.png", "LOGO.jpeg"))

    def test_initialization_options_include_website_and_icon(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BIOMCP_WEBSITE_URL", "https://example.com/biomcp")
        monkeypatch.setenv("BIOMCP_ICON_URL", "https://example.com/biomcp/logo.png")

        options = create_server().create_initialization_options()

        assert options.server_name == "Heuris-BioMCP"
        assert options.website_url == "https://example.com/biomcp"
        assert options.icons is not None
        assert len(options.icons) == 1
        assert options.icons[0].src == "https://example.com/biomcp/logo.png"
        assert options.icons[0].mimeType == "image/png"


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
        assert report["status_url"] == "https://example.com/biomcp/status"

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

    def test_cors_allowed_origins_is_empty_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("BIOMCP_CORS_ALLOW_ORIGINS", raising=False)

        assert server_module._cors_allowed_origins() == []

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_after_window_capacity(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BIOMCP_HTTP_RATE_LIMIT_ENABLED", "1")
        monkeypatch.setenv("BIOMCP_HTTP_RATE_LIMIT_REQUESTS", "2")
        monkeypatch.setenv("BIOMCP_HTTP_RATE_LIMIT_WINDOW_SECONDS", "60")
        server_module._HTTP_RATE_LIMIT_STATE.clear()

        assert await server_module._check_rate_limit("127.0.0.1", now=10.0) == (True, 0)
        assert await server_module._check_rate_limit("127.0.0.1", now=11.0) == (True, 0)
        allowed, retry_after = await server_module._check_rate_limit("127.0.0.1", now=12.0)

        assert allowed is False
        assert retry_after > 0
