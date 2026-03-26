"""
Tests — MCP Server Core
=========================
Validates tool registry, schema correctness, and dispatch routing.
"""

from __future__ import annotations

import pytest

from biomcp.server import TOOLS


class TestToolRegistry:
    def test_minimum_tool_count(self):
        assert len(TOOLS) >= 20, f"Expected ≥20 tools, got {len(TOOLS)}"

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
        assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_known_tools_present(self):
        names = {t.name for t in TOOLS}
        expected = {
            "search_pubmed", "get_gene_info", "run_blast",
            "get_protein_info", "search_proteins", "get_alphafold_structure",
            "search_pdb_structures", "search_pathways", "get_pathway_genes",
            "get_reactome_pathways", "get_drug_targets", "get_compound_info",
            "get_gene_disease_associations", "get_gene_variants",
            "search_gene_expression", "search_scrna_datasets",
            "search_clinical_trials", "get_trial_details",
            "multi_omics_gene_report", "query_neuroimaging_datasets",
        }
        missing = expected - names
        assert not missing, f"Missing tools: {missing}"

    def test_schema_types_are_valid(self):
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        for tool in TOOLS:
            for prop_name, prop_def in tool.inputSchema.get("properties", {}).items():
                ptype = prop_def.get("type", "")
                assert ptype in valid_types, (
                    f"Tool '{tool.name}', property '{prop_name}' has invalid type '{ptype}'"
                )
