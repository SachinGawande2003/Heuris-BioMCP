"""
BioMCP Core Infrastructure
============================
Novel architectural components that elevate BioMCP beyond an API wrapper
collection into a stateful, intelligent biological research system.

Components:
  knowledge_graph  — Session Knowledge Graph (SKG): live entity graph across
                     all tool calls, cross-database connection discovery,
                     contradiction detection, provenance export
  entity_resolver  — Canonical biological entity resolution across HGNC,
                     UniProt, Ensembl, NCBI — eliminates duplicate lookups
  query_planner    — DAG-based adaptive research planner with parallel
                     execution, dependency resolution, and insight synthesis
"""

from biomcp.core.entity_resolver import (
    BioEntity,
    EntityRegistry,
    EntityResolver,
    get_registry,
    get_resolver,
)
from biomcp.core.knowledge_graph import (
    EdgeType,
    NodeType,
    SessionKnowledgeGraph,
    SKGEdge,
    SKGNode,
    auto_index,
    get_skg,
    index_clinical_trials_result,
    index_disease_associations_result,
    index_drug_targets_result,
    index_gene_result,
    index_pathways_result,
    index_protein_result,
    # Extractors
    index_pubmed_result,
    index_variants_result,
    reset_skg,
)
from biomcp.core.query_planner import (
    AdaptiveQueryPlanner,
    NodeStatus,
    PlanNode,
    ResearchPlan,
)

__all__ = [
    # Knowledge Graph
    "SessionKnowledgeGraph", "SKGNode", "SKGEdge",
    "NodeType", "EdgeType",
    "get_skg", "reset_skg", "auto_index",
    "index_pubmed_result", "index_gene_result", "index_protein_result",
    "index_drug_targets_result", "index_disease_associations_result",
    "index_pathways_result", "index_clinical_trials_result",
    "index_variants_result",
    # Entity Resolver
    "BioEntity", "EntityRegistry", "EntityResolver",
    "get_resolver", "get_registry",
    # Query Planner
    "AdaptiveQueryPlanner", "ResearchPlan", "PlanNode", "NodeStatus",
]
