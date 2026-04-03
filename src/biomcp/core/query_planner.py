"""
BioMCP — Adaptive Query Planner
=================================
Transforms a natural-language research goal into an optimized, dependency-
aware execution plan — then executes it, handling failures gracefully.

Architecture:
  Research Goal (natural language)
        ↓
  Intent Classifier → Research Plan (DAG of tool calls)
        ↓
  Dependency Resolver → Parallelization Groups
        ↓
  Async Executor → Collects results → Feeds SKG
        ↓
  Synthesizer → Integrated Research Report

DAG Example for "Understand KRAS G12C as a drug target in NSCLC":
  ┌──────────────────────────────────────────────────────┐
  │  Level 0 (parallel):                                  │
  │    get_gene_info("KRAS")                              │
  │    search_pubmed("KRAS G12C NSCLC review")           │
  │    get_gene_variants("KRAS")                          │
  │                                                       │
  │  Level 1 (parallel, after Level 0):                   │
  │    get_protein_info("P01116")  ← from Level 0        │
  │    get_drug_targets("KRAS")                           │
  │    get_reactome_pathways("KRAS")                      │
  │                                                       │
  │  Level 2 (parallel, after Level 1):                   │
  │    search_clinical_trials("KRAS G12C NSCLC")         │
  │    get_gene_disease_associations("KRAS")              │
  │    multi_omics_gene_report("KRAS")  ← optional depth  │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Plan Node — a single tool call in the DAG
# ─────────────────────────────────────────────────────────────────────────────

class NodeStatus(StrEnum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class PlanNode:
    """A single tool call within a research plan DAG."""
    node_id:      str
    tool_name:    str
    tool_args:    dict[str, Any]
    depends_on:   list[str]  = field(default_factory=list)  # node_ids
    status:       NodeStatus = NodeStatus.PENDING
    result:       Any        = None
    error:        str        = ""
    started_at:   float      = 0.0
    completed_at: float      = 0.0
    rationale:    str        = ""   # why this tool was chosen

    @property
    def elapsed_s(self) -> float:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 2)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":   self.node_id,
            "tool":      self.tool_name,
            "args":      self.tool_args,
            "status":    self.status.value,
            "elapsed_s": self.elapsed_s,
            "error":     self.error,
            "rationale": self.rationale,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Research Plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchPlan:
    """
    A directed acyclic graph of tool calls for a research goal.
    Nodes at the same dependency level execute in parallel.
    """
    plan_id:   str
    goal:      str
    nodes:     list[PlanNode]
    strategy:  str   = ""
    depth:     str   = "standard"   # "quick" | "standard" | "deep"
    created_at: float = field(default_factory=time.monotonic)

    def execution_levels(self) -> list[list[PlanNode]]:
        """
        Topological sort → returns groups that can execute in parallel.
        Each group has no dependencies on each other.
        """
        completed: set[str] = set()
        remaining  = list(self.nodes)
        levels: list[list[PlanNode]] = []

        while remaining:
            # Find all nodes whose dependencies are already completed
            ready = [
                n for n in remaining
                if all(dep in completed for dep in n.depends_on)
                and n.status not in (NodeStatus.COMPLETE, NodeStatus.FAILED, NodeStatus.SKIPPED)
            ]
            if not ready:
                # Avoid infinite loop — mark rest as skipped (circular dep guard)
                for n in remaining:
                    n.status = NodeStatus.SKIPPED
                break
            levels.append(ready)
            for n in ready:
                completed.add(n.node_id)
                remaining.remove(n)

        return levels

    def summary(self) -> dict[str, Any]:
        total    = len(self.nodes)
        complete = sum(1 for n in self.nodes if n.status == NodeStatus.COMPLETE)
        failed   = sum(1 for n in self.nodes if n.status == NodeStatus.FAILED)
        return {
            "plan_id":       self.plan_id,
            "goal":          self.goal,
            "depth":         self.depth,
            "strategy":      self.strategy,
            "total_steps":   total,
            "completed":     complete,
            "failed":        failed,
            "success_rate":  f"{(complete / total * 100):.0f}%" if total else "0%",
            "steps":         [n.to_dict() for n in self.nodes],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Intent Templates — maps research goals to tool sequences
# ─────────────────────────────────────────────────────────────────────────────

def _gene_drug_target_plan(gene: str, depth: str) -> list[dict[str, Any]]:
    """Plan: understand a gene as a drug target."""
    n0 = str(uuid4().hex[:6])
    n1 = str(uuid4().hex[:6])
    n2 = str(uuid4().hex[:6])
    n3 = str(uuid4().hex[:6])
    n4 = str(uuid4().hex[:6])
    n5 = str(uuid4().hex[:6])
    n6 = str(uuid4().hex[:6])
    n7 = str(uuid4().hex[:6])

    base = [
        {"id": n0, "tool": "get_gene_info",              "args": {"gene_symbol": gene}, "deps": [], "rationale": "Establish genomic context for the target"},
        {"id": n1, "tool": "search_pubmed",              "args": {"query": f"{gene} drug target inhibitor", "max_results": 8}, "deps": [], "rationale": "Survey current drug discovery literature"},
        {"id": n2, "tool": "get_drug_targets",           "args": {"gene_symbol": gene, "max_results": 20}, "deps": [n0], "rationale": "Identify known compounds and IC50 values"},
        {"id": n3, "tool": "get_gene_disease_associations", "args": {"gene_symbol": gene, "max_results": 10}, "deps": [n0], "rationale": "Understand disease relevance"},
        {"id": n4, "tool": "get_reactome_pathways",      "args": {"gene_symbol": gene}, "deps": [n0], "rationale": "Identify pathway context for target"},
        {"id": n5, "tool": "search_clinical_trials",     "args": {"query": gene, "max_results": 10}, "deps": [n2, n3], "rationale": "Find active trials for target-directed therapies"},
    ]

    if depth in ("standard", "deep"):
        base += [
            {"id": n6, "tool": "get_gene_variants",     "args": {"gene_symbol": gene, "max_results": 15}, "deps": [n0], "rationale": "Identify clinically relevant mutations"},
            {"id": n7, "tool": "search_gene_expression", "args": {"gene_symbol": gene, "max_results": 5}, "deps": [n0], "rationale": "Tissue-specific expression context"},
        ]

    if depth == "deep":
        n8 = str(uuid4().hex[:6])
        base.append({"id": n8, "tool": "multi_omics_gene_report", "args": {"gene_symbol": gene}, "deps": [n0, n1], "rationale": "Comprehensive multi-omics integration"})

    return base


def _disease_genomics_plan(disease: str, gene: str) -> list[dict[str, Any]]:
    """Plan: genomics-first disease investigation."""
    n0 = str(uuid4().hex[:6])
    n1 = str(uuid4().hex[:6])
    n2 = str(uuid4().hex[:6])
    n3 = str(uuid4().hex[:6])
    n4 = str(uuid4().hex[:6])
    return [
        {"id": n0, "tool": "search_pubmed",       "args": {"query": f"{disease} {gene} genomics mechanism", "max_results": 8}, "deps": [], "rationale": "Literature overview"},
        {"id": n1, "tool": "get_gene_info",        "args": {"gene_symbol": gene}, "deps": [], "rationale": "Gene context"},
        {"id": n2, "tool": "get_gene_variants",    "args": {"gene_symbol": gene, "consequence_type": "all", "max_results": 25}, "deps": [n1], "rationale": "Mutation landscape"},
        {"id": n3, "tool": "get_gene_disease_associations", "args": {"gene_symbol": gene, "max_results": 10}, "deps": [n1], "rationale": "Evidence-based disease links"},
        {"id": n4, "tool": "search_clinical_trials", "args": {"query": f"{disease} {gene}", "max_results": 8}, "deps": [n2, n3], "rationale": "Clinical translation status"},
    ]


def _protein_structure_plan(
    uniprot: str, ligand: str | None = None
) -> list[dict[str, Any]]:
    """Plan: protein structure and binding analysis."""
    n0 = str(uuid4().hex[:6])
    n1 = str(uuid4().hex[:6])
    n2 = str(uuid4().hex[:6])

    base = [
        {"id": n0, "tool": "get_protein_info",         "args": {"accession": uniprot}, "deps": [], "rationale": "Full protein annotation"},
        {"id": n1, "tool": "get_alphafold_structure",  "args": {"uniprot_accession": uniprot}, "deps": [n0], "rationale": "Predicted structure confidence"},
        {"id": n2, "tool": "search_pdb_structures",    "args": {"query": uniprot, "max_results": 5}, "deps": [n0], "rationale": "Experimental structures"},
    ]
    if ligand:
        n3 = str(uuid4().hex[:6])
        base.append({"id": n3, "tool": "predict_structure_boltz2", "args": {"protein_sequences": ["__from__:get_protein_info:sequence"], "ligand_smiles": [ligand], "predict_affinity": True}, "deps": [n0], "rationale": "AI structure + binding affinity"})
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Query Planner
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveQueryPlanner:
    """
    Builds and executes optimized research plans.

    Usage::
        planner = AdaptiveQueryPlanner(dispatcher_fn)
        result  = await planner.plan_and_execute(
            goal="Understand KRAS G12C as a drug target in NSCLC",
            depth="standard",
        )
    """

    def __init__(self, dispatcher: Callable[[str, dict[str, Any]], Any]) -> None:
        self._dispatch = dispatcher

    def _classify_intent(self, goal: str) -> tuple[str, dict[str, str]]:
        """
        Classify research goal and extract entities.
        Returns: (intent_type, entities_dict)
        """
        goal_lower = goal.lower()
        entities: dict[str, str] = {}

        # Extract gene symbols (uppercase 2–10 char tokens)
        import re
        gene_matches = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', goal)
        if gene_matches:
            entities["gene"] = gene_matches[0]

        # Extract UniProt accessions
        uniprot_matches = re.findall(r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b', goal)
        if uniprot_matches:
            entities["uniprot"] = uniprot_matches[0]

        # Classify intent
        if any(w in goal_lower for w in ("drug target", "inhibit", "therapeut", "ic50", "ligand", "compound")):
            return "drug_target", entities
        if any(w in goal_lower for w in ("disease", "cancer", "mutation", "variant", "genomic")):
            return "disease_genomics", entities
        if any(w in goal_lower for w in ("structure", "fold", "binding", "pocket", "docking")):
            return "protein_structure", entities
        if any(w in goal_lower for w in ("pathway", "signaling", "network", "interact")):
            return "pathway_analysis", entities
        if any(w in goal_lower for w in ("expression", "tissue", "rna", "transcriptom")):
            return "expression_analysis", entities

        return "general_gene", entities

    def build_plan(
        self,
        goal: str,
        depth: str = "standard",
        entities: dict[str, str] | None = None,
    ) -> ResearchPlan:
        """
        Build a dependency-aware execution DAG for the research goal.
        """
        intent, auto_entities = self._classify_intent(goal)
        ents = {**auto_entities, **(entities or {})}
        gene    = ents.get("gene", "")
        uniprot = ents.get("uniprot", "")
        disease = ents.get("disease", "")
        ligand  = ents.get("ligand", "")

        # Build node specs based on intent
        if intent == "drug_target" and gene:
            node_specs = _gene_drug_target_plan(gene, depth)
            strategy   = f"Drug target analysis pipeline for {gene}"
        elif intent == "disease_genomics" and gene:
            node_specs = _disease_genomics_plan(disease or goal[:30], gene)
            strategy   = f"Disease genomics pipeline for {gene}"
        elif intent == "protein_structure" and uniprot:
            node_specs = _protein_structure_plan(uniprot, ligand or None)
            strategy   = f"Protein structure analysis pipeline for {uniprot}"
        elif gene:
            node_specs = _gene_drug_target_plan(gene, depth)
            strategy   = f"Comprehensive gene analysis pipeline for {gene}"
        else:
            # Fallback: PubMed + hypothesis
            n0 = uuid4().hex[:6]
            n1 = uuid4().hex[:6]
            node_specs = [
                {"id": n0, "tool": "search_pubmed",              "args": {"query": goal, "max_results": 10}, "deps": [], "rationale": "Primary literature survey"},
                {"id": n1, "tool": "generate_research_hypothesis","args": {"topic": goal, "max_hypotheses": 3}, "deps": [n0], "rationale": "Hypothesis generation from literature"},
            ]
            strategy = "Literature-driven research pipeline"

        nodes = [
            PlanNode(
                node_id=spec["id"],
                tool_name=spec["tool"],
                tool_args=spec["args"],
                depends_on=spec.get("deps", []),
                rationale=spec.get("rationale", ""),
            )
            for spec in node_specs
        ]

        return ResearchPlan(
            plan_id=uuid4().hex[:8],
            goal=goal,
            nodes=nodes,
            strategy=strategy,
            depth=depth,
        )

    async def execute(
        self,
        plan: ResearchPlan,
        timeout_per_tool: float = 60.0,
    ) -> dict[str, Any]:
        """
        Execute a research plan, respecting dependency order.
        Parallelizes within each dependency level.
        """
        t_start    = time.monotonic()
        all_results: dict[str, Any] = {}

        levels = plan.execution_levels()
        logger.info(f"[Planner] Executing plan '{plan.goal}' "
                    f"— {len(plan.nodes)} steps in {len(levels)} levels")

        for level_idx, level in enumerate(levels):
            logger.info(f"[Planner] Level {level_idx + 1}/{len(levels)}: "
                        f"running {[n.tool_name for n in level]} in parallel")

            tasks = [
                asyncio.create_task(self._execute_node(node, timeout_per_tool))
                for node in level
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            for node in level:
                if node.status == NodeStatus.COMPLETE:
                    all_results[node.node_id] = node.result

        elapsed = round(time.monotonic() - t_start, 2)

        # Build integrated report
        return {
            "goal":              plan.goal,
            "strategy":          plan.strategy,
            "depth":             plan.depth,
            "total_elapsed_s":   elapsed,
            "execution_summary": plan.summary(),
            "results":           all_results,
            "insights":          self._synthesize_insights(plan, all_results),
        }

    async def _execute_node(
        self,
        node: PlanNode,
        timeout: float,
    ) -> None:
        """Execute a single plan node with timeout and error capture."""
        node.status     = NodeStatus.RUNNING
        node.started_at = time.monotonic()
        try:
            node.result      = await asyncio.wait_for(
                self._dispatch(node.tool_name, node.tool_args),
                timeout=timeout,
            )
            node.status      = NodeStatus.COMPLETE
            node.completed_at= time.monotonic()
            logger.debug(f"[Planner] ✓ {node.tool_name} ({node.elapsed_s}s)")
        except TimeoutError:
            node.status       = NodeStatus.FAILED
            node.error        = f"Timed out after {timeout}s"
            node.completed_at = time.monotonic()
            logger.warning(f"[Planner] ✗ {node.tool_name} timed out")
        except Exception as exc:
            node.status       = NodeStatus.FAILED
            node.error        = str(exc)
            node.completed_at = time.monotonic()
            logger.warning(f"[Planner] ✗ {node.tool_name}: {exc}")

    def _synthesize_insights(
        self,
        plan: ResearchPlan,
        results: dict[str, Any],
    ) -> list[str]:
        """
        Generate high-level insights by cross-referencing results.
        This is where the planner adds value beyond raw tool calls.
        """
        insights: list[str] = []

        # Extract drug count from drug targets
        drug_nodes = [n for n in plan.nodes if n.tool_name == "get_drug_targets" and n.status == NodeStatus.COMPLETE]
        for node in drug_nodes:
            result = node.result or {}
            if isinstance(result, str):
                import json
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {}
            drugs = result.get("data", {}).get("drugs", []) if isinstance(result, dict) else []
            if isinstance(result, dict) and "drugs" in result:
                drugs = result["drugs"]
            total = result.get("total_activities", len(drugs)) if isinstance(result, dict) else 0
            gene  = node.tool_args.get("gene_symbol", "")
            if total:
                insights.append(f"{gene} has {total} ChEMBL compound activities — active drug target.")

        # Cross-reference variants with clinical trials
        variant_nodes = [n for n in plan.nodes if n.tool_name == "get_gene_variants" and n.status == NodeStatus.COMPLETE]
        trial_nodes   = [n for n in plan.nodes if n.tool_name == "search_clinical_trials" and n.status == NodeStatus.COMPLETE]
        if variant_nodes and trial_nodes:
            insights.append("Variant data and clinical trials retrieved — cross-reference mutations with trial eligibility criteria.")

        # Literature vs database gap
        pubmed_nodes = [n for n in plan.nodes if n.tool_name == "search_pubmed" and n.status == NodeStatus.COMPLETE]
        if pubmed_nodes and drug_nodes:
            insights.append("Literature and ChEMBL data integrated — validate experimental IC50 values against published reports.")

        complete = sum(1 for n in plan.nodes if n.status == NodeStatus.COMPLETE)
        failed   = sum(1 for n in plan.nodes if n.status == NodeStatus.FAILED)
        if failed:
            insights.append(f"⚠ {failed} tool call(s) failed — retry individually for complete data coverage.")
        insights.append(f"Research plan executed: {complete}/{len(plan.nodes)} steps successful.")

        return insights

    async def plan_and_execute(
        self,
        goal: str,
        depth: str = "standard",
        entities: dict[str, str] | None = None,
        timeout_per_tool: float = 60.0,
    ) -> dict[str, Any]:
        """
        One-shot: build plan, execute it, return integrated report.

        Args:
            goal:             Natural language research objective.
            depth:            'quick' | 'standard' | 'deep'. Default 'standard'.
            entities:         Override auto-extracted entities.
            timeout_per_tool: Per-tool timeout in seconds.
        """
        plan   = self.build_plan(goal, depth=depth, entities=entities)
        result = await self.execute(plan, timeout_per_tool=timeout_per_tool)
        return result
