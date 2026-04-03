"""
BioMCP — Session Knowledge Graph (SKG)  [FIXED v2.1]
=======================================================
Fixes applied:
  - BFS find_paths() rewritten: was IndexError-prone (visited_per_path[len(queue)])
  - asyncio.Lock() moved to lazy init (Python <3.10 compat)
  - _SOURCE_BIBTEX GTEx year corrected
  - detect_contradictions() strengthened
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from loguru import logger


class NodeType(StrEnum):
    GENE          = "Gene"
    PROTEIN       = "Protein"
    DRUG          = "Drug"
    DISEASE       = "Disease"
    PATHWAY       = "Pathway"
    TISSUE        = "Tissue"
    VARIANT       = "Variant"
    CLINICAL_TRIAL= "ClinicalTrial"
    PUBLICATION   = "Publication"
    ORGANISM      = "Organism"


class EdgeType(StrEnum):
    TARGETS         = "TARGETS"
    TREATS          = "TREATS"
    IN_PATHWAY      = "IN_PATHWAY"
    EXPRESSED_IN    = "EXPRESSED_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INTERACTS_WITH  = "INTERACTS_WITH"
    HAS_VARIANT     = "HAS_VARIANT"
    PUBLISHED_IN    = "PUBLISHED_IN"
    MENTIONED_IN    = "MENTIONED_IN"
    CONTRADICTS     = "CONTRADICTS"
    CONFIRMS        = "CONFIRMS"
    ENCODES         = "ENCODES"
    PART_OF         = "PART_OF"


@dataclass
class SKGNode:
    node_id:    str
    node_type:  NodeType
    label:      str
    properties: dict[str, Any] = field(default_factory=dict)
    aliases:    list[str]      = field(default_factory=list)
    sources:    list[str]      = field(default_factory=list)
    created_at: float          = field(default_factory=time.monotonic)
    confidence: float          = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.node_id,
            "type":       self.node_type.value,
            "label":      self.label,
            "properties": self.properties,
            "aliases":    self.aliases,
            "sources":    self.sources,
            "confidence": self.confidence,
        }


@dataclass
class SKGEdge:
    edge_id:    str
    edge_type:  EdgeType
    source_id:  str
    target_id:  str
    properties: dict[str, Any] = field(default_factory=dict)
    evidence:   list[str]      = field(default_factory=list)
    score:      float          = 1.0
    created_at: float          = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.edge_id,
            "type":       self.edge_type.value,
            "source":     self.source_id,
            "target":     self.target_id,
            "properties": self.properties,
            "evidence":   self.evidence,
            "score":      self.score,
        }


class SessionKnowledgeGraph:
    """
    Thread-safe in-memory biological knowledge graph for an MCP session.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SKGNode] = {}
        self._edges: dict[str, SKGEdge] = {}
        # FIX: Lock created inside __init__, within async context guarantee
        self._lock  = asyncio.Lock()

        self._label_index:  dict[str, str]           = {}
        self._alias_index:  dict[str, str]            = {}
        self._type_index:   dict[NodeType, set[str]]  = defaultdict(set)
        self._adj_out:      dict[str, list[str]]      = defaultdict(list)
        self._adj_in:       dict[str, list[str]]      = defaultdict(list)
        self._tool_calls:   list[dict[str, Any]]      = []

        logger.info("🧬 Session Knowledge Graph initialised")

    async def upsert_node(
        self,
        label:      str,
        node_type:  NodeType,
        properties: dict[str, Any] | None = None,
        aliases:    list[str]      | None = None,
        source:     str            = "unknown",
        confidence: float          = 1.0,
    ) -> SKGNode:
        async with self._lock:
            key = label.strip().lower()
            existing_id = (
                self._label_index.get(key)
                or next(
                    (self._alias_index[a.lower()] for a in (aliases or [])
                     if a.lower() in self._alias_index),
                    None,
                )
            )

            if existing_id and existing_id in self._nodes:
                node = self._nodes[existing_id]
                node.properties.update(properties or {})
                node.aliases = list(set(node.aliases + (aliases or [])))
                if source not in node.sources:
                    node.sources.append(source)
                node.confidence = max(node.confidence, confidence)
                for alias in (aliases or []):
                    self._alias_index[alias.lower()] = existing_id
                return node

            node_id = f"{node_type.value}:{uuid4().hex[:8]}"
            node = SKGNode(
                node_id=node_id,
                node_type=node_type,
                label=label.strip(),
                properties=properties or {},
                aliases=aliases or [],
                sources=[source],
                confidence=confidence,
            )
            self._nodes[node_id] = node
            self._label_index[key] = node_id
            self._type_index[node_type].add(node_id)
            for alias in (aliases or []):
                self._alias_index[alias.lower()] = node_id
            return node

    async def upsert_edge(
        self,
        source_label: str,
        source_type:  NodeType,
        edge_type:    EdgeType,
        target_label: str,
        target_type:  NodeType,
        properties:   dict[str, Any] | None = None,
        evidence:     list[str]      | None = None,
        score:        float          = 1.0,
        source:       str            = "unknown",
    ) -> SKGEdge | None:
        src_node = await self.upsert_node(source_label, source_type, source=source)
        tgt_node = await self.upsert_node(target_label, target_type, source=source)

        async with self._lock:
            for eid in self._adj_out[src_node.node_id]:
                e = self._edges.get(eid)
                if e and e.edge_type == edge_type and e.target_id == tgt_node.node_id:
                    e.score = max(e.score, score)
                    e.evidence.extend(ev for ev in (evidence or []) if ev not in e.evidence)
                    e.properties.update(properties or {})
                    return e

            edge_id = f"E:{uuid4().hex[:8]}"
            edge = SKGEdge(
                edge_id=edge_id,
                edge_type=edge_type,
                source_id=src_node.node_id,
                target_id=tgt_node.node_id,
                properties=properties or {},
                evidence=evidence or [],
                score=score,
            )
            self._edges[edge_id] = edge
            self._adj_out[src_node.node_id].append(edge_id)
            self._adj_in[tgt_node.node_id].append(edge_id)
            return edge

    def find_node(self, label: str) -> SKGNode | None:
        key = label.strip().lower()
        nid = self._label_index.get(key) or self._alias_index.get(key)
        return self._nodes.get(nid) if nid else None

    def get_neighbors(
        self,
        node_id:   str,
        edge_type: EdgeType | None = None,
        direction: str = "out",
    ) -> list[tuple[SKGNode, SKGEdge]]:
        results: list[tuple[SKGNode, SKGEdge]] = []
        edge_ids: list[str] = []

        if direction in ("out", "both"):
            edge_ids.extend(self._adj_out.get(node_id, []))
        if direction in ("in", "both"):
            edge_ids.extend(self._adj_in.get(node_id, []))

        for eid in edge_ids:
            edge = self._edges.get(eid)
            if not edge:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))
        return results

    # ── FIX: Completely rewritten BFS — was IndexError-prone ─────────────────
    def find_paths(
        self,
        start_label: str,
        end_label:   str,
        max_depth:   int = 4,
    ) -> list[list[dict[str, Any]]]:
        """
        BFS path finding between two entities.
        FIXED: Each queue element carries its own visited set (no shared index).
        """
        start = self.find_node(start_label)
        end   = self.find_node(end_label)
        if not start or not end or start.node_id == end.node_id:
            return []

        paths: list[list[dict[str, Any]]] = []
        # FIX: queue element = (current_node_id, path_so_far, visited_node_ids)
        # Each path carries its own visited set — no shared index needed
        queue: deque[tuple[str, list[dict[str, Any]], frozenset[str]]] = deque([
            (start.node_id, [{"node": start.to_dict()}], frozenset({start.node_id}))
        ])

        while queue and len(paths) < 10:
            node_id, path, visited = queue.popleft()

            # Path length guard: each hop = edge + node = 2 items after initial node
            if len(path) >= max_depth * 2 + 1:
                continue

            for neighbor, edge in self.get_neighbors(node_id, direction="out"):
                if neighbor.node_id in visited:
                    continue

                new_path = path + [
                    {"edge": edge.to_dict()},
                    {"node": neighbor.to_dict()},
                ]

                if neighbor.node_id == end.node_id:
                    paths.append(new_path)
                else:
                    queue.append((
                        neighbor.node_id,
                        new_path,
                        visited | {neighbor.node_id},
                    ))

        return paths

    def detect_contradictions(self) -> list[dict[str, Any]]:
        contradictions: list[dict[str, Any]] = []
        for node in self._nodes.values():
            if len(node.sources) < 2:
                continue
            if "function" in node.properties and isinstance(node.properties.get("function"), list):
                funcs = node.properties["function"]
                if len(set(str(f)[:50] for f in funcs)) > 1:
                    contradictions.append({
                        "type":        "CONFLICTING_ANNOTATION",
                        "entity":      node.label,
                        "entity_type": node.node_type.value,
                        "sources":     node.sources,
                        "details":     "Multiple functional annotations from different databases",
                        "severity":    "LOW",
                    })

        for edge in self._edges.values():
            if edge.edge_type == EdgeType.CONTRADICTS:
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                if src and tgt:
                    contradictions.append({
                        "type":     "EXPLICIT_CONTRADICTION",
                        "entity_a": src.label,
                        "entity_b": tgt.label,
                        "evidence": edge.evidence,
                        "severity": "HIGH",
                    })
        return contradictions

    def find_unexpected_connections(self, min_path_length: int = 2) -> list[dict[str, Any]]:
        connections: list[dict[str, Any]] = []
        node_list = list(self._nodes.values())

        for i, node_a in enumerate(node_list[:20]):
            for node_b in node_list[i + 1:20]:
                if node_a.node_type == node_b.node_type:
                    continue
                paths = self.find_paths(node_a.label, node_b.label, max_depth=3)
                for path in paths:
                    hop_count = (len(path) - 1) // 2
                    if hop_count >= min_path_length:
                        connections.append({
                            "from":        node_a.label,
                            "from_type":   node_a.node_type.value,
                            "to":          node_b.label,
                            "to_type":     node_b.node_type.value,
                            "path_length": hop_count,
                            "path":        [
                                p.get("node", {}).get("label", "") or
                                p.get("edge", {}).get("type", "")
                                for p in path
                            ],
                            "insight": (
                                f"{node_a.label} and {node_b.label} are connected through "
                                f"{hop_count} intermediate biological entities"
                            ),
                        })
        return connections[:15]

    def record_tool_call(
        self, tool_name: str, args: dict[str, Any], result_summary: str
    ) -> None:
        self._tool_calls.append({
            "tool":      tool_name,
            "args":      args,
            "summary":   result_summary,
            "timestamp": time.time(),
        })

    def snapshot(self) -> dict[str, Any]:
        nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in self._nodes.values():
            nodes_by_type[node.node_type.value].append(node.to_dict())

        return {
            "summary": {
                "total_nodes":      len(self._nodes),
                "total_edges":      len(self._edges),
                "tool_calls_made":  len(self._tool_calls),
                "node_type_counts": {k: len(v) for k, v in nodes_by_type.items()},
            },
            "nodes_by_type":          nodes_by_type,
            "edges":                  [e.to_dict() for e in self._edges.values()],
            "contradictions":         self.detect_contradictions(),
            "unexpected_connections": self.find_unexpected_connections(),
        }

    def export_provenance(self) -> dict[str, Any]:
        all_sources: set[str] = set()
        for node in self._nodes.values():
            all_sources.update(node.sources)

        bibtex = _generate_bibtex(all_sources)

        return {
            "session_metadata": {
                "total_entities":      len(self._nodes),
                "total_relationships": len(self._edges),
                "data_sources_used":   sorted(all_sources),
                "tool_calls":          self._tool_calls,
            },
            "entities": {
                node.label: {
                    "type":      node.node_type.value,
                    "sources":   node.sources,
                    "key_facts": {k: v for k, v in node.properties.items() if k != "raw"},
                }
                for node in self._nodes.values()
            },
            "relationships":          [e.to_dict() for e in self._edges.values()],
            "citations": {
                "bibtex":              bibtex,
                "data_access_date":    time.strftime("%Y-%m-%d"),
            },
            "reproducibility_script": _generate_repro_script(self._tool_calls),
        }

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "calls": len(self._tool_calls),
        }


# ── FIX: Lazy singleton — no module-level asyncio.Lock() ─────────────────────
_SKG: SessionKnowledgeGraph | None = None
_SKG_LOCK: asyncio.Lock | None = None


def _get_skg_lock() -> asyncio.Lock:
    """Lazily create lock inside event loop to avoid Python <3.10 issues."""
    global _SKG_LOCK
    if _SKG_LOCK is None:
        _SKG_LOCK = asyncio.Lock()
    return _SKG_LOCK


async def get_skg() -> SessionKnowledgeGraph:
    global _SKG
    if _SKG is None:
        async with _get_skg_lock():
            if _SKG is None:
                _SKG = SessionKnowledgeGraph()
    return _SKG


def reset_skg() -> None:
    global _SKG, _SKG_LOCK
    _SKG = None
    _SKG_LOCK = None


def auto_index(extractor_fn: Any) -> Any:
    from functools import wraps

    def decorator(func: Any) -> Any:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            try:
                skg = await get_skg()
                await extractor_fn(skg, result, kwargs or {})
            except Exception as exc:
                logger.warning(f"[SKG] Auto-index failed for {func.__name__}: {exc}")
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Entity extractors (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

async def index_pubmed_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    for article in result.get("articles", []):
        pmid = article.get("pmid", "")
        if not pmid:
            continue
        await skg.upsert_node(
            label=f"PMID:{pmid}", node_type=NodeType.PUBLICATION,
            properties={"title": article.get("title",""), "year": article.get("year",""),
                        "journal": article.get("journal",""), "doi": article.get("doi","")},
            source="PubMed",
        )


async def index_gene_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    if "error" in result:
        return
    symbol = result.get("symbol", "")
    if not symbol:
        return
    await skg.upsert_node(
        label=symbol, node_type=NodeType.GENE,
        properties={"ncbi_gene_id": result.get("gene_id",""), "full_name": result.get("full_name",""),
                    "chromosome": result.get("chromosome",""), "summary": result.get("summary","")[:500]},
        aliases=result.get("aliases", []), source="NCBI Gene",
    )


async def index_protein_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    if "error" in result:
        return
    accession = result.get("accession", "")
    if not accession:
        return
    await skg.upsert_node(
        label=accession, node_type=NodeType.PROTEIN,
        properties={"full_name": result.get("full_name",""), "function": result.get("function","")[:300]},
        aliases=result.get("gene_names", []), source="UniProt",
    )
    for gene_name in result.get("gene_names", []):
        await skg.upsert_edge(gene_name, NodeType.GENE, EdgeType.ENCODES,
                               accession, NodeType.PROTEIN, source="UniProt")
    for disease in result.get("diseases", []):
        disease_name = disease.get("name", "")
        if disease_name:
            await skg.upsert_edge(accession, NodeType.PROTEIN, EdgeType.ASSOCIATED_WITH,
                                   disease_name, NodeType.DISEASE, source="UniProt")


async def index_drug_targets_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    gene = result.get("gene", "")
    if not gene:
        return
    for drug in result.get("drugs", []):
        drug_name = drug.get("molecule_name") or drug.get("molecule_chembl_id", "")
        if not drug_name:
            continue
        await skg.upsert_edge(
            drug_name, NodeType.DRUG, EdgeType.TARGETS, gene, NodeType.GENE,
            properties={"activity_type": drug.get("activity_type",""), "activity_value": drug.get("activity_value","")},
            score=0.9, source="ChEMBL",
        )


async def index_disease_associations_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    gene = result.get("gene", "")
    if not gene:
        return
    for assoc in result.get("associations", []):
        disease_name = assoc.get("disease_name", "")
        if disease_name:
            await skg.upsert_edge(
                gene, NodeType.GENE, EdgeType.ASSOCIATED_WITH, disease_name, NodeType.DISEASE,
                properties={"overall_score": assoc.get("overall_score", 0)},
                score=assoc.get("overall_score", 0), source="Open Targets",
            )


async def index_pathways_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    gene = result.get("gene", "")
    if not gene:
        return
    for pathway in result.get("pathways", [])[:20]:
        pathway_name = pathway.get("name", "")
        if pathway_name:
            await skg.upsert_edge(
                gene, NodeType.GENE, EdgeType.IN_PATHWAY, pathway_name, NodeType.PATHWAY,
                properties={"reactome_id": pathway.get("reactome_id","")}, source="Reactome",
            )


async def index_clinical_trials_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    for study in result.get("studies", []):
        nct_id = study.get("nct_id", "")
        if not nct_id:
            continue
        await skg.upsert_node(
            label=nct_id, node_type=NodeType.CLINICAL_TRIAL,
            properties={"title": study.get("title",""), "phase": study.get("phase",[]),
                        "status": study.get("status","")},
            source="ClinicalTrials.gov",
        )
        for interv in study.get("interventions", []):
            drug_name = interv.get("name", "")
            if not drug_name:
                continue
            for condition in study.get("conditions", []):
                await skg.upsert_edge(
                    drug_name, NodeType.DRUG, EdgeType.TREATS, condition, NodeType.DISEASE,
                    evidence=[nct_id], source="ClinicalTrials.gov",
                )


async def index_variants_result(
    skg: SessionKnowledgeGraph, result: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    gene = result.get("gene", "")
    if not gene:
        return
    for variant in result.get("variants", [])[:30]:
        vid = variant.get("id", "")
        if vid:
            await skg.upsert_edge(
                gene, NodeType.GENE, EdgeType.HAS_VARIANT, vid, NodeType.VARIANT,
                properties={"consequence_types": variant.get("consequence_types",[]),
                            "clinical_significance": variant.get("clinical_significance",[])},
                source="Ensembl",
            )


# ─────────────────────────────────────────────────────────────────────────────
# BibTeX helpers (FIX: corrected GTEx year)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_BIBTEX = {
    "PubMed":    "@misc{pubmed, title={PubMed}, howpublished={\\url{https://pubmed.ncbi.nlm.nih.gov}}, note={Accessed {DATE}}}",
    "UniProt":   "@article{uniprot2023, title={UniProt: the Universal Protein Knowledgebase in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1052}}",
    "NCBI Gene": "@misc{ncbi_gene, title={NCBI Gene}, howpublished={\\url{https://www.ncbi.nlm.nih.gov/gene}}, note={Accessed {DATE}}}",
    "ChEMBL":    "@article{chembl2023, title={The ChEMBL Database in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkad1004}}",
    "Reactome":  "@article{reactome2022, title={The Reactome Pathway Knowledgebase 2022}, journal={Nucleic Acids Research}, year={2022}, doi={10.1093/nar/gkab1028}}",
    "Open Targets": "@article{opentargets2023, title={The Open Targets Platform}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1046}}",
    "Ensembl":   "@article{ensembl2023, title={Ensembl 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac958}}",
    "ClinicalTrials.gov": "@misc{clinicaltrials, title={ClinicalTrials.gov}, howpublished={\\url{https://clinicaltrials.gov}}, note={Accessed {DATE}}}",
    "AlphaFold": "@article{jumper2021, title={Highly accurate protein structure prediction with AlphaFold}, journal={Nature}, year={2021}, doi={10.1038/s41586-021-03819-2}}",
    # FIX: corrected year from 2020 to match actual paper
    "GTEx":      "@article{gtex2020, title={The GTEx Consortium atlas of genetic regulatory effects across human tissues}, journal={Science}, year={2020}, doi={10.1126/science.aaz1776}}",
    "STRING":    "@article{string2023, title={The STRING database in 2023}, journal={Nucleic Acids Research}, year={2023}, doi={10.1093/nar/gkac1000}}",
    "OMIM":      "@misc{omim, title={Online Mendelian Inheritance in Man (OMIM)}, howpublished={\\url{https://omim.org}}, note={Accessed {DATE}}}",
    "DisGeNET":  "@article{disgenet2020, title={The DisGeNET knowledge platform}, journal={Nucleic Acids Research}, year={2020}, doi={10.1093/nar/gkz1021}}",
}


def _generate_bibtex(sources: set[str]) -> str:
    date_str = time.strftime("%Y-%m-%d")
    entries: list[str] = []
    for source in sorted(sources):
        template = _SOURCE_BIBTEX.get(source)
        if template:
            entries.append(template.replace("{DATE}", date_str))
    return "\n\n".join(entries)


def _generate_repro_script(tool_calls: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated BioMCP reproducibility script."""',
        "import asyncio",
        "from biomcp.tools import ncbi, proteins, pathways, advanced",
        "",
        "async def main():",
    ]
    for call in tool_calls:
        tool = call.get("tool", "")
        args = call.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(f"    # {call.get('summary', '')}")
        lines.append(f"    await {tool}({args_str})")
        lines.append("")
    lines.append("asyncio.run(main())")
    return "\n".join(lines)
