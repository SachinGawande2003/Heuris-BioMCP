"""
BioMCP — Biological Entity Resolver  [FIXED v2.2]
=====================================
Fixes applied:
  - Bug #9: Module-level asyncio.Lock() replaced with lazy init pattern
    (consistent with knowledge_graph.py v2.1 fix, robust across Python versions)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class BioEntity:
    """Fully resolved biological entity with IDs from all major databases."""
    canonical_symbol:   str
    canonical_name:     str
    entity_class:       str

    hgnc_id:            str = ""
    ncbi_gene_id:       str = ""
    ensembl_gene_id:    str = ""
    uniprot_accession:  str = ""
    refseq_mrna:        str = ""
    refseq_protein:     str = ""
    chembl_target_id:   str = ""
    omim_id:            str = ""
    pharmgkb_id:        str = ""

    aliases:            list[str]  = field(default_factory=list)
    organism:           str        = "Homo sapiens"
    chromosome:         str        = ""
    resolution_sources: list[str]  = field(default_factory=list)
    confidence:         float      = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_symbol":  self.canonical_symbol,
            "canonical_name":    self.canonical_name,
            "entity_class":      self.entity_class,
            "cross_references": {
                "hgnc_id":           self.hgnc_id,
                "ncbi_gene_id":      self.ncbi_gene_id,
                "ensembl_gene_id":   self.ensembl_gene_id,
                "uniprot_accession": self.uniprot_accession,
                "refseq_mrna":       self.refseq_mrna,
                "omim_id":           self.omim_id,
            },
            "aliases":            self.aliases,
            "organism":           self.organism,
            "chromosome":         self.chromosome,
            "resolution_sources": self.resolution_sources,
            "confidence":         self.confidence,
        }


class EntityRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, BioEntity] = {}
        self._alias_map: dict[str, str]       = {}
        self._lock = asyncio.Lock()

    async def register(self, entity: BioEntity) -> None:
        async with self._lock:
            self._registry[entity.canonical_symbol] = entity
            for alias in [entity.canonical_symbol] + entity.aliases:
                self._alias_map[alias.lower().strip()] = entity.canonical_symbol
            if entity.ncbi_gene_id:
                self._alias_map[entity.ncbi_gene_id] = entity.canonical_symbol
            if entity.uniprot_accession:
                self._alias_map[entity.uniprot_accession.lower()] = entity.canonical_symbol

    def lookup(self, query: str) -> BioEntity | None:
        key = query.lower().strip()
        canonical = self._alias_map.get(key)
        return self._registry.get(canonical) if canonical else None

    def all_entities(self) -> list[BioEntity]:
        return list(self._registry.values())


class EntityResolver:
    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    async def resolve(self, query: str, hint_type: str = "gene") -> BioEntity:
        cached = self._registry.lookup(query)
        if cached:
            return cached

        logger.info(f"[EntityResolver] Resolving '{query}'")
        results = await asyncio.gather(
            self._resolve_via_ncbi(query),
            self._resolve_via_uniprot(query),
            self._resolve_via_ensembl(query),
            return_exceptions=True,
        )
        entity = _merge_resolution_results(query, hint_type, list(results))
        await self._registry.register(entity)
        return entity

    async def _resolve_via_ncbi(self, query: str) -> dict[str, Any]:
        try:
            from biomcp.utils import get_http_client, ncbi_params
            client = await get_http_client()
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=ncbi_params({
                    "db": "gene",
                    "term": f"{query}[Gene Name] AND Homo sapiens[Organism]",
                    "retmax": 1,
                }),
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return {}
            summ = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=ncbi_params({"db": "gene", "id": ids[0]}),
            )
            summ.raise_for_status()
            gd = summ.json().get("result", {}).get(ids[0], {})
            return {
                "source":           "NCBI Gene",
                "canonical_symbol": gd.get("name", query).upper(),
                "canonical_name":   gd.get("description", ""),
                "ncbi_gene_id":     ids[0],
                "chromosome":       gd.get("chromosome", ""),
                "aliases":          [a.strip() for a in gd.get("otheraliases", "").split(",") if a.strip()],
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] NCBI failed for '{query}': {exc}")
            return {}

    async def _resolve_via_uniprot(self, query: str) -> dict[str, Any]:
        try:
            from biomcp.utils import get_http_client
            client = await get_http_client()
            resp = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query":  f"gene:{query} AND organism_id:9606 AND reviewed:true",
                    "format": "json", "size": 1,
                    "fields": "accession,gene_names,protein_name,xref_refseq",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return {}
            entry     = results[0]
            acc       = entry.get("primaryAccession", "")
            genes     = entry.get("genes", [])
            gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""
            rec_name  = (entry.get("proteinDescription", {})
                         .get("recommendedName", {})
                         .get("fullName", {})
                         .get("value", ""))
            return {
                "source":            "UniProt",
                "canonical_symbol":  gene_name.upper() if gene_name else query.upper(),
                "canonical_name":    rec_name,
                "uniprot_accession": acc,
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] UniProt failed for '{query}': {exc}")
            return {}

    async def _resolve_via_ensembl(self, query: str) -> dict[str, Any]:
        try:
            from biomcp.utils import get_http_client
            client = await get_http_client()
            resp = await client.get(
                f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{query}",
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            hits = [e for e in resp.json() if e.get("type") == "gene"]
            if not hits:
                return {}
            return {
                "source":          "Ensembl",
                "ensembl_gene_id": hits[0].get("id", ""),
            }
        except Exception as exc:
            logger.debug(f"[EntityResolver] Ensembl failed for '{query}': {exc}")
            return {}


def _merge_resolution_results(query: str, hint_type: str, results: list[Any]) -> BioEntity:
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for result in results:
        if isinstance(result, Exception) or not result:
            continue
        sources.append(result.get("source", ""))
        for key, val in result.items():
            if key == "source":
                continue
            if key == "aliases":
                merged.setdefault("aliases", [])
                merged["aliases"].extend(val)
            elif not merged.get(key):
                merged[key] = val

    aliases   = list(set(merged.get("aliases", [])))
    canonical = (merged.get("canonical_symbol") or query).upper().strip()
    canonical = re.sub(r"[^A-Z0-9\-]", "", canonical) or query.upper()

    return BioEntity(
        canonical_symbol    = canonical,
        canonical_name      = merged.get("canonical_name", ""),
        entity_class        = hint_type,
        ncbi_gene_id        = merged.get("ncbi_gene_id", ""),
        ensembl_gene_id     = merged.get("ensembl_gene_id", ""),
        uniprot_accession   = merged.get("uniprot_accession", ""),
        chromosome          = merged.get("chromosome", ""),
        aliases             = aliases,
        resolution_sources  = [s for s in sources if s],
        confidence          = min(1.0, 0.33 * len([s for s in sources if s])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX #9: Lazy singleton — no module-level asyncio.Lock()
# The previous code had `_INIT_LOCK = asyncio.Lock()` at module top,
# which could fail before an event loop exists in Python <3.10.
# Now mirrors the knowledge_graph.py lazy pattern.
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY:   EntityRegistry   | None = None
_RESOLVER:   EntityResolver   | None = None
_LAZY_LOCK:  asyncio.Lock     | None = None   # FIX #9: lazily constructed inside event loop


def _get_init_lock() -> asyncio.Lock:
    """Lazily create the resolver init lock inside a running event loop."""
    global _LAZY_LOCK
    if _LAZY_LOCK is None:
        _LAZY_LOCK = asyncio.Lock()
    return _LAZY_LOCK


async def get_resolver() -> EntityResolver:
    """Return the session-scoped entity resolver singleton."""
    global _REGISTRY, _RESOLVER
    if _RESOLVER is None:
        async with _get_init_lock():
            if _RESOLVER is None:
                _REGISTRY = EntityRegistry()
                _RESOLVER = EntityResolver(_REGISTRY)
    return _RESOLVER


async def get_registry() -> EntityRegistry:
    """Return the session-scoped entity registry."""
    await get_resolver()
    return _REGISTRY  # type: ignore[return-value]
