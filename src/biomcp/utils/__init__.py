"""
BioMCP Utilities
================
Core infrastructure shared by all tool modules:

  • Async HTTP client  — pooled, connection-reused, user-agent set
  • Rate limiter       — token-bucket per API service
  • TTL cache          — namespace-isolated, configurable TTL
  • Retry decorator    — exponential backoff via tenacity
  • BioValidator       — validates all biological identifier formats
  • Response helpers   — format_success / format_error for MCP output

All public symbols are re-exported from this package so tool modules
only need:  from biomcp.utils import cached, rate_limited, BioValidator, ...
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

F = TypeVar("F", bound=Callable[..., Any])

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

_HTTP_TIMEOUT = float(os.getenv("BIOMCP_HTTP_TIMEOUT", "30"))
_CACHE_SIZE = int(os.getenv("BIOMCP_CACHE_SIZE", "512"))
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Async HTTP Client
# ─────────────────────────────────────────────────────────────────────────────

_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_CLIENT_LOOP_ID: int | None = None
_HTTP_LOCK: asyncio.Lock | None = None


async def _close_http_client_safely(
    client: httpx.AsyncClient,
    *,
    reason: str,
) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        logger.debug(f"HTTP client close skipped after {reason}: {exc}")
    else:
        logger.debug(f"HTTP client closed after {reason}")


def _schedule_http_client_close(
    client: httpx.AsyncClient | None,
    *,
    reason: str,
) -> None:
    if client is None or client.is_closed:
        return
    try:
        asyncio.get_running_loop().create_task(
            _close_http_client_safely(client, reason=reason)
        )
    except RuntimeError as exc:
        logger.debug(f"Unable to schedule HTTP client close after {reason}: {exc}")


async def get_http_client() -> httpx.AsyncClient:
    """
    Return the module-level shared async HTTP client.
    Thread-safe lazy init with asyncio.Lock.
    Connection pooling is handled by httpx internally.
    """
    global _HTTP_CLIENT, _HTTP_CLIENT_LOOP_ID, _HTTP_LOCK
    current_loop_id = id(asyncio.get_running_loop())

    if _HTTP_LOCK is None or _HTTP_CLIENT_LOOP_ID != current_loop_id:
        _HTTP_LOCK = asyncio.Lock()

    if (
        _HTTP_CLIENT is not None
        and not _HTTP_CLIENT.is_closed
        and _HTTP_CLIENT_LOOP_ID is not None
        and _HTTP_CLIENT_LOOP_ID != current_loop_id
    ):
        stale_client = _HTTP_CLIENT
        logger.debug("Replacing HTTP client bound to a different event loop")
        _HTTP_CLIENT = None
        _HTTP_CLIENT_LOOP_ID = None
        _schedule_http_client_close(stale_client, reason="event-loop change")

    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        async with _HTTP_LOCK:
            if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
                _HTTP_CLIENT = httpx.AsyncClient(
                    timeout=httpx.Timeout(_HTTP_TIMEOUT, connect=10.0),
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "Heuris-BioMCP/2.2 "
                            "(https://github.com/SachinGawande2003/Heuris-BioMCP; "
                            "bioinformatics MCP server for MCP clients)"
                        ),
                        "Accept-Encoding": "gzip, deflate",
                    },
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0,
                    ),
                )
                _HTTP_CLIENT_LOOP_ID = current_loop_id
                logger.debug("HTTP client initialized")
    return _HTTP_CLIENT


async def close_http_client() -> None:
    """Gracefully close the shared HTTP client on server shutdown."""
    global _HTTP_CLIENT, _HTTP_CLIENT_LOOP_ID, _HTTP_LOCK
    client = _HTTP_CLIENT
    client_loop_id = _HTTP_CLIENT_LOOP_ID

    _HTTP_CLIENT = None
    _HTTP_CLIENT_LOOP_ID = None
    _HTTP_LOCK = None

    if client and not client.is_closed:
        current_loop_id = id(asyncio.get_running_loop())
        reason = (
            "shutdown"
            if client_loop_id == current_loop_id
            else "shutdown across event-loop boundary"
        )
        await _close_http_client_safely(client, reason=reason)


# ─────────────────────────────────────────────────────────────────────────────
# TTL Cache — namespace-isolated
# ─────────────────────────────────────────────────────────────────────────────

# TTL values (seconds) tuned per data category
CACHE_TTLS: dict[str, int] = {
    "pubmed": 3_600,  # 1 h  — literature stable within a session
    "uniprot": 86_400,  # 24 h — protein records rarely change
    "alphafold": 604_800,  # 7 d  — structure predictions are static
    "kegg": 43_200,  # 12 h
    "ensembl": 86_400,  # 24 h
    "clinical_trials": 1_800,  # 30 m — trial status can change
    "expression": 3_600,  # 1 h
    "drug_target": 3_600,  # 1 h
    "chembl_registry": 604_800,  # 7 d — target registry is stable and reused heavily
    "blast": 1_800,  # 30 m
    "reactome": 43_200,  # 12 h
    "fda": 3_600,  # 1 h â€” FAERS / labels update frequently enough to stay short
    "omim": 86_400,  # 24 h â€” curated disease annotations
    "string": 43_200,  # 12 h â€” interaction network metadata is moderately stable
    "gtex": 604_800,  # 7 d â€” release-backed reference expression data
    "cbio": 21_600,  # 6 h â€” portal-backed cancer cohorts may refresh intra-day
    "gwas": 86_400,  # 24 h â€” catalog updates are not session-sensitive
    "disgenet": 86_400,  # 24 h â€” curated gene-disease associations
    "pharmgkb": 86_400,  # 24 h â€” pharmacogenomics annotations
    "crispr": 86_400,  # 24 h â€” design inputs are mostly stable genomic references
    "multi_omics": 3_600,  # 1 h â€” aggregate report is expensive and mostly session-stable
    "enrichment": 21_600,  # 6 h â€” derived pathway enrichment results
    "biorxiv": 3_600,  # 1 h â€” preprint search results shift quickly
    "interpro": 604_800,  # 7 d â€” domain/family annotations are release-oriented
    "coexpression": 43_200,  # 12 h â€” inferred expression relationships
    "hotspots": 86_400,  # 24 h â€” cancer hotspot summaries are curated releases
    "splice": 86_400,  # 24 h â€” splice interpretation benefits from daily refresh
    "biogrid": 21_600,  # 6 h â€” curated interaction catalog with periodic updates
    "orphanet": 86_400,  # 24 h â€” rare-disease reference data
    "tcga": 604_800,  # 7 d â€” legacy genomic cohorts are effectively static
    "cellmarker": 604_800,  # 7 d â€” cell marker reference compendia are static-ish
    "encode": 86_400,  # 24 h â€” experiment metadata changes slowly
    "metabolights": 43_200,  # 12 h â€” study metadata is moderately stable
    "ucsc": 86_400,  # 24 h â€” genome browser annotations refresh slowly
    "variant": 86_400,  # 24 h â€” ClinVar freshness dominates mixed variant sources
    "default": 3_600,  # 1 h fallback
}

_CACHES: dict[str, TTLCache] = {}


def get_cache(namespace: str, maxsize: int | None = None) -> TTLCache:
    """Get or lazily create a TTL cache for a given namespace."""
    if namespace not in _CACHES:
        ttl = CACHE_TTLS.get(namespace, CACHE_TTLS["default"])
        size = maxsize or _CACHE_SIZE
        _CACHES[namespace] = TTLCache(maxsize=size, ttl=ttl)
        logger.debug(f"Cache '{namespace}' created — TTL {ttl}s, maxsize {size}")
    return _CACHES[namespace]


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Produce a short deterministic hex key from arbitrary arguments."""
    payload = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _fingerprint_cache_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:20]


def _decorate_cached_result(
    payload: Any,
    *,
    namespace: str,
    cache_key: str,
    cached_at: float,
    ttl_s: int,
    fingerprint: str,
    status: str,
) -> Any:
    response = copy.deepcopy(payload)
    if not isinstance(response, dict):
        return response

    age_s = max(time.time() - cached_at, 0.0)
    response["_cache"] = {
        "namespace": namespace,
        "cache_key": cache_key,
        "fingerprint": fingerprint,
        "status": status,
        "cached_at": round(cached_at, 3),
        "age_s": round(age_s, 3),
        "ttl_s": ttl_s,
        "expires_in_s": round(max(ttl_s - age_s, 0.0), 3),
        "is_stale": age_s >= ttl_s,
    }
    return response


def strip_cache_metadata(payload: Any) -> Any:
    """Recursively remove internal cache metadata from nested result payloads."""
    if isinstance(payload, dict):
        return {
            key: strip_cache_metadata(value)
            for key, value in payload.items()
            if key != "_cache"
        }
    if isinstance(payload, list):
        return [strip_cache_metadata(item) for item in payload]
    return payload


_REFERENCE_TOOL_QUALITY = {
    "get_gene_info": 0.92,
    "get_protein_info": 0.94,
    "get_alphafold_structure": 0.88,
    "get_drug_targets": 0.84,
    "get_gene_disease_associations": 0.86,
    "search_clinical_trials": 0.91,
    "multi_omics_gene_report": 0.89,
    "verify_biological_claim": 0.9,
    "run_blast": 0.83,
    "search_pubmed": 0.78,
}


def _extract_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    if isinstance(value, str):
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def _collect_payload_years(payload: Any) -> list[int]:
    years: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"year", "document_year", "last_updated", "posted_date"}:
                year = _extract_year(value)
                if year is not None:
                    years.append(year)
            years.extend(_collect_payload_years(value))
    elif isinstance(payload, list):
        for item in payload:
            years.extend(_collect_payload_years(item))
    return years


def _normalize_confidence_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "very high"}:
            return 0.9
        if normalized == "medium":
            return 0.7
        if normalized in {"low", "very low"}:
            return 0.45
    return None


def _estimate_recency_score(tool_name: str, payload: dict[str, Any]) -> float:
    years = _collect_payload_years(payload)
    if not years:
        if tool_name in {"get_gene_info", "get_protein_info", "get_alphafold_structure"}:
            return 0.82
        return 0.75

    newest_year = max(years)
    age_years = max(time.gmtime().tm_year - newest_year, 0)
    if age_years <= 1:
        return 0.98
    if age_years <= 3:
        return 0.9
    if age_years <= 5:
        return 0.82
    return 0.7


def _estimate_cross_validation_score(payload: dict[str, Any]) -> float:
    if "conflicts_found" in payload:
        try:
            conflicts_found = int(payload.get("conflicts_found", 0) or 0)
        except (TypeError, ValueError):
            conflicts_found = 0
        return max(0.35, 0.9 - conflicts_found * 0.15)

    if isinstance(payload.get("data_sources"), list):
        return min(1.0, 0.6 + 0.05 * len(payload["data_sources"]))

    if isinstance(payload.get("databases_queried"), list):
        return min(1.0, 0.58 + 0.06 * len(payload["databases_queried"]))

    evidence_counts = payload.get("evidence_counts")
    if isinstance(evidence_counts, dict):
        supporting = int(evidence_counts.get("supporting", 0) or 0)
        contradicting = int(evidence_counts.get("contradicting", 0) or 0)
        return min(0.98, 0.55 + 0.07 * (supporting + contradicting))

    for key in ("associations", "pathways", "drugs", "studies", "datasets", "variants", "proteins"):
        values = payload.get(key)
        if isinstance(values, list):
            if not values:
                return 0.45
            return min(0.9, 0.62 + 0.03 * min(len(values), 8))

    return 0.65


def _estimate_response_confidence(
    tool_name: str,
    payload: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    if "error" in payload or payload.get("status") == "failed":
        confidence = 0.1 if "error" in payload else 0.18
        factors = {
            "source_quality": 0.35,
            "recency": 0.5,
            "cross_validation": 0.2,
        }
        return confidence, factors

    source_quality = _REFERENCE_TOOL_QUALITY.get(tool_name, 0.76)
    recency = _estimate_recency_score(tool_name, payload)
    cross_validation = _estimate_cross_validation_score(payload)

    estimated = (
        source_quality * 0.45
        + recency * 0.2
        + cross_validation * 0.35
    )

    explicit_confidence = (
        _normalize_confidence_value(payload.get("confidence_score"))
        or _normalize_confidence_value(payload.get("overall_confidence"))
        or _normalize_confidence_value(payload.get("consistency_score"))
        or _normalize_confidence_value(payload.get("confidence"))
    )
    if explicit_confidence is not None:
        estimated = explicit_confidence * 0.65 + estimated * 0.35

    factors = {
        "source_quality": round(source_quality, 3),
        "recency": round(recency, 3),
        "cross_validation": round(cross_validation, 3),
    }
    return round(max(0.0, min(1.0, estimated)), 3), factors


def attach_response_meta(tool_name: str, data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    enriched = copy.deepcopy(data)
    confidence, factors = _estimate_response_confidence(tool_name, enriched)
    existing_meta = enriched.get("_meta")
    meta = existing_meta.copy() if isinstance(existing_meta, dict) else {}
    meta.setdefault("confidence", confidence)
    meta.setdefault("confidence_factors", factors)
    meta.setdefault("response_scope", "tool_output")
    enriched["_meta"] = meta
    return enriched


def cached(namespace: str, maxsize: int | None = None) -> Callable[[F], F]:
    """
    Decorator — cache async function results in a named TTL cache.

    Usage::
        @cached("pubmed")
        async def search_pubmed(query: str, ...) -> dict: ...
    """

    def decorator(func: F) -> F:
        cache = get_cache(namespace, maxsize)
        ttl_s = CACHE_TTLS.get(namespace, CACHE_TTLS["default"])

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(*args, **kwargs)
            if key in cache:
                logger.debug(f"[cache HIT] {namespace}:{key[:8]}")
                entry = cache[key]
                if isinstance(entry, dict) and "payload" in entry:
                    return _decorate_cached_result(
                        entry["payload"],
                        namespace=namespace,
                        cache_key=key,
                        cached_at=float(entry["cached_at"]),
                        ttl_s=ttl_s,
                        fingerprint=str(entry["fingerprint"]),
                        status="cached",
                    )
                return copy.deepcopy(entry)
            result = await func(*args, **kwargs)
            cached_at = time.time()
            entry = {
                "payload": copy.deepcopy(result),
                "cached_at": cached_at,
                "fingerprint": _fingerprint_cache_payload(result),
            }
            cache[key] = entry
            logger.debug(f"[cache SET] {namespace}:{key[:8]}")
            return _decorate_cached_result(
                result,
                namespace=namespace,
                cache_key=key,
                cached_at=cached_at,
                ttl_s=ttl_s,
                fingerprint=str(entry["fingerprint"]),
                status="fresh",
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Token-Bucket Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────


class _RateLimiter:
    """
    Async token-bucket rate limiter keyed by service name.

    Each service gets an independent minimum-interval gate.
    Configures NCBI limits correctly (3/s without key, 10/s with key).
    """

    _LIMITS: dict[str, float] = {
        "ncbi": 3.0,
        "ncbi_key": 10.0,
        "uniprot": 10.0,
        "alphafold": 5.0,
        "kegg": 3.0,
        "ensembl": 15.0,
        "clinical_trials": 5.0,
        "chembl": 5.0,
        "reactome": 5.0,
        "geo": 3.0,
        "hca": 5.0,
        "pdb": 10.0,
        "opentargets": 5.0,
        "default": 5.0,
    }

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_call: dict[str, float] = defaultdict(float)

    async def acquire(self, service: str) -> None:
        async with self._locks[service]:
            rps = self._LIMITS.get(service, self._LIMITS["default"])
            interval = 1.0 / rps
            now = time.monotonic()
            elapsed = now - self._last_call[service]
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
            self._last_call[service] = time.monotonic()


_limiter = _RateLimiter()


def rate_limited(service: str) -> Callable[[F], F]:
    """
    Decorator — enforce per-service rate limiting on async functions.

    Usage::
        @rate_limited("ncbi")
        async def search_pubmed(...): ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await _limiter.acquire(service)
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Retry with Exponential Backoff
# ─────────────────────────────────────────────────────────────────────────────


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Any:
    """
    Decorator — retry async functions on transient network/HTTP errors.

    Uses tenacity with exponential backoff. Does NOT retry on
    ValueError / TypeError (bad user input — fail fast).
    """

    def _is_retryable_http_exception(exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code if exc.response is not None else None
            return status_code in {408, 425, 429} or (
                status_code is not None and 500 <= status_code < 600
            )
        return isinstance(exc, (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError))

    return retry(
        retry=retry_if_exception(_is_retryable_http_exception),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=lambda rs: logger.warning(
            f"Retry attempt {rs.attempt_number}/{max_attempts}: "
            f"{type(rs.outcome.exception()).__name__}: {rs.outcome.exception()}"
        ),
        reraise=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Biological Input Validator
# ─────────────────────────────────────────────────────────────────────────────


class BioValidator:
    """
    Validate and normalise biological identifiers before API calls.

    Raises ValueError with a human-readable message on bad input so
    the dispatcher can return a structured error instead of crashing.
    """

    # ── Identifiers ──────────────────────────────────────────────────────────

    @staticmethod
    def validate_pubmed_id(pmid: str) -> str:
        """Strip 'PMID:' prefix, ensure numeric."""
        pmid = pmid.strip().lstrip("PMID:").strip()
        if not pmid.isdigit():
            raise ValueError(f"Invalid PubMed ID '{pmid}'. Must be numeric (e.g. '37000000').")
        return pmid

    @staticmethod
    def validate_uniprot_accession(accession: str) -> str:
        """Validate UniProt accession format and uppercase it."""
        import re

        acc = accession.strip().upper().split("-")[0]  # strip isoform suffix
        pattern = (
            r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"  # 6-char Swiss-Prot
            r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"  # 6/10-char TrEMBL
        )
        if not re.match(pattern, acc):
            raise ValueError(
                f"Invalid UniProt accession '{accession}'. Expected e.g. 'P04637', 'Q8N6T3'."
            )
        return acc

    @staticmethod
    def validate_gene_symbol(symbol: str) -> str:
        """Normalise gene symbol to HGNC uppercase convention."""
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 25 or not symbol.replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                f"Invalid gene symbol '{symbol}'. "
                "Expected HGNC symbol e.g. 'TP53', 'BRCA1', 'EGFR'."
            )
        return symbol

    @staticmethod
    def validate_sequence(seq: str, seq_type: str = "protein") -> str:
        """Validate amino acid or nucleotide sequence characters."""
        seq = "".join(seq.strip().upper().split())  # strip all whitespace
        if seq_type == "protein":
            valid = set("ACDEFGHIKLMNPQRSTVWY*X-BZU")
        else:
            valid = set("ACGTURYSWKMBDHVN-")

        invalid = set(seq) - valid
        if invalid:
            raise ValueError(
                f"Sequence contains invalid {seq_type} characters: {sorted(invalid)}. "
                f"Preview: {seq[:40]}..."
            )
        if len(seq) < 5:
            raise ValueError(f"Sequence too short ({len(seq)} chars, minimum 5).")
        return seq

    @staticmethod
    def validate_nct_id(nct_id: str) -> str:
        """Validate ClinicalTrials.gov NCT identifier."""
        import re

        nct = nct_id.strip().upper()
        if not re.match(r"^NCT\d{8}$", nct):
            raise ValueError(f"Invalid NCT ID '{nct_id}'. Expected format: NCT12345678.")
        return nct

    @staticmethod
    def validate_chembl_id(chembl_id: str) -> str:
        """Validate ChEMBL compound identifier."""
        import re

        cid = chembl_id.strip().upper()
        if not re.match(r"^CHEMBL\d+$", cid):
            raise ValueError(f"Invalid ChEMBL ID '{chembl_id}'. Expected format: CHEMBL12345.")
        return cid

    @staticmethod
    def validate_kegg_pathway_id(pathway_id: str) -> str:
        """Validate KEGG pathway ID (e.g. 'hsa05200', 'map04010')."""
        import re

        pid = pathway_id.strip().lower()
        if not re.match(r"^[a-z]{2,5}\d{5}$", pid):
            raise ValueError(
                f"Invalid KEGG pathway ID '{pathway_id}'. "
                "Expected format: 'hsa05200' or 'map04010'."
            )
        return pid

    # ── Numeric bounds ────────────────────────────────────────────────────────

    @staticmethod
    def clamp_int(value: int, min_val: int, max_val: int, name: str) -> int:
        """Clamp an integer argument to [min_val, max_val]."""
        if not isinstance(value, int):
            raise TypeError(f"'{name}' must be an integer, got {type(value).__name__}.")
        if value < min_val or value > max_val:
            raise ValueError(f"'{name}' must be between {min_val} and {max_val}, got {value}.")
        return value


# ─────────────────────────────────────────────────────────────────────────────
# MCP Response Formatters
# ─────────────────────────────────────────────────────────────────────────────


def format_success(tool_name: str, data: Any, metadata: dict | None = None) -> str:
    """
    Wrap a tool result in a consistent success envelope.

    Returns compact JSON string ready for MCP TextContent.
    """
    sanitized_data = strip_cache_metadata(data)
    payload: dict[str, Any] = {
        "status": "success",
        "tool": tool_name,
        "data": attach_response_meta(tool_name, sanitized_data),
    }
    if metadata:
        payload["metadata"] = metadata
    try:
        import orjson

        return orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS).decode()
    except ImportError:
        return json.dumps(payload, indent=2, default=str)


def format_error(
    tool_name: str,
    error: Exception,
    context: dict | None = None,
) -> str:
    """
    Wrap an exception in a structured error envelope.

    Includes traceback only for unexpected (non-validation) errors.
    Always logs at ERROR level.
    """
    import traceback

    error_type = type(error).__name__
    message = str(error)

    payload: dict[str, Any] = {
        "status": "error",
        "tool": tool_name,
        "error_type": error_type,
        "message": message,
    }

    if context:
        payload["context"] = context

    if isinstance(error, httpx.HTTPStatusError):
        payload["status_code"] = error.response.status_code
        payload["url"] = str(error.request.url)
    elif isinstance(error, httpx.RequestError) and error.request is not None:
        payload["url"] = str(error.request.url)

    # Include traceback for unexpected errors only
    if not isinstance(error, (ValueError, TypeError, KeyError, LookupError, httpx.HTTPError)):
        payload["traceback"] = traceback.format_exc()

    logger.error(f"[{tool_name}] {error_type}: {message}")
    return json.dumps(payload, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: NCBI params injector
# ─────────────────────────────────────────────────────────────────────────────


def ncbi_params(extra: dict[str, Any]) -> dict[str, Any]:
    """Inject retmode + optional NCBI API key into a params dict."""
    p: dict[str, Any] = {"retmode": "json", **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


_NCBI_SERVICE = "ncbi_key" if NCBI_API_KEY else "ncbi"

__all__ = [
    "get_http_client",
    "close_http_client",
    "cached",
    "get_cache",
    "make_cache_key",
    "strip_cache_metadata",
    "attach_response_meta",
    "rate_limited",
    "with_retry",
    "BioValidator",
    "format_success",
    "format_error",
    "ncbi_params",
    "NCBI_API_KEY",
    "_NCBI_SERVICE",
]
