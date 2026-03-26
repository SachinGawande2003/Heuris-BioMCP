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
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
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
_HTTP_LOCK = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """
    Return the module-level shared async HTTP client.
    Thread-safe lazy init with asyncio.Lock.
    Connection pooling is handled by httpx internally.
    """
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        async with _HTTP_LOCK:
            if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
                _HTTP_CLIENT = httpx.AsyncClient(
                    timeout=httpx.Timeout(_HTTP_TIMEOUT, connect=10.0),
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "BioMCP/1.0 (https://github.com/biomcp/biomcp; "
                            "bioinformatics MCP server for Claude)"
                        ),
                        "Accept-Encoding": "gzip, deflate",
                    },
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0,
                    ),
                )
                logger.debug("HTTP client initialized")
    return _HTTP_CLIENT


async def close_http_client() -> None:
    """Gracefully close the shared HTTP client on server shutdown."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None
        logger.debug("HTTP client closed")


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
    "blast": 1_800,  # 30 m
    "reactome": 43_200,  # 12 h
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


def cached(namespace: str, maxsize: int | None = None) -> Callable[[F], F]:
    """
    Decorator — cache async function results in a named TTL cache.

    Usage::
        @cached("pubmed")
        async def search_pubmed(query: str, ...) -> dict: ...
    """

    def decorator(func: F) -> F:
        cache = get_cache(namespace, maxsize)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = make_cache_key(*args, **kwargs)
            if key in cache:
                logger.debug(f"[cache HIT] {namespace}:{key[:8]}")
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            logger.debug(f"[cache SET] {namespace}:{key[:8]}")
            return result

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
    return retry(
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError)
        ),
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
    payload: dict[str, Any] = {
        "status": "success",
        "tool": tool_name,
        "data": data,
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

    # Include traceback for unexpected errors only
    if not isinstance(error, (ValueError, TypeError, KeyError, LookupError)):
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
    "rate_limited",
    "with_retry",
    "BioValidator",
    "format_success",
    "format_error",
    "ncbi_params",
    "NCBI_API_KEY",
    "_NCBI_SERVICE",
]
