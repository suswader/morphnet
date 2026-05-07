"""URL noise classification using Brave's adblock engine + EasyPrivacy.

Two interfaces:
  - is_noise_url(url) — full URL check via adblock engine + supplementary list
  - get_noise_domains() — domain set for Playwright route blocking

The adblock engine is built once (lazy), serialized to disk for fast reload.
Falls back to domain-only matching if the engine can't be initialized.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("MORPHNET_CACHE_DIR", Path.home() / ".cache" / "morphnet"))
_ENGINE_CACHE = _CACHE_DIR / "adblock_engine.dat"
_FILTER_LISTS = [
    # EasyPrivacy — trackers, analytics, telemetry
    ("easyprivacy", [
        "https://secure.fanboy.co.nz/easyprivacy.txt",
        "https://easylist.to/easylist/easyprivacy.txt",
    ]),
    # EasyList — ads, ad networks, ad scripts
    ("easylist", [
        "https://secure.fanboy.co.nz/easylist.txt",
        "https://easylist.to/easylist/easylist.txt",
    ]),
]
# Re-download if cache is older than 7 days
_CACHE_TTL_SECS = 7 * 24 * 3600

# ---------------------------------------------------------------------------
# Supplementary noise domains — infra that EasyPrivacy won't block but is
# irrelevant for MCP discovery (config bootstrapping, perf monitoring, CDN).
# ---------------------------------------------------------------------------
_SUPPLEMENTARY_NOISE_DOMAINS = frozenset({
    # Firebase infrastructure
    "firebase.googleapis.com",
    "firebaseinstallations.googleapis.com",
    "firebaseremoteconfig.googleapis.com",
    "firebaselogging.googleapis.com",
    "firebaseperf.googleapis.com",
    "firebasestorage.googleapis.com",
    "fcm.googleapis.com",
    "fcmregistrations.googleapis.com",
    # Font / asset CDNs
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    # Cookie consent
    "cdn.cookielaw.org",
    "geolocation.onetrust.com",
})

# Domains needed by session_manager for Playwright route blocking (network idle).
# This is the union of domains that EasyPrivacy would block plus supplementary.
# Kept as a small set for fast route-match checks — NOT a full filter list.
_ROUTE_BLOCK_DOMAINS = frozenset({
    "google-analytics.com", "googletagmanager.com", "analytics.google.com",
    "segment.io", "segment.com", "cdn.segment.com", "api.segment.io",
    "mixpanel.com", "api.mixpanel.com",
    "hotjar.com", "static.hotjar.com", "script.hotjar.com",
    "facebook.net", "connect.facebook.net",
    "clevertap.com", "clevertap-prod.com",
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adservice.google.com",
    "sentry.io", "browser.sentry-cdn.com",
    "newrelic.com", "bam.nr-data.net",
    "browser-intake-datadoghq.com", "browser-intake-datadoghq.eu",
    "rum.browser-intake-datadoghq.com", "rum-http-intake.logs.datadoghq.eu",
    "fullstory.com", "rs.fullstory.com", "edge.fullstory.com",
    "clarity.ms", "clarity.microsoft.com",
    "amplitude.com", "api.amplitude.com",
    "heap.io", "heapanalytics.com",
    "intercom.io", "widget.intercom.io",
    "crisp.chat",
    "cloudflareinsights.com",
    "lr-ingest.io", "lr-in.com",
    "bugsnag.com",
    "datadog-agent.com",
}) | _SUPPLEMENTARY_NOISE_DOMAINS


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------
_engine = None  # type: ignore
_engine_loaded = False


def _download_filter_lists() -> list[str]:
    """Download EasyPrivacy + EasyList filter lists. Returns list of raw texts."""
    import urllib.request

    results = []
    for name, urls in _FILTER_LISTS:
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MorphNet/1.0"})
                data = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
                if len(data) > 1000:
                    logger.info("Downloaded %s from %s (%d rules)", name, url, data.count("\n"))
                    results.append(data)
                    break  # Got this list, move to next
            except Exception as exc:
                logger.debug("Filter list download failed (%s): %s", url, exc)
    return results


def _build_engine():
    """Build or load the adblock engine. Returns Engine or None."""
    global _engine, _engine_loaded
    if _engine_loaded:
        return _engine

    _engine_loaded = True

    try:
        import adblock
    except ImportError:
        logger.warning("adblock package not installed — falling back to domain-only filtering")
        return None

    # Try loading from cache
    if _ENGINE_CACHE.exists():
        age = time.time() - _ENGINE_CACHE.stat().st_mtime
        if age < _CACHE_TTL_SECS:
            try:
                engine = adblock.Engine(adblock.FilterSet())
                engine.deserialize_from_file(str(_ENGINE_CACHE))
                _engine = engine
                logger.info("Loaded adblock engine from cache (%dKB, %.0fh old)",
                            _ENGINE_CACHE.stat().st_size // 1024, age / 3600)
                return _engine
            except Exception as exc:
                logger.debug("Cache load failed: %s", exc)

    # Download fresh filter lists (EasyPrivacy + EasyList)
    filter_texts = _download_filter_lists()
    if not filter_texts:
        logger.warning("Could not download any filter lists — falling back to domain-only filtering")
        return None

    try:
        fs = adblock.FilterSet()
        for text in filter_texts:
            fs.add_filter_list(text)
        engine = adblock.Engine(fs)

        # Cache to disk
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        engine.serialize_to_file(str(_ENGINE_CACHE))
        logger.info("Built and cached adblock engine (%dKB)", _ENGINE_CACHE.stat().st_size // 1024)

        _engine = engine
        return _engine
    except Exception as exc:
        logger.warning("Failed to build adblock engine: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _domain_matches(hostname: str, domain_set: frozenset[str]) -> bool:
    """Check if hostname or any parent domain is in the set."""
    parts = hostname.split(".")
    for i in range(len(parts)):
        if ".".join(parts[i:]) in domain_set:
            return True
    return False


def is_noise_url(url: str, source_url: str = "https://example.com") -> bool:
    """Check if a URL is noise (analytics, tracking, infra).

    Uses the adblock engine (EasyPrivacy) when available, plus
    supplementary domain checks for MCP-irrelevant infra.
    """
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False

    # Fast path: supplementary domain check
    if _domain_matches(host, _SUPPLEMENTARY_NOISE_DOMAINS):
        return True

    # Full adblock engine check (lazy init)
    engine = _build_engine()
    if engine is not None:
        try:
            result = engine.check_network_urls(url, source_url, "xmlhttprequest")
            return result.matched
        except Exception:
            pass

    # Fallback: domain-only check
    return _domain_matches(host, _ROUTE_BLOCK_DOMAINS)


def get_noise_domains() -> set[str]:
    """Return domain set for Playwright route blocking."""
    return set(_ROUTE_BLOCK_DOMAINS)
