"""Optional Redis cache.

Redis is never required: every call site degrades to a no-op on connection
failure and records that fact in ``status()`` so the Settings screen can show it.
Only pure computations are cached (analysis payloads keyed by content digest),
never user-specific authorisation decisions.
"""
from __future__ import annotations

import json
import os
import threading
import time

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
DEFAULT_TTL = int(os.environ.get("AE_CACHE_TTL", "900"))

_state = {"client": None, "checked": False, "error": None, "hits": 0, "misses": 0,
          "lock": threading.Lock()}


def _client():
    if _state["checked"]:
        return _state["client"]
    with _state["lock"]:
        if _state["checked"]:
            return _state["client"]
        _state["checked"] = True
        try:
            import redis

            client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.25,
                                          socket_timeout=0.25)
            client.ping()
            _state["client"] = client
        except Exception as exc:
            _state["error"] = str(exc)
            _state["client"] = None
    return _state["client"]


def get(key: str):
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:
        return None
    if raw is None:
        _state["misses"] += 1
        return None
    _state["hits"] += 1
    try:
        return json.loads(raw)
    except Exception:
        return None


def set(key: str, value, ttl: int | None = None) -> bool:
    client = _client()
    if client is None:
        return False
    try:
        client.setex(key, ttl or DEFAULT_TTL, json.dumps(value, separators=(",", ":")))
        return True
    except Exception:
        return False


def invalidate_prefix(prefix: str) -> int:
    client = _client()
    if client is None:
        return 0
    removed = 0
    try:
        for key in client.scan_iter(match=f"{prefix}*", count=200):
            removed += int(client.delete(key))
    except Exception:
        return 0
    return removed


def status() -> dict:
    client = _client()
    return {
        "enabled": client is not None,
        "url": REDIS_URL if client is not None else None,
        "error": _state["error"],
        "hits": _state["hits"],
        "misses": _state["misses"],
        "note": "Caching is optional; the API works with Redis stopped." if client is None
                else "Cache connected. Analysis payloads are keyed by content SHA-256.",
    }
