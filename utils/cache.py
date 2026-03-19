"""
Utility: disk-based cache with TTL for API responses.
"""

import os
import pickle
import time
import hashlib
from config.settings import CACHE_DIR


def _get_cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.pkl")


def cache_get(key: str, ttl: int):
    """Return cached value if fresh, else None."""
    path = _get_cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            entry = pickle.load(f)
        if time.time() - entry["ts"] > ttl:
            return None
        return entry["data"]
    except Exception:
        return None


def cache_set(key: str, data):
    """Store value in cache with current timestamp."""
    path = _get_cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump({"ts": time.time(), "data": data}, f)
    except Exception:
        pass


def cache_clear():
    """Clear all cache files."""
    if not os.path.exists(CACHE_DIR):
        return
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass
