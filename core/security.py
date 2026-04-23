"""
core/security.py
----------------
API key authentication and sliding-window rate limiting for the OSCC Flask app.
"""

import time
from collections import defaultdict, deque
from functools import wraps
from typing import Optional

from flask import request, jsonify

from core.config import settings

_rate_windows: dict[str, deque] = defaultdict(deque)


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit() -> Optional[tuple]:
    """
    Check sliding-window rate limit for the requesting IP.
    Returns a Flask error response tuple if limit exceeded, else None.
    """
    ip = _client_ip()
    window = _rate_windows[ip]
    now = time.monotonic()
    cutoff = now - 60.0

    while window and window[0] < cutoff:
        window.popleft()

    if len(window) >= settings.rate_limit_per_minute:
        return jsonify({
            "error": f"Rate limit exceeded: max {settings.rate_limit_per_minute} requests/minute."
        }), 429

    window.append(now)
    return None


def verify_api_key() -> Optional[tuple]:
    """
    Verify X-API-Key header if auth is enabled.
    Returns a Flask error response tuple if invalid, else None.
    """
    if not settings.auth_enabled:
        return None

    key = request.headers.get("X-API-Key", "").strip()
    if not key:
        return jsonify({"error": "Missing X-API-Key header."}), 401
    if key not in settings.api_keys:
        return jsonify({"error": "Invalid API key."}), 403

    return None


def protected(f):
    """
    Decorator that applies rate limiting + API key auth to a Flask route.
    Apply AFTER @app.route.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        err = check_rate_limit()
        if err:
            return err
        err = verify_api_key()
        if err:
            return err
        return f(*args, **kwargs)
    return wrapper


def rate_limited(f):
    """
    Decorator that applies only rate limiting (no auth) — used for /auth/verify.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        err = check_rate_limit()
        if err:
            return err
        return f(*args, **kwargs)
    return wrapper
