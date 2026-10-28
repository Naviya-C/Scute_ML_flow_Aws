"""Auth (API keys), rate limiting, and HTTP hardening middleware."""
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import settings

log = logging.getLogger("turtle.api")


# ------------------------------------------------------------------ API keys

def key_id(key: str) -> str:
    """Short, non-reversible id for logs and rate limiting (never log the key itself)."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _match(key: str | None, allowed: list[str]) -> bool:
    if not key:
        return False
    return any(hmac.compare_digest(key.encode(), k.encode()) for k in allowed)


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if not _match(x_api_key, settings.API_KEYS + settings.ADMIN_KEYS):
        raise HTTPException(401, "invalid or missing X-API-Key")
    return key_id(x_api_key)


def require_admin(x_admin_key: str | None = Header(default=None)) -> str:
    if not _match(x_admin_key, settings.ADMIN_KEYS):
        raise HTTPException(401, "invalid or missing X-Admin-Key")
    return key_id(x_admin_key)


# ------------------------------------------------------------------ rate limiting

class SlidingWindowLimiter:
    """In-memory, per-instance limiter. With Cloud Run max-instances=N the effective
    limit is up to N x the configured value, which is fine for abuse protection."""

    def __init__(self, window_s: int = 60):
        self.window = window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, bucket: str, limit: int) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.hits[bucket]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= limit:
                retry = int(self.window - (now - q[0])) + 1
                raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry)})
            q.append(now)
            if len(self.hits) > 50_000:  # bound memory
                self.hits.clear()


limiter = SlidingWindowLimiter()


# ------------------------------------------------------------------ middleware

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Cache-Control": "no-store",
}


class HardeningMiddleware(BaseHTTPMiddleware):
    """Request id, body-size limit, security headers, JSON access log, no stack traces out."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.MAX_BODY_BYTES:
            resp = JSONResponse({"detail": "request body too large"}, status_code=413)
        else:
            try:
                resp = await call_next(request)
            except Exception:
                log.exception(json.dumps({"severity": "ERROR", "request_id": rid,
                                          "message": "unhandled error"}))
                resp = JSONResponse({"detail": "internal error", "request_id": rid},
                                    status_code=500)

        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        resp.headers["X-Request-ID"] = rid
        if request.url.path not in ("/", "/health"):
            # One JSON line per request; Cloud Logging parses "severity" automatically.
            print(json.dumps({
                "severity": "WARNING" if resp.status_code >= 400 else "INFO",
                "request_id": rid, "method": request.method, "path": request.url.path,
                "status": resp.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }), flush=True)
        return resp
