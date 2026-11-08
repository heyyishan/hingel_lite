"""Shared FastAPI dependencies: auth, rate limiting, error envelope."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..services import auth

_bearer = HTTPBearer(auto_error=False)

# Fixed-window token bucket per (ip, route class). In-process, which is correct
# for the single-process deployment this product targets; with multiple workers
# the limit becomes per-worker and is documented as such.
_RATE_LIMIT = int(os.environ.get("AE_RATE_LIMIT_PER_MIN", "240"))
_buckets: dict[str, deque[float]] = defaultdict(deque)


def rate_limiter(request: Request) -> None:
    if _RATE_LIMIT <= 0:
        return
    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path.split('/')[1] if '/' in request.url.path else 'root'}"
    now = time.monotonic()
    bucket = _buckets[key]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="rate limit exceeded; retry in a minute",
                            headers={"Retry-After": "60"})
    bucket.append(now)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    user_id = auth.resolve(creds.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return user_id
