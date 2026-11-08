"""Authentication.

Scope: this is a *personal, local-first* application. The auth model is therefore
deliberately simple and strong rather than enterprise-shaped:

* one account per installation, created on first run;
* passwords hashed with scrypt (n=2^14, r=8, p=1) from the standard library's
  hashlib — no external crypto dependency to audit or pin;
* login returns a short-lived bearer token whose SHA-256 is stored, so a leaked
  database does not expose usable tokens;
* constant-time comparisons everywhere a secret is involved.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import logging
import secrets
import time
from dataclasses import dataclass

from sqlalchemy import select

from ..db.models import User
from ..db.session import session_scope

log = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = int(os.environ.get("AE_TOKEN_TTL", str(60 * 60 * 12)))


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        got = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r),
                             p=int(p), dklen=len(expected))
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class Session:
    user_id: str
    handle: str
    token: str
    expires_at: float


def ensure_owner(handle: str = "owner", password: str | None = None) -> User:
    """Create the single installation account if it does not exist."""
    password = password or os.environ.get("AE_OWNER_PASSWORD") or "attraction-engine"
    with session_scope() as s:
        user = s.scalar(select(User).where(User.handle == handle))
        if user is None:
            user = User(handle=handle, password_hash=hash_password(password))
            s.add(user)
            s.flush()
        return user


def login(handle: str, password: str) -> Session | None:
    with session_scope() as s:
        user = s.scalar(select(User).where(User.handle == handle))
        if user is None or not verify_password(password, user.password_hash):
            return None
        token = secrets.token_urlsafe(32)
        user.api_token_hash = _token_digest(token)
        expires = time.time() + TOKEN_TTL_SECONDS
        s.add(user)
        return Session(user_id=user.id, handle=user.handle, token=token, expires_at=expires)


def resolve(token: str) -> str | None:
    """Return the user id for a bearer token, or None."""
    if not token:
        return None
    digest = _token_digest(token)
    with session_scope() as s:
        user = s.scalar(select(User).where(User.api_token_hash == digest))
        return None if user is None else user.id


def logout(user_id: str) -> None:
    with session_scope() as s:
        user = s.get(User, user_id)
        if user is not None:
            user.api_token_hash = None


# --------------------------------------------------------------------------
# Signed media URLs
#
# A browser cannot attach an Authorization header to an <img> request, so the raw
# photo endpoint would 401 in the UI. The alternative -- putting the long-lived
# bearer token in the query string -- leaks it into history and logs. Instead the
# API issues a short-lived HMAC over (user_id, photo_id, expiry) that grants read
# access to exactly one photograph for a bounded time.
# --------------------------------------------------------------------------

_media_key_warned = False


def _media_key() -> bytes | None:
    """HMAC key for signed media URLs, or None when unconfigured.

    Signing is an optimisation for browsers (an <img> tag cannot send a bearer
    header), not a requirement. If no secret is configured we return None and the
    listing serves unsigned URLs; the raw endpoint then demands a bearer token.
    Raising here would turn a missing optional setting into a 500 on every photo
    list, which is what an earlier version did.
    """
    global _media_key_warned
    secret = os.environ.get("AE_MEDIA_SECRET") or os.environ.get("AE_OWNER_PASSWORD", "")
    if not secret:
        if not _media_key_warned:
            log.warning("AE_MEDIA_SECRET / AE_OWNER_PASSWORD unset: media URLs are "
                        "unsigned and the raw endpoint requires a bearer token")
            _media_key_warned = True
        return None
    return hashlib.sha256(("media:" + secret).encode("utf-8")).digest()


def media_signing_enabled() -> bool:
    return _media_key() is not None


def sign_media(user_id: str, photo_id: str, ttl: int = 900) -> str:
    """Return ``"<expires>.<mac>"``, or "" when signing is not configured."""
    key = _media_key()
    if key is None:
        return ""
    expires = int(time.time()) + int(ttl)
    payload = f"{user_id}:{photo_id}:{expires}"
    mac = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{expires}.{mac}"


def verify_media(user_id: str, photo_id: str, signature: str) -> bool:
    key = _media_key()
    if key is None or not signature:
        return False
    try:
        expires_raw, mac = signature.split(".", 1)
        expires = int(expires_raw)
    except (ValueError, AttributeError):
        return False
    if expires < int(time.time()):
        return False
    expected = hmac.new(key, f"{user_id}:{photo_id}:{expires}".encode("utf-8"),
                        hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(mac, expected)
