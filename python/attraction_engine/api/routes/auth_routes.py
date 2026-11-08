"""Authentication endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...services import auth
from ..deps import current_user, rate_limiter
from ..schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(rate_limiter)])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    session = auth.login(body.handle, body.password)
    if session is None:
        raise HTTPException(status_code=401, detail="invalid handle or password")
    return TokenResponse(token=session.token, expires_at=session.expires_at,
                         user_id=session.user_id, handle=session.handle)


@router.post("/logout", status_code=204)
def logout(user_id: str = Depends(current_user)):
    auth.logout(user_id)


@router.get("/me")
def me(user_id: str = Depends(current_user)):
    return {"user_id": user_id}


@router.get("/bootstrap")
def bootstrap():
    """Whether the installation has been initialised, and the default handle.

    Exposed without auth so the login screen can render the first-run hint. The
    password is never returned.
    """
    default_handle = "owner"
    has_env_password = bool(__import__("os").environ.get("AE_OWNER_PASSWORD"))
    return {"installation_handle": default_handle,
            "default_password_in_use": not has_env_password,
            "action": "Log in with handle 'owner'. If you have not set "
                      "AE_OWNER_PASSWORD, the development default is "
                      "'attraction-engine' — change it in Settings before storing "
                      "anything private." if not has_env_password else
                      "Log in with handle 'owner' and the AE_OWNER_PASSWORD you set."}
