#!/usr/bin/env python3
"""Reset the installation account password to the development default.

Used by the E2E harness and by developers who have wiped the database. It always
creates the account if missing, so the next login always succeeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from sqlalchemy import select  # noqa: E402

from attraction_engine.db.models import User  # noqa: E402
from attraction_engine.db.session import session_scope  # noqa: E402
from attraction_engine.services import auth  # noqa: E402

password = os.environ.get("AE_OWNER_PASSWORD", "attraction-engine")
with session_scope() as s:
    user = s.scalar(select(User).where(User.handle == "owner"))
    if user is None:
        user = User(handle="owner", password_hash=auth.hash_password(password))
        s.add(user)
        s.flush()
        print(f"created owner {user.id}")
    else:
        user.password_hash = auth.hash_password(password)
        user.api_token_hash = None
        print(f"reset owner {user.id}")
print("password:", password)
