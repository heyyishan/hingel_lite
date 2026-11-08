"""Engine/session management.

PostgreSQL is the primary store. When ``DATABASE_URL`` is unset the application
falls back to a SQLite file so a fresh clone runs immediately; the API reports
the backend and disables pgvector-specific features in that mode.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_SQLITE = "sqlite:///" + str(Path.cwd() / "attraction_engine.db")
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_SQLITE)
IS_POSTGRES = DATABASE_URL.startswith("postgresql")

_engine_kwargs: dict = {"future": True, "pool_pre_ping": True}
if IS_POSTGRES:
    _engine_kwargs.update(pool_size=5, max_overflow=5, pool_recycle=1800)
else:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)

if not IS_POSTGRES:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def backend_info() -> dict:
    info = {"url_scheme": DATABASE_URL.split(":", 1)[0], "is_postgres": IS_POSTGRES}
    try:
        with engine.connect() as conn:
            if IS_POSTGRES:
                row = conn.execute(text("select version()")).scalar()
                info["version"] = str(row).split(" on ")[0]
                info["pgvector"] = conn.execute(
                    text("select extversion from pg_extension where extname='vector'")
                ).scalar()
            else:
                info["version"] = conn.execute(text("select sqlite_version()")).scalar()
                info["pgvector"] = None
    except Exception as exc:
        info["error"] = str(exc)
    return info
