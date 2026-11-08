"""Content-addressed photograph storage.

Layout: ``<STORE_DIR>/<user_id>/<sha256[:2]>/<sha256>.<ext>``
Content addressing means duplicate uploads cost nothing and the DB only needs the
digest. Deletion is explicit: ``purge_user`` removes every file for a user, and is
called from the privacy endpoints and from the retention sweep.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

STORE_DIR = Path(os.environ.get("AE_STORE_DIR", Path.cwd() / "data" / "photos"))


def _ext_for(mime: str) -> str:
    return {
        "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
        "image/gif": ".gif", "image/bmp": ".bmp",
    }.get(mime, ".bin")


def store_bytes(user_id: str, data: bytes, mime: str) -> tuple[str, Path, bool]:
    """Return (sha256, path, deduplicated)."""
    digest = hashlib.sha256(data).hexdigest()
    directory = STORE_DIR / user_id / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}{_ext_for(mime)}"
    if path.exists():
        return digest, path, True
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)          # atomic publish
    return digest, path, False


def path_for(user_id: str, digest: str, mime: str) -> Path:
    return STORE_DIR / user_id / digest[:2] / f"{digest}{_ext_for(mime)}"


def read(user_id: str, digest: str, mime: str) -> bytes | None:
    p = path_for(user_id, digest, mime)
    return p.read_bytes() if p.exists() else None


def purge_user(user_id: str) -> int:
    """Delete every stored file for the user. Returns the file count."""
    root = STORE_DIR / user_id
    if not root.exists():
        return 0
    count = sum(1 for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root, ignore_errors=True)
    return count


def disk_usage(user_id: str) -> dict:
    root = STORE_DIR / user_id
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = [p for p in root.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
