"""SQLAlchemy 2.0 schema.

Privacy model
-------------
* Every row that belongs to a person is reachable from ``users.id`` through an FK
  chain, so ``DELETE FROM users WHERE id=...`` (cascading) is a complete erasure.
* Photograph bytes are stored on disk under ``STORE_DIR/<user_id>/<sha256>`` and
  the DB keeps only the digest + metadata. Deleting the row triggers an
  application-level file purge (services.storage.purge_user).
* No field stores a name, email, phone or free-form identifier beyond the
  hashed password. Authentication is a per-installation single-user token.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, LargeBinary, String,
    Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

try:  # pgvector is optional at import time (SQLite test runs)
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except Exception:  # pragma: no cover
    Vector = None  # type: ignore
    HAS_PGVECTOR = False

IMAGE_DIM = 512
TEXT_DIM = 384


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _embedding_type(dim: int):
    if HAS_PGVECTOR and Vector is not None:
        return Vector(dim)
    return JSON  # SQLite fallback: list of floats


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    handle: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    api_token_hash: Mapped[str | None] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    retention_days: Mapped[int] = mapped_column(Integer, default=365)

    profiles: Mapped[list["Profile"]] = relationship(back_populates="user",
                                                     cascade="all, delete-orphan")


class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(Text)
    structured_prefs: Mapped[dict] = mapped_column(JSON, default=dict)
    interests: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now,
                                                 onupdate=_now)
    user: Mapped[User] = relationship(back_populates="profiles")
    photos: Mapped[list["Photograph"]] = relationship(back_populates="profile",
                                                      cascade="all, delete-orphan",
                                                      order_by="Photograph.position")
    prompts: Mapped[list["Prompt"]] = relationship(back_populates="profile",
                                                   cascade="all, delete-orphan")
    __table_args__ = (Index("ix_profiles_user_primary", "user_id", "is_primary"),)


class Photograph(Base):
    __tablename__ = "photographs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    profile_id: Mapped[str] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"),
                                            index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mime: Mapped[str] = mapped_column(String(64), default="image/png")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    profile: Mapped[Profile] = relationship(back_populates="photos")
    embedding: Mapped["Embedding | None"] = relationship(back_populates="photo",
                                                         cascade="all, delete-orphan",
                                                         uselist=False)
    __table_args__ = (UniqueConstraint("profile_id", "position", name="uq_photo_position"),
                      Index("ix_photos_profile_pos", "profile_id", "position"))


class Prompt(Base):
    __tablename__ = "prompts"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    profile_id: Mapped[str] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"),
                                            index=True)
    kind: Mapped[str] = mapped_column(String(32), default="prompt")   # prompt|bio
    question: Mapped[str | None] = mapped_column(String(200))
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0)
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    profile: Mapped[Profile] = relationship(back_populates="prompts")


class Embedding(Base):
    """Vector table. In PostgreSQL the columns are pgvector `vector` types and are
    indexed with HNSW; on SQLite they degrade to JSON arrays (no ANN)."""
    __tablename__ = "embeddings"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    photo_id: Mapped[str | None] = mapped_column(ForeignKey("photographs.id",
                                                            ondelete="CASCADE"), index=True)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("profiles.id",
                                                              ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="image")   # image|text|profile
    encoder: Mapped[str] = mapped_column(String(64), default="unknown")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    vec: Mapped[list] = mapped_column(_embedding_type(IMAGE_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    photo: Mapped[Photograph | None] = relationship(back_populates="embedding")


class Preference(Base):
    __tablename__ = "preferences"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="weights")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("profiles.id",
                                                              ondelete="SET NULL"), index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    metric: Mapped[str] = mapped_column(String(64), default="conversion")
    prior: Mapped[str] = mapped_column(String(32), default="weak_centred")
    min_observations: Mapped[int] = mapped_column(Integer, default=30)
    loss_tolerance: Mapped[float] = mapped_column(Float, default=0.005)
    family_level: Mapped[float] = mapped_column(Float, default=0.05)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String(16), default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    variants: Mapped[list["ExperimentVariant"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan")
    __table_args__ = (Index("ix_experiments_user_status", "user_id", "status"),)


class ExperimentVariant(Base):
    __tablename__ = "experiment_variants"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id",
                                                          ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    missing: Mapped[int] = mapped_column(Integer, default=0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    experiment: Mapped[Experiment] = relationship(back_populates="variants")
    __table_args__ = (UniqueConstraint("experiment_id", "label", name="uq_variant_label"),)


class Observation(Base):
    """One recorded datum. Missing observations are stored with
    observed=False so they can never be counted as failures."""
    __tablename__ = "observations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    variant_id: Mapped[str] = mapped_column(ForeignKey("experiment_variants.id",
                                                        ondelete="CASCADE"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    observed: Mapped[bool] = mapped_column(Boolean, default=True)
    success: Mapped[bool | None] = mapped_column(Boolean)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (Index("ix_observations_variant_time", "variant_id", "observed_at"),)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    candidate_ref: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    terms: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback: Mapped[int] = mapped_column(Integer, default=0)   # -1/+1 preference signal
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(120), default="imported conversation")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    analytics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                                 index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now,
                                                 index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="encoder")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_version"),)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    suite: Mapped[str] = mapped_column(String(64), index=True)
    machine: Mapped[dict] = mapped_column(JSON, default=dict)
    results: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now,
                                                 index=True)
