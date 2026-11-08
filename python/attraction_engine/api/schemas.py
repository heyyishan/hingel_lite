"""Pydantic request/response models. Validation lives here so routers stay thin."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------- auth
class LoginRequest(BaseModel):
    handle: str = Field(default="owner", min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: float
    user_id: str
    handle: str


# ------------------------------------------------------------------ profiles
class PromptIn(BaseModel):
    kind: Literal["prompt", "bio", "caption"] = "prompt"
    question: str | None = Field(default=None, max_length=200)
    answer: str = Field(min_length=1, max_length=4000)
    position: int = Field(default=0, ge=0, le=64)


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=8000)
    is_primary: bool = False
    structured_prefs: dict[str, Any] = Field(default_factory=dict)
    interests: list[str] = Field(default_factory=list, max_length=64)
    prompts: list[PromptIn] = Field(default_factory=list, max_length=16)

    @field_validator("interests")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        return [i.strip() for i in v if i and i.strip()]


class ProfileOut(ORMModel):
    id: str
    name: str
    description: str | None
    is_primary: bool
    structured_prefs: dict[str, Any]
    interests: list[str]
    created_at: datetime | None = None
    photo_count: int = 0
    prompt_count: int = 0


class PromptOut(ORMModel):
    id: str
    kind: str
    question: str | None
    answer: str
    position: int


# ------------------------------------------------------------------- photos
class PhotoOut(ORMModel):
    id: str
    position: int
    width: int
    height: int
    bytes: int
    sha256: str
    mime: str
    quality: float


class PhotoOrderIn(BaseModel):
    photo_ids: list[str] = Field(min_length=1)


# -------------------------------------------------------------- experiments
class VariantIn(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    configuration: dict[str, Any] = Field(default_factory=dict)
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)
    weight: float = Field(default=1.0, gt=0, le=10)


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    hypothesis: str | None = Field(default=None, max_length=2000)
    profile_id: str | None = None
    metric: str = Field(default="conversion", max_length=64)
    prior: Literal["uniform", "jeffreys", "weak_centred", "empirical"] = "weak_centred"
    min_observations: int = Field(default=30, ge=1, le=100000)
    loss_tolerance: float = Field(default=0.005, gt=0, le=0.5)
    family_level: float = Field(default=0.05, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    variants: list[VariantIn] = Field(min_length=2, max_length=8)


class ObservationIn(BaseModel):
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)
    observed_at: datetime | None = None
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _at_least_one(self) -> "ObservationIn":
        if self.successes + self.failures + self.missing == 0:
            raise ValueError("observation must record at least one datum")
        return self


class ExperimentOut(ORMModel):
    id: str
    name: str
    hypothesis: str | None
    metric: str
    prior: str
    min_observations: int
    loss_tolerance: float
    family_level: float
    seed: int
    status: str
    created_at: datetime | None
    variants: list[dict[str, Any]]


# --------------------------------------------------------------- analytics
class ConversationMessageIn(BaseModel):
    sender: str = Field(min_length=1, max_length=120)
    timestamp: datetime
    text: str = Field(min_length=0, max_length=8000)


class ConversationIn(BaseModel):
    label: str = Field(default="imported conversation", max_length=120)
    me: str = Field(default="me", max_length=120)
    messages: list[ConversationMessageIn] = Field(min_length=2, max_length=20000)


# ----------------------------------------------------------- counterfactual
class CounterfactualIn(BaseModel):
    profile_id: str
    baseline_photo_ids: list[str] = Field(min_length=1)
    modified_photo_ids: list[str] = Field(min_length=1)
    # A None entry means "keep the profile's current text at this index".
    baseline_prompts: list[str | None] | None = None
    modified_prompts: list[str | None] | None = None
    seed: int = 42


# ------------------------------------------------------------- optimizer
class OptimizeRequest(BaseModel):
    profile_id: str
    lambda_: float = Field(default=0.35, ge=0, le=1, alias="lambda")
    mu: float = Field(default=0.20, ge=0, le=1)
    seed: int = 42
    strategies: list[Literal["greedy", "beam", "simulated_annealing", "random_restart",
                             "exhaustive"]] | None = None
    beam_width: int = Field(default=8, ge=1, le=64)
    sa_iterations: int = Field(default=20000, ge=100, le=500000)

    model_config = ConfigDict(populate_by_name=True)


# -------------------------------------------------------------- retrieval
class RetrievalRequest(BaseModel):
    profile_id: str | None = None
    query_text: str | None = Field(default=None, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=200)
    use_ann: bool = True
    ef_search: int = Field(default=64, ge=1, le=1000)


class BenchmarkRequest(BaseModel):
    sizes: list[int] = Field(default_factory=lambda: [128, 512, 2048], max_length=8)
    dim: int = Field(default=384, ge=8, le=2048)
    k: int = Field(default=10, ge=1, le=500)
    repeats: int = Field(default=5, ge=1, le=50)
    strategies: list[Literal["sort_full", "partial_sort", "nth_element",
                             "heap_bounded"]] | None = None
    threads: list[int] = Field(default_factory=lambda: [1, 2], max_length=8)

    @field_validator("sizes")
    @classmethod
    def _cap(cls, v: list[int]) -> list[int]:
        caps = [s for s in v if 1 <= s <= 200000]
        if not caps:
            raise ValueError("no valid sizes")
        return caps
