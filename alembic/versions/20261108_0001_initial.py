"""initial schema: profiles, photographs, prompts, embeddings, experiments

Revision ID: 0001
Revises:
Create Date: 2026-11-08

The pgvector columns and HNSW indexes are created with raw DDL guarded by the
dialect, so the same migration applies to SQLite (JSON embeddings, no ANN) and to
PostgreSQL (vector(512) + HNSW cosine index).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

IMAGE_DIM = 512


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    is_pg = dialect == "postgresql"

    if is_pg:
        # Extensions need superuser on older clusters; attempt, and fall back to
        # a non-vector schema when the deployment cannot install pgvector.
        for ext in ("vector", "pg_trgm"):
            try:
                with bind.begin_nested():
                    op.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            except Exception:
                pass
        has_vector = bool(bind.execute(sa.text(
            "select 1 from pg_extension where extname = 'vector'")).scalar())
        if not has_vector:
            is_pg = False  # degrade to JSON embeddings for this database

    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("handle", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("api_token_hash", sa.String(128), unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("retention_days", sa.Integer, server_default="365"),
    )
    op.create_table(
        "profiles",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("is_primary", sa.Boolean, server_default=sa.false()),
        sa.Column("description", sa.Text),
        sa.Column("structured_prefs", sa.JSON, server_default="{}"),
        sa.Column("interests", sa.JSON, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_profiles_user_primary", "profiles", ["user_id", "is_primary"])

    op.create_table(
        "photographs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("profile_id", sa.String(32), sa.ForeignKey("profiles.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("position", sa.Integer, server_default="0"),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("mime", sa.String(64)),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("bytes", sa.Integer),
        sa.Column("analysis", sa.JSON, server_default="{}"),
        sa.Column("quality", sa.Float, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("profile_id", "position", name="uq_photo_position"),
    )
    op.create_index("ix_photographs_profile_id", "photographs", ["profile_id"])
    op.create_index("ix_photographs_sha256", "photographs", ["sha256"])
    op.create_index("ix_photos_profile_pos", "photographs", ["profile_id", "position"])

    op.create_table(
        "prompts",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("profile_id", sa.String(32), sa.ForeignKey("profiles.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("kind", sa.String(32), server_default="prompt"),
        sa.Column("question", sa.String(200)),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, server_default="0"),
        sa.Column("analysis", sa.JSON, server_default="{}"),
    )
    op.create_index("ix_prompts_profile_id", "prompts", ["profile_id"])

    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("photo_id", sa.String(32), sa.ForeignKey("photographs.id", ondelete="CASCADE")),
        sa.Column("profile_id", sa.String(32), sa.ForeignKey("profiles.id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(16), server_default="image"),
        sa.Column("encoder", sa.String(64)),
        sa.Column("dim", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_embeddings_photo_id", "embeddings", ["photo_id"])
    op.create_index("ix_embeddings_profile_id", "embeddings", ["profile_id"])

    if is_pg:
        # pgvector column + HNSW index (cosine distance) for ANN retrieval.
        # vector(512) matches the CLIP image space; text rows use the same
        # column but only 384 dims and are filtered by `kind`.
        op.execute(f"ALTER TABLE embeddings ADD COLUMN vec vector({IMAGE_DIM})")
        op.execute(
            "CREATE INDEX ix_embeddings_vec_hnsw ON embeddings "
            "USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        # `interests`/`structured_prefs` are JSON columns; a GIN index on JSON needs
        # jsonb. Convert to jsonb so containment queries are indexable.
        op.execute("ALTER TABLE profiles ALTER COLUMN interests TYPE jsonb USING interests::jsonb")
        op.execute("ALTER TABLE profiles ALTER COLUMN structured_prefs TYPE jsonb "
                   "USING structured_prefs::jsonb")
        op.execute("CREATE INDEX ix_profiles_interests_gin ON profiles USING gin (interests)")
        op.execute("CREATE INDEX ix_profiles_name_trgm ON profiles USING gin (name gin_trgm_ops)")
    else:
        op.add_column("embeddings", sa.Column("vec", sa.JSON, server_default="[]"))

    op.create_table(
        "preferences",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("kind", sa.String(32), server_default="weights"),
        sa.Column("payload", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_preferences_user_id", "preferences", ["user_id"])

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("profile_id", sa.String(32), sa.ForeignKey("profiles.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("hypothesis", sa.Text),
        sa.Column("metric", sa.String(64), server_default="conversion"),
        sa.Column("prior", sa.String(32), server_default="weak_centred"),
        sa.Column("min_observations", sa.Integer, server_default="30"),
        sa.Column("loss_tolerance", sa.Float, server_default="0.005"),
        sa.Column("family_level", sa.Float, server_default="0.05"),
        sa.Column("seed", sa.Integer, server_default="42"),
        sa.Column("status", sa.String(16), server_default="running"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_experiments_user_id", "experiments", ["user_id"])
    op.create_index("ix_experiments_user_status", "experiments", ["user_id", "status"])

    op.create_table(
        "experiment_variants",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("experiment_id", sa.String(32),
                  sa.ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("configuration", sa.JSON, server_default="{}"),
        sa.Column("successes", sa.Integer, server_default="0"),
        sa.Column("failures", sa.Integer, server_default="0"),
        sa.Column("missing", sa.Integer, server_default="0"),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.UniqueConstraint("experiment_id", "label", name="uq_variant_label"),
        sa.CheckConstraint("successes >= 0 AND failures >= 0", name="ck_variant_counts"),
    )
    op.create_index("ix_experiment_variants_experiment_id", "experiment_variants",
                    ["experiment_id"])

    op.create_table(
        "observations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("variant_id", sa.String(32),
                  sa.ForeignKey("experiment_variants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True)),
        sa.Column("observed", sa.Boolean, server_default=sa.true()),
        sa.Column("success", sa.Boolean),
        sa.Column("context", sa.JSON, server_default="{}"),
    )
    op.create_index("ix_observations_variant_id", "observations", ["variant_id"])
    op.create_index("ix_observations_variant_time", "observations",
                    ["variant_id", "observed_at"])

    op.create_table(
        "matches",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("candidate_ref", sa.String(64)),
        sa.Column("score", sa.Float, server_default="0"),
        sa.Column("terms", sa.JSON, server_default="{}"),
        sa.Column("feedback", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_matches_user_id", "matches", ["user_id"])
    op.create_index("ix_matches_candidate_ref", "matches", ["candidate_ref"])

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("label", sa.String(120)),
        sa.Column("message_count", sa.Integer, server_default="0"),
        sa.Column("analytics", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    op.create_table(
        "analytics_events",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(64)),
        sa.Column("payload", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_analytics_events_kind", "analytics_events", ["kind"])
    op.create_index("ix_analytics_events_created_at", "analytics_events", ["created_at"])
    if is_pg:
        # Append-only log: BRIN is dramatically smaller than btree for
        # time-ordered data while still supporting range pruning.
        op.execute("CREATE INDEX ix_analytics_events_created_brin ON analytics_events "
                   "USING brin (created_at)")

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), server_default="encoder"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("name", "version", name="uq_model_version"),
    )

    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("suite", sa.String(64)),
        sa.Column("machine", sa.JSON, server_default="{}"),
        sa.Column("results", sa.JSON, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_benchmark_runs_suite", "benchmark_runs", ["suite"])
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])


def downgrade() -> None:
    for t in ["benchmark_runs", "model_versions", "analytics_events", "conversations",
              "matches", "observations", "experiment_variants", "experiments",
              "preferences", "embeddings", "prompts", "photographs", "profiles", "users"]:
        op.drop_table(t)
