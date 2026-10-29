"""initial schema: turtles, jobs, sightings, embeddings, review_queue

Revision ID: 0001
Revises:
Create Date: 2026-10-17
"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = postgresql.TIMESTAMP(timezone=True)
TABLES = ["turtles", "jobs", "sightings", "embeddings", "review_queue"]


def upgrade() -> None:
    op.execute("create extension if not exists vector")
    op.execute("create extension if not exists pgcrypto")
    op.execute("create sequence if not exists turtle_seq")

    op.create_table(
        "turtles",
        sa.Column("id", sa.Text(), primary_key=True,
                  server_default=sa.text("'GT-' || lpad(nextval('turtle_seq')::text, 6, '0')")),
        sa.Column("species", sa.Text(), nullable=False, server_default=sa.text("'green_turtle'")),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'confirmed'")),
        sa.Column("merged_into", sa.Text(), sa.ForeignKey("turtles.id"), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status in ('confirmed', 'provisional')", name="turtles_status_check"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("modal_call_id", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("status in ('queued', 'running', 'completed', 'failed')",
                           name="jobs_status_check"),
        sa.UniqueConstraint("user_id", "idempotency_key", name="jobs_user_idem_key"),
    )
    op.create_index("jobs_created_idx", "jobs", ["created_at"])
    op.create_index("jobs_user_idx", "jobs", ["user_id", "created_at"])

    op.create_table(
        "sightings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("turtle_id", sa.Text(), sa.ForeignKey("turtles.id"), nullable=True),
        sa.Column("species", sa.Text(), nullable=False),
        sa.Column("species_confidence", postgresql.REAL(), nullable=False),
        sa.Column("match_status", sa.Text(), nullable=False),
        sa.Column("match_confidence", postgresql.REAL(), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("sightings_turtle_idx", "sightings", ["turtle_id"])

    op.create_table(
        "embeddings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("turtle_id", sa.Text(), sa.ForeignKey("turtles.id"), nullable=False),
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sightings.id", ondelete="SET NULL"), nullable=True),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("det_crop_key", sa.Text(), nullable=False),
        sa.Column("scute_crop_key", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(768), nullable=False),
        sa.Column("embed_model_version", sa.Text(), nullable=False),
        sa.Column("seg_model_version", sa.Text(), nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("side in ('left_face', 'right_face')", name="embeddings_side_check"),
    )
    op.create_index("embeddings_hnsw_idx", "embeddings", ["embedding"], postgresql_using="hnsw",
                    postgresql_ops={"embedding": "vector_cosine_ops"})
    op.create_index("embeddings_lookup_idx", "embeddings",
                    ["side", "embed_model_version", "seg_model_version"])
    op.create_index("embeddings_turtle_idx", "embeddings", ["turtle_id"])

    op.create_table(
        "review_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sightings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provisional_turtle_id", sa.Text(), sa.ForeignKey("turtles.id"), nullable=False),
        sa.Column("candidates", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("resolution", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", TS, nullable=True),
        sa.CheckConstraint("status in ('pending', 'resolved')", name="review_queue_status_check"),
    )
    op.create_index("review_pending_idx", "review_queue", ["status", "created_at"])

    # Supabase exposes the public schema via its REST API (anon/authenticated keys).
    # RLS with no policies blocks that; our backend connects as postgres, which bypasses RLS.
    for t in TABLES + ["alembic_version"]:
        op.execute(f"alter table {t} enable row level security")


def downgrade() -> None:
    op.execute("alter table alembic_version disable row level security")
    for t in reversed(TABLES):
        op.drop_table(t)
    op.execute("drop sequence if exists turtle_seq")
