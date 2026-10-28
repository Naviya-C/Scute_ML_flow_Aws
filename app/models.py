"""SQLAlchemy models: the source of truth for the database schema.

Change the schema by editing this file, then:
    alembic revision --autogenerate -m "describe change"   # review the generated file!
    alembic upgrade head

The Modal worker writes to these same tables with plain SQL (worker/pipeline/store.py),
so keep column names in sync when you rename anything.
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, CheckConstraint, ForeignKey, Index, Sequence, Text, UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB, REAL, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 768  # DINOv2 ViT-B/14; must match worker EMBED_DIM


class Base(DeclarativeBase):
    pass


turtle_seq = Sequence("turtle_seq", metadata=Base.metadata)

TS = TIMESTAMP(timezone=True)


class Turtle(Base):
    __tablename__ = "turtles"

    id: Mapped[str] = mapped_column(
        Text, primary_key=True,
        server_default=text("'GT-' || lpad(nextval('turtle_seq')::text, 6, '0')"),
    )
    species: Mapped[str] = mapped_column(Text, server_default=text("'green_turtle'"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'confirmed'"))
    merged_into: Mapped[str | None] = mapped_column(Text, ForeignKey("turtles.id"))
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())

    __table_args__ = (
        CheckConstraint("status in ('confirmed', 'provisional')", name="turtles_status_check"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          server_default=text("gen_random_uuid()"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'queued'"))
    request: Mapped[dict] = mapped_column(JSONB)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    modal_call_id: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(Text)              # frontend "uid"
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())

    __table_args__ = (
        CheckConstraint("status in ('queued', 'running', 'completed', 'failed')",
                        name="jobs_status_check"),
        UniqueConstraint("user_id", "idempotency_key", name="jobs_user_idem_key"),
        Index("jobs_created_idx", "created_at"),
        Index("jobs_user_idx", "user_id", "created_at"),
    )


class Sighting(Base):
    __tablename__ = "sightings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          server_default=text("gen_random_uuid()"))
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),
                                              ForeignKey("jobs.id", ondelete="CASCADE"))
    turtle_id: Mapped[str | None] = mapped_column(Text, ForeignKey("turtles.id"))
    species: Mapped[str] = mapped_column(Text)
    species_confidence: Mapped[float] = mapped_column(REAL)
    match_status: Mapped[str] = mapped_column(Text)  # matched|new|uncertain|skipped|insufficient_views
    match_confidence: Mapped[float | None] = mapped_column(REAL)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())

    __table_args__ = (Index("sightings_turtle_idx", "turtle_id"),)


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    turtle_id: Mapped[str] = mapped_column(Text, ForeignKey("turtles.id"))
    sighting_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sightings.id", ondelete="SET NULL"))
    side: Mapped[str] = mapped_column(Text)
    det_crop_key: Mapped[str] = mapped_column(Text)    # storage key of detector crop (re-index source)
    scute_crop_key: Mapped[str] = mapped_column(Text)  # storage key of scute crop (LightGlue)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    embed_model_version: Mapped[str] = mapped_column(Text)
    seg_model_version: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())

    __table_args__ = (
        CheckConstraint("side in ('left_face', 'right_face')", name="embeddings_side_check"),
        Index("embeddings_hnsw_idx", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("embeddings_lookup_idx", "side", "embed_model_version", "seg_model_version"),
        Index("embeddings_turtle_idx", "turtle_id"),
    )


class ReviewItem(Base):
    __tablename__ = "review_queue"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          server_default=text("gen_random_uuid()"))
    sighting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sightings.id", ondelete="CASCADE"))
    provisional_turtle_id: Mapped[str] = mapped_column(Text, ForeignKey("turtles.id"))
    candidates: Mapped[list] = mapped_column(JSONB)   # [{turtle_id, inliers, sides}]
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    resolution: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(TS)

    __table_args__ = (
        CheckConstraint("status in ('pending', 'resolved')", name="review_queue_status_check"),
        Index("review_pending_idx", "status", "created_at"),
    )


# Tables exposed by Supabase's REST API get RLS (see the initial migration).
RLS_TABLES = [t.name for t in Base.metadata.sorted_tables] + ["alembic_version"]
