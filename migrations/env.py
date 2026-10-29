"""Alembic environment.

URL: MIGRATION_DATABASE_URL, falling back to DATABASE_URL.
For Supabase use the *session* pooler (port 5432) for migrations: DDL and long transactions
are safest there. The app itself uses the transaction pooler (port 6543).
"""
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    url = os.environ.get("MIGRATION_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set MIGRATION_DATABASE_URL (or DATABASE_URL) to run migrations")
    for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def include_object(obj, name, type_, reflected, compare_to):
    # Only manage objects defined in our models. Never touch anything else that lives in the
    # same database (Supabase internals, other apps' tables, alembic_version).
    if type_ == "table" and reflected and compare_to is None:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      include_object=include_object, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool,
                           connect_args={"prepare_threshold": None})
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          include_object=include_object, compare_type=True,
                          transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
