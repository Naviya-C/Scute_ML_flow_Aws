"""Tests run against a real Postgres with pgvector, e.g.:
    docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres pytest
The schema is rebuilt with Alembic (downgrade base -> upgrade head) for every test session.
"""
import os

import pytest

URL = os.environ.get("TEST_DATABASE_URL")
if not URL:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

API_KEY = "k" * 40
ADMIN_KEY = "a" * 40
os.environ.update(
    DATABASE_URL=URL, MIGRATION_DATABASE_URL=URL, ENV="development",
    API_KEYS=API_KEY, ADMIN_KEYS=ADMIN_KEY, ALLOWED_IMAGE_HOSTS="pub-x.r2.dev",
    CORS_ORIGINS="https://app.example.com", RATE_LIMIT_PER_UID="5",
)


@pytest.fixture(scope="session", autouse=True)
def schema():
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield


RESULT = {"species": {"label": "green_turtle", "confidence": 0.94, "scores": {}},
          "identification": {"status": "matched", "turtle_id": "GT-000042", "confidence": 0.91,
                             "is_provisional": False}}


def fake_worker(call_id, timeout):
    """Stands in for the Modal worker: marks the job completed, like run.process does."""
    import json

    from sqlalchemy import text

    from app.db import engine
    with engine.begin() as c:
        c.execute(text("update jobs set status='completed', result=cast(:r as jsonb) where id=:id"),
                  {"r": json.dumps(RESULT), "id": call_id.removeprefix("fc-")})
    return RESULT


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from app import dispatch
    from app.main import app
    from app.security import limiter

    limiter.hits.clear()
    calls = []
    monkeypatch.setattr(dispatch, "start_identify",
                        lambda job_id, images, uid: calls.append((job_id, images, uid)) or f"fc-{job_id}")
    monkeypatch.setattr(dispatch, "wait_identify", fake_worker)
    monkeypatch.setattr(dispatch, "start_reindex", lambda: "fc-r")
    with TestClient(app) as c:
        c.calls = calls
        yield c


H = {"X-API-Key": API_KEY}
A = {"X-Admin-Key": ADMIN_KEY}
