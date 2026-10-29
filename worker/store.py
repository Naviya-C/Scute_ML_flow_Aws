"""Plain-SQL access to the tables defined in app/models.py (keep column names in sync)."""
import json
import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb


def connect() -> psycopg.Connection:
    """One short-lived connection per job; used as a context manager it commits on success."""
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def nearest(conn, vec, side: str, embed_version: str):
    return conn.execute(
        """select e.turtle_id, e.embedding <=> %s as dist
             from embeddings e join turtles t on t.id = e.turtle_id
            where e.side = %s and e.embed_model_version = %s and t.merged_into is null
            order by dist limit 1""",
        (vec, side, embed_version)).fetchone()


def new_turtle(conn, species: str, provisional: bool) -> str:
    return conn.execute(
        "insert into turtles (species, status) values (%s, %s) returning id",
        (species, "provisional" if provisional else "confirmed")).fetchone()[0]


def save_sighting(conn, job_id: str, turtle_id, species: dict, match_status: str, conf):
    return conn.execute(
        """insert into sightings (job_id, turtle_id, species, species_confidence,
                                  match_status, match_confidence)
           values (%s, %s, %s, %s, %s, %s) returning id""",
        (job_id, turtle_id, species["label"], species["confidence"], match_status, conf)
    ).fetchone()[0]


def add_embeddings(conn, turtle_id, sighting_id, embeds: dict, embed_version, seg_version):
    # det/scute crop keys point at the source S3 key until the crop stages write their own crops.
    for side, (vec, key) in embeds.items():
        conn.execute(
            """insert into embeddings (turtle_id, sighting_id, side, det_crop_key, scute_crop_key,
                                       embedding, embed_model_version, seg_model_version)
               values (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (turtle_id, sighting_id, side, key, key, vec, embed_version, seg_version))


def add_review(conn, sighting_id, provisional_id: str, candidates: list):
    conn.execute(
        "insert into review_queue (sighting_id, provisional_turtle_id, candidates) values (%s, %s, %s)",
        (sighting_id, provisional_id, Jsonb(candidates)))


def set_running(job_id: str):
    with connect() as conn:
        conn.execute("update jobs set status='running', updated_at=now() where id=%s", (job_id,))


def finish_job(conn, job_id: str, result: dict):
    conn.execute("update jobs set status='completed', result=%s, updated_at=now() where id=%s",
                 (Jsonb(result), job_id))


def fail_job(job_id: str, error: str):
    with connect() as conn:
        conn.execute("update jobs set status='failed', error=%s, updated_at=now() where id=%s",
                     (error[:500], job_id))


def job_row(job_id: str):
    with connect() as conn:
        return conn.execute("select status, result from jobs where id=%s", (job_id,)).fetchone()
