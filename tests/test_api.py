from sqlalchemy import text

from app.db import engine

from .conftest import A, H

IMG = "https://pub-x.r2.dev/u/{}.jpg"
BODY = {"uid": "user-42", "images": {"whole": IMG.format("w"), "left": IMG.format("l"),
                                     "right": IMG.format("r"), "front": "", "shell": IMG.format("s")}}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"ok": True}
    assert client.head("/").status_code == 200
    assert client.head("/health").status_code == 200
    assert client.get("/ready").json() == {"ok": True}


def test_security_headers_and_request_id(client):
    r = client.get("/health", headers={"X-Request-ID": "abc"})
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Request-ID"] == "abc"


def test_auth_required(client):
    assert client.post("/v1/identify", json=BODY).status_code == 401
    assert client.post("/v1/identify", json=BODY, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/admin/review", headers=H).status_code == 401  # API key is not admin


def test_validation(client):
    bad = [
        {"images": {"left": IMG.format("l")}},                                   # no uid
        {"uid": "u", "images": {"left": "", "right": ""}},                       # no images
        {"uid": "u", "images": {"tail": IMG.format("t")}},                       # unknown slot
        {"uid": "u", "images": {"left": "http://pub-x.r2.dev/l.jpg"}},           # not https
        {"uid": "u", "images": {"left": "https://evil.com/l.jpg"}},              # host not allowed
    ]
    for b in bad:
        assert client.post("/v1/identify", json=b, headers=H).status_code == 422, b


def test_body_size_limit(client):
    big = {"uid": "u", "images": {"left": IMG.format("x" * 40_000)}}
    assert client.post("/v1/identify", json=big, headers=H).status_code == 413


def test_identify_flow(client):
    r = client.post("/v1/identify", json=BODY, headers=H)
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert r.json() == {"success": True, "job_id": jid, "uid": "user-42",
                        "species": "green_turtle", "species_confidence": 0.94,
                        "turtle_id": "GT-000042", "status": "matched", "confidence": 0.91,
                        "error": None}
    _, images, uid = client.calls[-1]
    assert uid == "user-42"
    assert [(i["slot"], i["view"]) for i in images] == [
        ("whole", "whole"), ("left", "left_face"), ("right", "right_face"), ("shell", "carapace")]
    job = client.get(f"/v1/jobs/{jid}", headers=H).json()
    assert job["status"] == "completed" and job["uid"] == "user-42"
    assert jid in [j["job_id"] for j in client.get("/v1/users/user-42/jobs", headers=H).json()["items"]]
    assert client.get("/v1/jobs/00000000-0000-0000-0000-000000000000", headers=H).status_code == 404


def test_idempotency(client):
    h = {**H, "Idempotency-Key": "submit-123"}
    a = client.post("/v1/identify", json=BODY, headers=h).json()["job_id"]
    b = client.post("/v1/identify", json=BODY, headers=h).json()["job_id"]
    assert a == b
    assert len(client.calls) == 1  # dispatched once


def test_rate_limit(client):
    body = {**BODY, "uid": "spammer"}
    codes = [client.post("/v1/identify", json=body, headers=H).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_identify_timeout_and_worker_failure(client, monkeypatch):
    from app import dispatch

    def slow(call_id, timeout):
        raise TimeoutError
    monkeypatch.setattr(dispatch, "wait_identify", slow)
    r = client.post("/v1/identify", json={**BODY, "uid": "u-slow"}, headers=H)
    assert r.status_code == 504 and r.json()["success"] is False
    assert client.get(f"/v1/jobs/{r.json()['job_id']}", headers=H).status_code == 200

    def crashed(call_id, timeout):
        with engine.begin() as c:
            c.execute(text("update jobs set status='failed', error='x' where id=:id"),
                      {"id": call_id.removeprefix("fc-")})
        raise RuntimeError("worker crashed")
    monkeypatch.setattr(dispatch, "wait_identify", crashed)
    r = client.post("/v1/identify", json={**BODY, "uid": "u-crash"}, headers=H)
    assert r.status_code == 502
    assert r.json()["success"] is False and r.json()["error"] == "processing failed"


def test_dispatch_failure_marks_job_failed(client, monkeypatch):
    from app import dispatch

    def boom(*a):
        raise RuntimeError("modal down")
    monkeypatch.setattr(dispatch, "start_identify", boom)
    r = client.post("/v1/identify", json={**BODY, "uid": "u-fail"}, headers=H)
    assert r.status_code == 503
    jobs = client.get("/v1/users/u-fail/jobs", headers=H).json()["items"]
    assert jobs[0]["status"] == "failed"


def test_review_and_merge(client):
    with engine.begin() as c:
        c.execute(text("insert into turtles (status) values ('confirmed'), ('confirmed'), ('provisional')"))
        tid = [r[0] for r in c.execute(text("select id from turtles order by id"))]
        job = c.execute(text("insert into jobs (request, user_id) values ('{}', 'u') returning id")).scalar()
        s = c.execute(text("insert into sightings (job_id, turtle_id, species, species_confidence, match_status)"
                           " values (:j, :t, 'green_turtle', .9, 'uncertain') returning id"),
                      {"j": job, "t": tid[2]}).scalar()
        c.execute(text("insert into review_queue (sighting_id, provisional_turtle_id, candidates)"
                       " values (:s, :t, '[]')"), {"s": s, "t": tid[2]})
    items = client.get("/v1/admin/review", headers=A).json()["items"]
    assert len(items) == 1
    r = client.post(f"/v1/admin/review/{items[0]['id']}/resolve", headers=A,
                    json={"action": "merge", "target_turtle_id": tid[0]})
    assert r.json()["turtle_id"] == tid[0]
    assert client.get(f"/v1/turtles/{tid[2]}", headers=H).json() == {"id": tid[2], "merged_into": tid[0]}
    assert client.get(f"/v1/turtles/{tid[0]}", headers=H).json()["sightings_count"] == 1
    assert client.post("/v1/admin/turtles/merge", headers=A,
                       json={"source_turtle_id": tid[1], "target_turtle_id": tid[1]}).status_code == 400
    assert client.post("/v1/admin/turtles/merge", headers=A,
                       json={"source_turtle_id": tid[1], "target_turtle_id": tid[0]}).status_code == 200
    assert client.post("/v1/admin/reindex", headers=A).status_code == 202


def test_production_guards():
    import pytest
    from pydantic import ValidationError

    from app.config import Settings
    with pytest.raises(ValidationError, match="unsafe production config"):
        Settings(ENV="production", DATABASE_URL="postgresql://x", API_KEYS="short",
                 ADMIN_KEYS="", CORS_ORIGINS="*", ALLOWED_IMAGE_HOSTS="")
