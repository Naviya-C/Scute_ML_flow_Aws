"""Load test for POST /v1/identify.

For each concurrency level it first uploads N fresh sets of photos (not timed, the worker deletes
incoming/ files after a successful run so keys cannot be reused), then fires all N identify calls
with at most `level` in flight and reports latency and throughput.

Every request uses its own uid (the API limits 10 submissions/min per uid) and the API limits
300 identify calls/min per key, so levels are separated by a pause.

    python loadtest.py --dir /path/to/whole.jpeg,left.jpg,right.jpeg --levels 1:12,4:32,8:48,16:64
Reads API_BASE_URL and API_KEY from the environment (or ../frontend/.env).
"""
import argparse
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "frontend" / ".env")
BASE = os.environ["API_BASE_URL"].rstrip("/")
HEAD = {"X-API-Key": os.environ["API_KEY"]}
SLOTS = ["whole", "left", "right"]
CTYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def upload(path: Path) -> str:
    ctype = CTYPE[path.suffix.lower()]
    s = requests.post(f"{BASE}/v1/uploads", json={"content_type": ctype}, headers=HEAD, timeout=30)
    s.raise_for_status()
    s = s.json()
    r = requests.post(s["url"], data=s["fields"], files={"file": (path.name, path.read_bytes(), ctype)},
                      timeout=120)
    if r.status_code != 204:
        raise RuntimeError(f"S3 {r.status_code}: {r.text[:200]}")
    return s["key"]


def prepare(files: dict[str, Path]) -> dict:
    return {slot: upload(files[slot]) for slot in SLOTS}


def identify(images: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE}/v1/identify", json={"uid": f"load-{uuid.uuid4().hex[:10]}", "images": images},
                          headers={**HEAD, "Idempotency-Key": uuid.uuid4().hex}, timeout=300)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {"code": r.status_code, "status": body.get("status"), "job_id": body.get("job_id"),
                "sec": time.perf_counter() - t0}
    except requests.RequestException as e:
        return {"code": type(e).__name__, "status": None, "job_id": None, "sec": time.perf_counter() - t0}


def server_seconds(job_id: str) -> float | None:
    from datetime import datetime
    try:
        j = requests.get(f"{BASE}/v1/jobs/{job_id}", headers=HEAD, timeout=30).json()
        f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (f(j["updated_at"]) - f(j["created_at"])).total_seconds()
    except Exception:
        return None


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def run_level(files, level: int, n: int):
    print(f"\n=== concurrency {level}, {n} submissions ===", flush=True)
    with ThreadPoolExecutor(16) as ex:
        sets = list(ex.map(lambda _: prepare(files), range(n)))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(level) as ex:
        results = list(ex.map(identify, sets))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["code"] == 200]
    lat = [r["sec"] for r in ok]
    codes = {}
    for r in results:
        codes[r["code"]] = codes.get(r["code"], 0) + 1
    with ThreadPoolExecutor(8) as ex:
        srv = [s for s in ex.map(server_seconds, [r["job_id"] for r in ok if r["job_id"]]) if s is not None]
    print(f"http codes: {codes}   statuses: { {s: sum(1 for r in ok if r['status'] == s) for s in {r['status'] for r in ok}} }")
    if lat:
        print(f"wall {wall:.1f}s   throughput {len(ok) / wall:.2f} submissions/s")
        print(f"client latency  p50 {pct(lat, 50):.2f}s  p95 {pct(lat, 95):.2f}s  max {max(lat):.2f}s")
    if srv:
        print(f"server time     p50 {pct(srv, 50):.2f}s  p95 {pct(srv, 95):.2f}s  max {max(srv):.2f}s  (job created -> finished)")
    return {"level": level, "n": n, "ok": len(ok), "wall": wall, "p50": pct(lat, 50), "p95": pct(lat, 95)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="three files, comma separated: whole,left,right")
    ap.add_argument("--levels", default="1:12,4:32,8:48,16:64", help="concurrency:count pairs")
    ap.add_argument("--pause", type=int, default=45, help="seconds between levels (rate limit window)")
    a = ap.parse_args()
    paths = [Path(p) for p in a.dir.split(",")]
    files = dict(zip(SLOTS, paths))
    rows = []
    for i, pair in enumerate(a.levels.split(",")):
        lvl, n = map(int, pair.split(":"))
        if i:
            time.sleep(a.pause)
        rows.append(run_level(files, lvl, n))
    print("\nsummary\nconc  n   ok   thr/s   p50    p95")
    for r in rows:
        print(f"{r['level']:>4} {r['n']:>3} {r['ok']:>4} {r['ok'] / r['wall']:>7.2f} {r['p50']:>6.2f} {r['p95']:>6.2f}")
