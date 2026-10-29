"""GPU worker HTTP service (phase 1). Models load once at startup; jobs run in a thread pool.

Job state lives in Postgres, so the API can poll it. The in-memory queue is lost on restart
(the API's jobs would stay 'queued'); replacing it with SQS in phase 3 fixes that.
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

import store
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("turtle.worker")

CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "4"))
executor = ThreadPoolExecutor(CONCURRENCY)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = Pipeline()
    yield
    executor.shutdown(wait=True)


app = FastAPI(lifespan=lifespan)


class JobIn(BaseModel):
    job_id: str
    uid: str
    images: list[dict]


def run(job: JobIn):
    try:
        store.set_running(job.job_id)
        state["pipeline"].identify(job.job_id, job.images, job.uid)
    except Exception as e:
        log.exception("job %s failed", job.job_id)
        store.fail_job(job.job_id, f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    if "pipeline" not in state:
        raise HTTPException(503, "models loading")
    return {"ok": True, "device": state["pipeline"].device, "gpu": torch.cuda.is_available()}


@app.post("/jobs", status_code=202)
def submit(job: JobIn):
    if "pipeline" not in state:
        raise HTTPException(503, "models loading")
    executor.submit(run, job)
    return {"accepted": job.job_id}


@app.get("/jobs/{job_id}/wait")
async def wait(job_id: str, timeout: int = 240):
    """Long-poll until the job is done. 202 if it is still running at the deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = await asyncio.to_thread(store.job_row, job_id)
        if row and row[0] == "completed":
            return row[1]
        if row and row[0] == "failed":
            raise HTTPException(500, "job failed")
        await asyncio.sleep(0.5)
    return Response(status_code=202)
