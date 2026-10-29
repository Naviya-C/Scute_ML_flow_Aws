"""Hands identify jobs to the GPU worker over HTTP.

This module is the seam between the API and the model service. Phase 1 (docker compose) talks
HTTP to the worker container; later phases swap the body of these three functions (Kubernetes
Service URL, then an SQS enqueue) without touching the routers.

Contract used by the routers:
    start_identify(job_id, images, uid) -> call_id   (returns as soon as the job is accepted)
    wait_identify(call_id, timeout)                  (TimeoutError if still running)
The worker writes status/result into the `jobs` row itself.
"""
import httpx
from fastapi import HTTPException

from .config import settings

_PREFIX = "fc-"


def start_identify(job_id: str, images: list[dict], uid: str) -> str:
    r = httpx.post(f"{settings.WORKER_URL}/jobs", timeout=10,
                   json={"job_id": job_id, "images": images, "uid": uid})
    r.raise_for_status()
    return _PREFIX + job_id


def wait_identify(call_id: str, timeout: int) -> dict:
    job_id = call_id.removeprefix(_PREFIX)
    r = httpx.get(f"{settings.WORKER_URL}/jobs/{job_id}/wait",
                  params={"timeout": timeout}, timeout=timeout + 10)
    if r.status_code == 202:
        raise TimeoutError(job_id)
    r.raise_for_status()
    return r.json()


def start_reindex() -> str:
    raise HTTPException(501, "reindex is not implemented for the self-hosted worker yet")
