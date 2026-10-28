import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .db import engine
from .routers import admin, public
from .security import HardeningMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Sea Turtle ID API",
    version="1.0.0",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

app.add_middleware(HardeningMiddleware)
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Idempotency-Key", "X-Request-ID"],
        max_age=600,
    )


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)  # platform probes (Render's HEAD /)
@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
def health():
    """Liveness: the process is up (no DB call, so it never flaps on DB blips)."""
    return {"ok": True}


@app.get("/ready", include_in_schema=False)
def ready():
    """Readiness: the database answers."""
    with engine.connect() as c:
        c.execute(text("select 1"))
    return {"ok": True}


app.include_router(public.router)
app.include_router(admin.router)
