from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import settings

# Frontend slot name -> pipeline view. left/right are the face profiles used for re-ID.
SLOT_TO_VIEW = {
    "whole": "whole",
    "left": "left_face",
    "right": "right_face",
    "front": "front",
    "shell": "carapace",
    "rear": "rear",
}


def _check_url(slot: str, url: str):
    u = urlparse(url)
    if u.scheme != "https":
        raise ValueError(f"images.{slot}: url must be https")
    if settings.ALLOWED_IMAGE_HOSTS:
        host = u.hostname or ""
        is_allowed = (
            host in settings.ALLOWED_IMAGE_HOSTS
            or host.endswith(".r2.cloudflarestorage.com")
            or host.endswith(".r2.dev")
        )
        if not is_allowed:
            raise ValueError(f"images.{slot}: host not allowed: {u.hostname}")


class ImagesIn(BaseModel):
    """One R2 link per slot. Empty string or missing = photo not taken."""
    model_config = ConfigDict(extra="forbid")

    whole: str | None = Field(default=None, max_length=2048)
    left: str | None = Field(default=None, max_length=2048)
    right: str | None = Field(default=None, max_length=2048)
    front: str | None = Field(default=None, max_length=2048)
    shell: str | None = Field(default=None, max_length=2048)
    rear: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def check(self):
        given = 0
        for slot in SLOT_TO_VIEW:
            url = (getattr(self, slot) or "").strip()
            setattr(self, slot, url or None)
            if url:
                _check_url(slot, url)
                given += 1
        if given == 0:
            raise ValueError("at least one image is required")
        return self


class IdentifyIn(BaseModel):
    uid: str = Field(min_length=1, max_length=128)
    images: ImagesIn

    def to_pipeline(self) -> list[dict]:
        return [
            {"slot": slot, "view": view, "url": getattr(self.images, slot)}
            for slot, view in SLOT_TO_VIEW.items()
            if getattr(self.images, slot)
        ]


class IdentifyOut(BaseModel):
    success: bool
    job_id: str
    uid: str | None = None
    species: str | None = None
    species_confidence: float | None = None
    turtle_id: str | None = None
    status: str | None = None      # "skipped", "rejected", "no_sea_turtle", "retry_scute", "review", "matched", "new"
    reason: str | None = None      
    can_retry: bool = False        
    confidence: float | None = None


class JobOut(BaseModel):
    job_id: str
    uid: str
    status: Literal["queued", "running", "completed", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ResolveIn(BaseModel):
    action: Literal["confirm_new", "merge"]
    target_turtle_id: str | None = None   # required for merge


class MergeIn(BaseModel):
    source_turtle_id: str
    target_turtle_id: str
