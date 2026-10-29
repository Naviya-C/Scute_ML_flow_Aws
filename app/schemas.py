import re
from datetime import datetime
from typing import Any, Literal

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


KEY_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def _check_key(slot: str, key: str):
    """Images are S3 keys in the quarantine prefix (issued by POST /v1/uploads), never URLs."""
    prefix = settings.S3_INCOMING_PREFIX
    if not key.startswith(prefix) or not KEY_RE.match(key) or ".." in key or "//" in key:
        raise ValueError(f"images.{slot}: invalid key (expected {prefix}...)")


class ImagesIn(BaseModel):
    """One S3 object key per slot. Empty string or missing = photo not taken."""
    model_config = ConfigDict(extra="forbid")

    whole: str | None = Field(default=None, max_length=2048)
    left: str | None = Field(default=None, max_length=2048)
    right: str | None = Field(default=None, max_length=2048)
    front: str | None = Field(default=None, max_length=2048)
    shell: str | None = Field(default=None, max_length=2048)
    rear: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def check(self):
        for slot in SLOT_TO_VIEW:
            key = (getattr(self, slot) or "").strip()
            setattr(self, slot, key or None)
            if key:
                _check_key(slot, key)
        missing = [s for s in settings.REQUIRED_SLOTS if not getattr(self, s, None)]
        if missing:
            raise ValueError("required images missing: " + ", ".join(missing))
        return self


class IdentifyIn(BaseModel):
    uid: str = Field(min_length=1, max_length=128)
    images: ImagesIn

    def to_pipeline(self) -> list[dict]:
        return [
            {"slot": slot, "view": view, "key": getattr(self.images, slot)}
            for slot, view in SLOT_TO_VIEW.items()
            if getattr(self.images, slot)
        ]


class UploadIn(BaseModel):
    content_type: Literal["image/jpeg", "image/png", "image/webp"]


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
