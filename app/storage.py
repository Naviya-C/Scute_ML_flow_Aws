"""S3 access for the API: presigned uploads into the quarantine prefix only.
Credentials come from the instance/pod IAM role, never from env vars."""
import uuid
from datetime import date
from functools import lru_cache

import boto3
from botocore.config import Config

from .config import settings

EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
EXPIRES_S = 900


@lru_cache
def _s3():
    return boto3.client("s3", region_name=settings.AWS_REGION,
                        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def presign_upload(content_type: str) -> dict:
    """Presigned POST: the browser uploads straight to S3; S3 enforces type and max size."""
    key = f"{settings.S3_INCOMING_PREFIX}{date.today():%Y/%m}/{uuid.uuid4().hex}.{EXT[content_type]}"
    post = _s3().generate_presigned_post(
        settings.S3_INCOMING_BUCKET, key,
        Fields={"Content-Type": content_type},
        Conditions=[{"Content-Type": content_type}, ["content-length-range", 1, MAX_UPLOAD_BYTES]],
        ExpiresIn=EXPIRES_S)
    return {"key": key, "url": post["url"], "fields": post["fields"], "expires_in": EXPIRES_S}
