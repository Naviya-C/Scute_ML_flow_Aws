"""The ML pipeline. One job = one user's submission (whole, left, right + optional photos).

    1 decode + quality -> 2 detect -> 3 BioCLIP (every photo) -> leatherback? -> 4 segment
    -> 5 DINOv2 embed -> pgvector match -> 6 decide

Rules:
  * A required photo (REQUIRED_SLOTS) that fails, or ANY photo that is not a sea turtle,
    rejects the whole request: nothing is kept, all photos are deleted from incoming/.
  * An optional photo that fails the quality check is dropped; one the detector misses is used
    as the full image. Neither stops the request.
  * Leatherbacks have no scutes: log the sighting with the species and stop (status "skipped").

Every stage takes the whole submission as a batch, so the GPU sees one forward pass per stage per
job and the GPU lock is taken once per stage. Batching across different users' jobs is the next
step (a micro-batcher in front of the stages).

Transport-agnostic: main.py (HTTP, phase 1) and later an SQS consumer both call identify().

What is real vs placeholder:
  * gate (gate.py)     REAL   file checks + one BioCLIP pass: turtle or not, and species.
  * embed_batch()      REAL   DINOv2 ViT-B/14 (768-dim), the model the DB schema was built for.
  * detect()           STUB   returns the images unchanged (never "not detected"); plug in your detector.
  * segment()          STUB   returns the images unchanged; plug in your segmenter.
  * LightGlue re-rank  NOT YET  matching is cosine distance on embeddings only.
Thresholds are uncalibrated defaults. Tune them on labelled data.

S3 lifecycle: rejected images are deleted, accepted ones are copied to accepted/ before the DB
write and incoming/ is cleaned up after it. A crash leaves files in incoming/, which a bucket
lifecycle rule expires.
"""
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import store
from gate import BIOCLIP_VERSION, BioClip, check_image, combine

log = logging.getLogger("turtle.worker")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "facebook/dinov2-base")
EMBED_VERSION = os.environ.get("EMBED_MODEL_VERSION", "dinov2-base-v1")
SEG_VERSION = os.environ.get("SEG_MODEL_VERSION", "none")
MATCH_DIST = float(os.environ.get("MATCH_DIST", "0.20"))    # cosine distance <= this: same turtle
REVIEW_DIST = float(os.environ.get("REVIEW_DIST", "0.40"))  # <= this: ask a human; above: new turtle
MAX_IMAGE_BYTES = 15 * 1024 * 1024
FACE_VIEWS = ("left_face", "right_face")

IN_BUCKET = os.environ["S3_INCOMING_BUCKET"]
IN_PREFIX = os.environ.get("S3_INCOMING_PREFIX", "incoming/")
ACC_BUCKET = os.environ["S3_ACCEPTED_BUCKET"]
ACC_PREFIX = os.environ.get("S3_ACCEPTED_PREFIX", "accepted/")

REQUIRED = set(os.environ.get("REQUIRED_SLOTS", "whole,left,right").split(","))
LEATHERBACK_MIN_CONF = float(os.environ.get("LEATHERBACK_MIN_CONF", "0.70"))

REASON_TEXT = {
    "not_a_sea_turtle": "does not look like a sea turtle",
    "no_turtle_detected": "no turtle found in the photo",
    "too_blurry": "too blurry, retake it in good light",
    "too_small": "resolution too low",
    "download_failed": "could not be read as an image",
}


class Pipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_lock = threading.Lock()  # one forward pass at a time; S3 and DB stay parallel
        log.info("loading %s on %s", EMBED_MODEL, self.device)
        self.processor = AutoImageProcessor.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.bioclip = BioClip(self.device, self.gpu_lock)
        self.io = ThreadPoolExecutor(8)  # parallel S3 reads
        self.s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))

    # ---- storage
    def fetch(self, key: str) -> Image.Image:
        obj = self.s3.get_object(Bucket=IN_BUCKET, Key=key)
        if obj["ContentLength"] > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

    @staticmethod
    def accepted_key(key: str) -> str:
        return ACC_PREFIX + key[len(IN_PREFIX):]

    def _delete(self, bucket: str, keys: list[str]):
        if keys:
            self.s3.delete_objects(Bucket=bucket,
                                   Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True})

    # ---- stages (each takes the whole submission as a batch)
    def detect(self, imgs: list[Image.Image], views: list[str]) -> list[Image.Image | None]:
        return imgs  # TODO: batched turtle detector -> crops; None where no turtle is found

    def segment(self, imgs: list[Image.Image], views: list[str]) -> list[Image.Image]:
        return imgs  # TODO: batched scute segmentation

    @torch.inference_mode()
    def embed_batch(self, imgs: list[Image.Image]) -> np.ndarray:
        x = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
        if self.device == "cuda":
            x = x.half()
        with self.gpu_lock:
            v = self.model(pixel_values=x).pooler_output.float().cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _read(self, key: str) -> Image.Image | None:
        if not key.startswith(IN_PREFIX):
            return None
        try:
            return self.fetch(key)
        except Exception:
            log.warning("cannot read %s", key, exc_info=True)
            return None

    def _screen(self, images: list[dict]) -> tuple[list[dict], list[dict]]:
        """Stages 1-3. Returns the per-photo records and the ones that reject the whole request."""
        recs = [{"index": n, "slot": i["slot"], "view": i["view"], "key": i["key"],
                 "required": i["slot"] in REQUIRED, "reason": None, "detected": True,
                 "turtle_score": None, "verdict": None, "img": None, "crop": None}
                for n, i in enumerate(images, 1)]

        def fatal() -> list[dict]:
            return [r for r in recs if r["reason"] and (r["required"] or r["reason"] == "not_a_sea_turtle")]

        # 1. decode + quality (an optional photo that fails is just dropped)
        for r, img in zip(recs, self.io.map(self._read, [r["key"] for r in recs])):
            r["img"] = img
            r["reason"] = "download_failed" if img is None else check_image(img)
        if fatal():
            return recs, fatal()

        # 2. detect (required: must find a turtle; optional: fall back to the full image)
        live = [r for r in recs if not r["reason"]]
        for r, crop in zip(live, self.detect([r["img"] for r in live], [r["view"] for r in live])):
            if crop is not None:
                r["crop"] = crop
            elif r["required"]:
                r["reason"] = "no_turtle_detected"
            else:
                r["crop"], r["detected"] = r["img"], False
        if fatal():
            return recs, fatal()

        # 3. BioCLIP on every remaining photo; any non-turtle rejects the whole request
        live = [r for r in recs if not r["reason"]]
        for r, v in zip(live, self.bioclip.classify_batch([r["crop"] for r in live])):
            r["verdict"], r["turtle_score"] = v, round(v.turtle_score, 4)
            if not v.is_turtle:
                r["reason"] = "not_a_sea_turtle"
        return recs, fatal()

    def _embed_faces(self, accepted: list[dict]) -> dict:
        """Stages 4-5: segment, then DINOv2-embed the left/right face photos as one batch."""
        faces = [r for r in accepted if r["view"] in FACE_VIEWS and r["detected"]]
        if not faces:
            return {}
        segs = self.segment([r["crop"] for r in faces], [r["view"] for r in faces])
        vecs = self.embed_batch(segs)
        return {r["view"]: (v, self.accepted_key(r["key"])) for r, v in zip(faces, vecs)}

    # ---- orchestration
    def identify(self, job_id: str, images: list[dict], uid: str) -> dict:
        recs, fatal = self._screen(images)
        versions = {"species": BIOCLIP_VERSION, "embedder": EMBED_VERSION, "segmenter": SEG_VERSION}
        if fatal:
            return self._reject(job_id, recs, fatal, versions)

        accepted = [r for r in recs if not r["reason"]]
        label, conf, scores = combine([r["verdict"] for r in accepted])
        species = {"label": label, "confidence": round(conf, 4), "scores": scores}
        leatherback = label == "leatherback" and conf >= LEATHERBACK_MIN_CONF
        embeds = {} if leatherback else self._embed_faces(accepted)
        for r in recs:
            r["img"] = r["crop"] = None  # free memory

        copied = []
        try:
            for r in accepted:
                dst = self.accepted_key(r["key"])
                self.s3.copy_object(Bucket=ACC_BUCKET, Key=dst,
                                    CopySource={"Bucket": IN_BUCKET, "Key": r["key"]})
                copied.append(dst)
            with store.connect() as conn:
                if leatherback:
                    ident = {"status": "skipped", "turtle_id": None, "confidence": None,
                             "can_retry": False,
                             "reason": "Leatherback: species recorded. Leatherbacks have no scutes, "
                                       "so individual identification is not supported."}
                    store.save_sighting(conn, job_id, None, species, "skipped", None)
                elif not embeds:
                    ident = {"status": "retry_scute", "turtle_id": None, "can_retry": True,
                             "reason": "Upload a clear left or right face photo to identify the turtle."}
                    store.save_sighting(conn, job_id, None, species, "insufficient_views", None)
                else:
                    ident = self._match(conn, job_id, species, embeds)
                if any(r["verdict"].flagged for r in accepted):
                    ident["needs_review"] = True  # accepted, but BioCLIP was unsure it is a sea turtle
                result = {"species": species, "identification": ident,
                          "images": [self._image_out(r) for r in recs], "model_versions": versions}
                store.finish_job(conn, job_id, result)
        except Exception:
            self._delete(ACC_BUCKET, copied)  # DB write failed: do not leave orphans in accepted/
            raise
        self._delete(IN_BUCKET, [r["key"] for r in recs if r["key"].startswith(IN_PREFIX)])
        return result

    def _reject(self, job_id: str, recs: list[dict], fatal: list[dict], versions: dict) -> dict:
        """Whole-request rejection: nothing is kept; the reply says which photos to retake."""
        status = "no_sea_turtle" if any(r["reason"] == "not_a_sea_turtle" for r in fatal) else "rejected"
        text = "; ".join(f"{r['slot']}: {REASON_TEXT[r['reason']]}" for r in fatal)
        result = {
            "species": {"label": "unknown", "confidence": 0.0, "scores": {}},
            "identification": {"status": status, "turtle_id": None, "can_retry": True,
                               "reason": f"Please retake: {text}."},
            "images": [self._image_out(r) for r in recs], "model_versions": versions,
        }
        with store.connect() as conn:
            store.finish_job(conn, job_id, result)
        self._delete(IN_BUCKET, [r["key"] for r in recs if r["key"].startswith(IN_PREFIX)])
        return result

    @staticmethod
    def _image_out(r: dict) -> dict:
        return {"index": r["index"], "slot": r["slot"], "view": r["view"], "usable": r["reason"] is None,
                "reason": r["reason"], "detected": r["detected"], "turtle_score": r["turtle_score"]}

    def _match(self, conn, job_id, species, embeds) -> dict:
        best = None  # (dist, turtle_id)
        for side, (vec, _) in embeds.items():
            hit = store.nearest(conn, vec, side, EMBED_VERSION)
            if hit and (best is None or hit[1] < best[0]):
                best = (hit[1], hit[0])

        if best and best[0] <= MATCH_DIST:
            tid, conf = best[1], 1 - best[0]
            sid = store.save_sighting(conn, job_id, tid, species, "matched", conf)
            store.add_embeddings(conn, tid, sid, embeds, EMBED_VERSION, SEG_VERSION)  # grow gallery
            return {"status": "matched", "turtle_id": tid, "confidence": conf, "can_retry": False}

        if best and best[0] <= REVIEW_DIST:
            tid = store.new_turtle(conn, species["label"], provisional=True)
            sid = store.save_sighting(conn, job_id, tid, species, "uncertain", 1 - best[0])
            store.add_embeddings(conn, tid, sid, embeds, EMBED_VERSION, SEG_VERSION)
            store.add_review(conn, sid, tid, [{"turtle_id": best[1], "distance": best[0]}])
            return {"status": "review", "turtle_id": tid, "confidence": 1 - best[0],
                    "is_provisional": True, "can_retry": False}

        tid = store.new_turtle(conn, species["label"], provisional=False)
        sid = store.save_sighting(conn, job_id, tid, species, "new", None)
        store.add_embeddings(conn, tid, sid, embeds, EMBED_VERSION, SEG_VERSION)
        return {"status": "new", "turtle_id": tid, "confidence": None, "can_retry": False}
