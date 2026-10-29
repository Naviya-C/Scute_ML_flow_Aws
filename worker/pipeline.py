"""The ML pipeline: image -> species -> embedding -> match against known turtles.

Transport-agnostic: main.py (HTTP, phase 1) and later an SQS consumer both call identify().

What is real vs placeholder:
  * embed()            REAL   DINOv2 ViT-B/14 (768-dim), the model the DB schema was built for.
  * classify_species() STUB   returns "unknown"; plug in your species classifier.
  * crop()             STUB   returns the image unchanged; plug in your detector/segmenter.
  * LightGlue re-rank  NOT YET  matching is cosine distance on embeddings only.
Thresholds below are uncalibrated defaults. Tune them on labelled data.
"""
import io
import logging
import os
import threading

import httpx
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import store

log = logging.getLogger("turtle.worker")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "facebook/dinov2-base")
EMBED_VERSION = os.environ.get("EMBED_MODEL_VERSION", "dinov2-base-v1")
SEG_VERSION = os.environ.get("SEG_MODEL_VERSION", "none")
MATCH_DIST = float(os.environ.get("MATCH_DIST", "0.20"))    # cosine distance <= this: same turtle
REVIEW_DIST = float(os.environ.get("REVIEW_DIST", "0.40"))  # <= this: ask a human; above: new turtle
MAX_IMAGE_BYTES = 15 * 1024 * 1024
FACE_VIEWS = ("left_face", "right_face")


class Pipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading %s on %s", EMBED_MODEL, self.device)
        self.processor = AutoImageProcessor.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.gpu_lock = threading.Lock()  # one forward pass at a time; downloads/DB stay parallel
        self.http = httpx.Client(timeout=20, follow_redirects=False)

    # ---- stages
    def fetch(self, url: str) -> Image.Image:
        if not url.startswith("https://"):
            raise ValueError("image url must be https")
        with self.http.stream("GET", url) as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_bytes():
                buf.write(chunk)
                if buf.tell() > MAX_IMAGE_BYTES:
                    raise ValueError("image too large")
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def classify_species(self, img: Image.Image) -> tuple[str, float]:
        return "unknown", 0.0  # TODO: species classifier

    def crop(self, img: Image.Image, view: str) -> Image.Image:
        return img  # TODO: detector + scute segmentation

    @torch.inference_mode()
    def embed(self, img: Image.Image) -> np.ndarray:
        x = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        if self.device == "cuda":
            x = x.half()
        with self.gpu_lock:
            v = self.model(pixel_values=x).pooler_output[0].float().cpu().numpy()
        return v / np.linalg.norm(v)

    # ---- orchestration
    def identify(self, job_id: str, images: list[dict], uid: str) -> dict:
        by_view = {i["view"]: i["url"] for i in images}
        first = next(iter(by_view.values()))
        label, sp_conf = self.classify_species(self.fetch(by_view.get("whole", first)))
        species = {"label": label, "confidence": sp_conf}

        embeds = {}  # side -> (vector, url)
        for side in FACE_VIEWS:
            if side in by_view:
                img = self.crop(self.fetch(by_view[side]), side)
                embeds[side] = (self.embed(img), by_view[side])

        with store.connect() as conn:
            if not embeds:
                ident = {"status": "retry_scute", "turtle_id": None, "can_retry": True,
                         "reason": "Upload a clear left or right face photo to identify the turtle."}
                store.save_sighting(conn, job_id, None, species, "insufficient_views", None)
            else:
                ident = self._match(conn, job_id, species, embeds)
            result = {"species": species, "identification": ident}
            store.finish_job(conn, job_id, result)
        return result

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
