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
  * detect()           REAL   RF-DETR Large, single class "sea_turtle" (DETECT_MODEL_PATH). Best box
                              above DETECT_MIN_CONF, padded by DETECT_PAD, clamped -> crop; None if no box.
  * segment()          REAL   YOLO-seg scute model on the face views (SEG_MODEL_PATH). Best mask,
                              background set to black, cropped to the mask; no mask -> detection crop.
  * embed_batch()      REAL   DINOv2 ViT-B/14 (768-dim), the model the DB schema was built for.
  * LightGlue re-rank  NOT YET  matching is cosine distance on embeddings only.
detect()/segment() pass images through unchanged when their *_MODEL_PATH is unset (CI, local dev).
Thresholds are uncalibrated defaults. Tune them on labelled data.

Embedding versions: vectors of full photos, detector crops and segmented scutes must never be
compared, and store.nearest() only compares vectors with the same embed_model_version.
check_versions() refuses to start when EMBED_MODEL_VERSION does not match what is embedded:
"-seg" exactly when the segmenter is on, and never the full-photo "dinov2-base-v1" once the
detector is on.

S3 lifecycle: rejected images are deleted, accepted ones are copied to accepted/ before the DB
write and incoming/ is cleaned up after it. A crash leaves files in incoming/, which a bucket
lifecycle rule expires.
"""
import io
import logging
import math
import os
import threading
import time
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

DETECT_MODEL_PATH = os.environ.get("DETECT_MODEL_PATH", "")
SEG_MODEL_PATH = os.environ.get("SEG_MODEL_PATH", "")
DETECT_MIN_CONF = float(os.environ.get("DETECT_MIN_CONF", "0.5"))
DETECT_PAD = float(os.environ.get("DETECT_PAD", "0.10"))   # fraction of the box width/height
SEG_MIN_CONF = float(os.environ.get("SEG_MIN_CONF", "0.25"))
TURTLE_CLASS_ID = 0  # the detector has one class: "sea_turtle"

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


# ---- pure helpers for detect/segment (no torch; unit-tested with fake model outputs)
FULL_PHOTO_EMBED_VERSION = "dinov2-base-v1"  # the existing DB vectors: DINOv2 on full photos


def check_versions(detector_on: bool, segmenter_on: bool, embed_version: str, seg_version: str) -> tuple[str, str]:
    """Refuse configs that would store vectors of different inputs (full photo, detector crop,
    segmented scute) under the same embed_model_version, where store.nearest() would compare them."""
    if segmenter_on and ("-seg" not in embed_version or seg_version == "none"):
        raise RuntimeError(f"segmenter is on, so EMBED_MODEL_VERSION must contain '-seg' and "
                           f"SEG_MODEL_VERSION must be set (got {embed_version!r}, {seg_version!r})")
    if not segmenter_on and ("-seg" in embed_version or seg_version != "none"):
        raise RuntimeError(f"no SEG_MODEL_PATH: EMBED_MODEL_VERSION must not contain '-seg' and "
                           f"SEG_MODEL_VERSION must be 'none' (got {embed_version!r}, {seg_version!r})")
    if detector_on and embed_version == FULL_PHOTO_EMBED_VERSION:
        raise RuntimeError(f"detector is on, so faces are embedded as crops: EMBED_MODEL_VERSION must not be "
                           f"{FULL_PHOTO_EMBED_VERSION!r}, the full-photo version (e.g. dinov2-base-det-v1)")
    return embed_version, seg_version


def best_box(xyxy, conf, cls, min_conf: float, class_id: int = TURTLE_CLASS_ID):
    """Highest-confidence box of class_id with conf >= min_conf, as (box, conf); None if none."""
    xyxy, conf, cls = np.asarray(xyxy, dtype=float).reshape(-1, 4), np.asarray(conf), np.asarray(cls)
    keep = np.flatnonzero((cls == class_id) & (conf >= min_conf))
    if not keep.size:
        return None
    i = keep[conf[keep].argmax()]
    return xyxy[i], float(conf[i])


def pad_box(box, pad: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Grow a box by pad * its size on every side and clamp it to the image. None if nothing is left."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = pad * (x2 - x1), pad * (y2 - y1)
    x1, y1 = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    x2, y2 = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def best_mask(masks, conf):
    """The mask with the highest confidence from an (N, H, W) stack; None if there are none."""
    if masks is None or len(masks) == 0:
        return None
    return np.asarray(masks)[int(np.asarray(conf).argmax())]


def mask_crop(img: Image.Image, mask) -> Image.Image | None:
    """Black out everything outside the mask and crop to the mask's bounding box. None if empty."""
    m = np.asarray(mask)
    if m.shape != (img.height, img.width):  # masks can come back at the model's input size
        m = np.asarray(Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize(img.size, Image.NEAREST))
    m = m > 0.5 if m.dtype != bool else m
    if not m.any():
        return None
    arr = np.asarray(img.convert("RGB")).copy()
    arr[~m] = 0
    ys, xs = np.nonzero(m)
    return Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _np(t) -> np.ndarray:
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


class Pipeline:
    detector = segmenter = None  # None: that stage passes images through unchanged
    embed_version, seg_version = EMBED_VERSION, SEG_VERSION

    def __init__(self):
        # fail fast, before any model loads
        self.embed_version, self.seg_version = check_versions(bool(DETECT_MODEL_PATH), bool(SEG_MODEL_PATH),
                                                              EMBED_VERSION, SEG_VERSION)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_lock = threading.Lock()  # one forward pass at a time; S3 and DB stay parallel
        self.detector = self._load_detector()
        self.segmenter = self._load_segmenter()
        log.info("loading %s on %s", EMBED_MODEL, self.device)
        self.processor = AutoImageProcessor.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.bioclip = BioClip(self.device, self.gpu_lock)
        self.io = ThreadPoolExecutor(8)  # parallel S3 reads
        self.s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-southeast-2"))
        log.info("versions: embedder=%s segmenter=%s", self.embed_version, self.seg_version)

    # ---- model loading (rfdetr / ultralytics are imported only when their weights are configured)
    def _load_detector(self):
        if not DETECT_MODEL_PATH:
            log.warning("DETECT_MODEL_PATH not set: detect() passes images through unchanged")
            return None
        if not os.path.isfile(DETECT_MODEL_PATH):
            raise FileNotFoundError(f"DETECT_MODEL_PATH={DETECT_MODEL_PATH} does not exist")
        from rfdetr import RFDETRLarge
        log.info("loading RF-DETR %s on %s", DETECT_MODEL_PATH, self.device)
        m = RFDETRLarge(pretrain_weights=DETECT_MODEL_PATH, num_classes=1, device=self.device)
        # fp16 without tracing: a traced model only accepts the batch size it was traced with
        m.inference(compile=False, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        return m

    def _load_segmenter(self):
        if not SEG_MODEL_PATH:
            log.warning("SEG_MODEL_PATH not set: segment() passes images through unchanged")
            return None
        if not os.path.isfile(SEG_MODEL_PATH):
            raise FileNotFoundError(f"SEG_MODEL_PATH={SEG_MODEL_PATH} does not exist")
        from ultralytics import YOLO
        log.info("loading YOLO-seg %s on %s", SEG_MODEL_PATH, self.device)
        m = YOLO(SEG_MODEL_PATH)
        if m.task != "segment":
            raise RuntimeError(f"SEG_MODEL_PATH={SEG_MODEL_PATH} is a '{m.task}' model "
                               f"({len(m.names)} classes, e.g. {list(m.names.values())[:3]}), "
                               f"not a segmentation model")
        log.info("segmenter classes: %s", m.names)
        return m

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
        """One RF-DETR pass for the batch: padded crop of the best turtle box, None where there is none."""
        if self.detector is None or not imgs:
            return imgs
        t = time.monotonic()
        with self.gpu_lock:
            dets = self.detector.predict(imgs, threshold=DETECT_MIN_CONF, include_source_image=False)
        if not isinstance(dets, list):  # a batch of one comes back as a bare Detections
            dets = [dets]
        out, found = [], []
        for img, view, d in zip(imgs, views, dets):
            hit = best_box(d.xyxy, d.confidence, d.class_id, DETECT_MIN_CONF)
            box = hit and pad_box(hit[0], DETECT_PAD, img.width, img.height)
            out.append(img.crop(box) if box else None)
            found.append(f"{view}={hit[1]:.2f}" if box else f"{view}=none")
        log.info("detect: %d images in %.2fs [%s]", len(imgs), time.monotonic() - t, " ".join(found))
        return out

    def segment(self, imgs: list[Image.Image], views: list[str]) -> list[Image.Image]:
        """One YOLO-seg pass over the face views: masked scute crop, or the detection crop if no mask."""
        idx = [i for i, v in enumerate(views) if v in FACE_VIEWS]
        if self.segmenter is None or not idx:
            return imgs
        t = time.monotonic()
        with self.gpu_lock:
            res = self.segmenter.predict([imgs[i] for i in idx], conf=SEG_MIN_CONF, retina_masks=True,
                                         half=self.device == "cuda", device=self.device, verbose=False)
        out, found = list(imgs), []
        for i, r in zip(idx, res):
            mask = None if r.masks is None else best_mask(_np(r.masks.data), _np(r.boxes.conf))
            crop = None if mask is None else mask_crop(imgs[i], mask)
            if crop is None:
                log.warning("segment %s: no scute mask, using the detection crop", views[i])
            else:
                out[i] = crop
            found.append(f"{views[i]}={'%dx%d' % crop.size if crop else 'none'}")
        log.info("segment: %d faces in %.2fs [%s]", len(idx), time.monotonic() - t, " ".join(found))
        return out

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
        versions = {"species": BIOCLIP_VERSION, "embedder": self.embed_version, "segmenter": self.seg_version}
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
            hit = store.nearest(conn, vec, side, self.embed_version)
            if hit and (best is None or hit[1] < best[0]):
                best = (hit[1], hit[0])

        if best and best[0] <= MATCH_DIST:
            tid, conf = best[1], 1 - best[0]
            sid = store.save_sighting(conn, job_id, tid, species, "matched", conf)
            store.add_embeddings(conn, tid, sid, embeds, self.embed_version, self.seg_version)  # grow gallery
            return {"status": "matched", "turtle_id": tid, "confidence": conf, "can_retry": False}

        if best and best[0] <= REVIEW_DIST:
            tid = store.new_turtle(conn, species["label"], provisional=True)
            sid = store.save_sighting(conn, job_id, tid, species, "uncertain", 1 - best[0])
            store.add_embeddings(conn, tid, sid, embeds, self.embed_version, self.seg_version)
            store.add_review(conn, sid, tid, [{"turtle_id": best[1], "distance": best[0]}])
            return {"status": "review", "turtle_id": tid, "confidence": 1 - best[0],
                    "is_provisional": True, "can_retry": False}

        tid = store.new_turtle(conn, species["label"], provisional=False)
        sid = store.save_sighting(conn, job_id, tid, species, "new", None)
        store.add_embeddings(conn, tid, sid, embeds, self.embed_version, self.seg_version)
        return {"status": "new", "turtle_id": tid, "confidence": None, "can_retry": False}
