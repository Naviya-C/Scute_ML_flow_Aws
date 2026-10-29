"""The image gate: cheap CPU checks first, then ONE BioCLIP pass with two outputs.

    turtle_score  probability mass on the sea-turtle labels (vs. the negatives below)
    species       the best sea-turtle label, or "unknown" when BioCLIP is sure it is a sea
                  turtle but cannot tell which species

All labels share one zero-shot softmax, so a single forward pass answers both questions.
Label lists and thresholds are uncalibrated placeholders: tune them on labelled data
(precision/recall on real phone photos), and log the scores while you do.
"""
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
import open_clip
import torch
from PIL import Image

log = logging.getLogger("turtle.worker")

# API species label -> scientific name BioCLIP was trained on
SPECIES = {
    "green_turtle": "Chelonia mydas",
    "hawksbill": "Eretmochelys imbricata",
    "loggerhead": "Caretta caretta",
    "olive_ridley": "Lepidochelys olivacea",
    "kemps_ridley": "Lepidochelys kempii",
    "leatherback": "Dermochelys coriacea",
    "flatback": "Natator depressus",
}
# Things users upload by mistake, and look-alikes. Hard negatives matter most.
NEGATIVES = [
    "a freshwater turtle", "a tortoise", "a lizard", "a fish", "a dolphin", "a seal", "a bird",
    "a person", "a hand", "a beach", "the ocean", "a rock", "a boat",
    "a toy turtle", "a drawing of a turtle", "a screenshot of text",
]

BIOCLIP_MODEL = os.environ.get("BIOCLIP_MODEL", "hf-hub:imageomics/bioclip-2")
BIOCLIP_VERSION = os.environ.get("BIOCLIP_MODEL_VERSION", "bioclip-2")
REJECT_BELOW = float(os.environ.get("TURTLE_REJECT_BELOW", "0.30"))  # below: not a sea turtle
ACCEPT_ABOVE = float(os.environ.get("TURTLE_ACCEPT_ABOVE", "0.70"))  # above: clearly one; between: flag
SPECIES_MIN_CONF = float(os.environ.get("SPECIES_MIN_CONF", "0.50"))
MIN_SIDE_PX = int(os.environ.get("MIN_SIDE_PX", "224"))
MIN_SHARPNESS = float(os.environ.get("MIN_SHARPNESS", "15"))  # variance of the Laplacian


@dataclass
class Verdict:
    turtle_score: float
    species: str            # API label or "unknown"
    species_conf: float
    scores: dict            # species label -> share among sea-turtle labels
    flagged: bool           # accepted, but turtle_score is in the uncertain band

    @property
    def is_turtle(self) -> bool:
        return self.turtle_score >= REJECT_BELOW


def combine(verdicts: list[Verdict]) -> tuple[str, float, dict]:
    """One species for the whole submission: per-photo species shares, weighted by how sure each
    photo is that it shows a sea turtle. A photo that cannot tell the species adds little."""
    if not verdicts:
        return "unknown", 0.0, {}
    w = np.array([v.turtle_score for v in verdicts])
    mat = np.array([[v.scores[n] for n in SPECIES] for v in verdicts])
    agg = (w[:, None] * mat).sum(axis=0) / max(float(w.sum()), 1e-9)
    best = int(agg.argmax())
    label = list(SPECIES)[best] if agg[best] >= SPECIES_MIN_CONF else "unknown"
    return label, float(agg[best]), {n: round(float(a), 4) for n, a in zip(SPECIES, agg)}


def check_image(img: Image.Image) -> str | None:
    """Cheap CPU checks. Returns a rejection reason, or None if the image is usable."""
    if min(img.size) < MIN_SIDE_PX:
        return "too_small"
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if float(lap.var()) < MIN_SHARPNESS:
        return "too_blurry"
    return None


class BioClip:
    def __init__(self, device: str, lock: threading.Lock):
        self.device, self.lock = device, lock
        log.info("loading %s on %s", BIOCLIP_MODEL, device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
        self.model = self.model.to(device).eval()
        if device == "cuda":
            self.model.half()
        tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
        self.names = list(SPECIES)
        prompts = [f"a photo of {sci}." for sci in SPECIES.values()]
        prompts += [f"a photo of {n}." for n in NEGATIVES]
        with torch.inference_mode():  # text side runs once; each request only runs the image tower
            t = self.model.encode_text(tokenizer(prompts).to(device))
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.scale = self.model.logit_scale.exp().item()

    @torch.inference_mode()
    def classify_batch(self, imgs: list[Image.Image]) -> list[Verdict]:
        """One forward pass for all of a submission's photos."""
        if not imgs:
            return []
        x = torch.stack([self.preprocess(i) for i in imgs]).to(self.device)
        if self.device == "cuda":
            x = x.half()
        with self.lock:
            f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        p = (self.scale * f @ self.text.T).softmax(dim=-1).float().cpu().numpy()
        return [self._verdict(row) for row in p]

    def _verdict(self, p: np.ndarray) -> Verdict:
        sea = p[: len(self.names)]
        turtle_score = float(sea.sum())
        share = sea / max(turtle_score, 1e-9)
        best = int(share.argmax())
        species = self.names[best] if share[best] >= SPECIES_MIN_CONF else "unknown"
        return Verdict(
            turtle_score=turtle_score, species=species, species_conf=float(share[best]),
            scores={n: round(float(s), 4) for n, s in zip(self.names, share)},
            flagged=REJECT_BELOW <= turtle_score < ACCEPT_ABOVE,
        )
