"""Decision rules of the worker pipeline, run with stubbed models, S3 and DB (no GPU needed).

    pytest worker/tests
"""
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

os.environ.update(S3_INCOMING_BUCKET="in", S3_ACCEPTED_BUCKET="acc", REQUIRED_SLOTS="whole,left,right")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Heavy / external modules are replaced before the worker code is imported.
for name in ("torch", "open_clip", "transformers", "boto3", "psycopg", "pgvector", "pgvector.psycopg",
             "psycopg.types", "psycopg.types.json"):
    sys.modules.setdefault(name, MagicMock())
torch = sys.modules["torch"]
torch.inference_mode = lambda: (lambda f: f)  # decorator that does nothing
sys.modules["store"] = MagicMock()

import gate  # noqa: E402
import pipeline  # noqa: E402

SLOTS = {"whole": "whole", "left": "left_face", "right": "right_face",
         "front": "front", "shell": "carapace", "rear": "rear"}


def verdict(turtle=0.95, species="green_turtle", flagged=False):
    scores = {n: 0.0 for n in gate.SPECIES}
    scores[species if species != "unknown" else "green_turtle"] = 1.0
    return gate.Verdict(turtle_score=turtle, species=species, species_conf=1.0,
                        scores=scores, flagged=flagged)


class Fake(pipeline.Pipeline):
    """Pipeline with scripted models: per-slot image quality and per-slot BioCLIP verdicts."""

    def __init__(self, verdicts, blurry=(), missing=(), undetected=()):
        self.verdicts, self.blurry, self.missing, self.undetected = verdicts, set(blurry), set(missing), set(undetected)
        self.io = types.SimpleNamespace(map=lambda f, xs: map(f, xs))
        self.s3, self.calls = MagicMock(), []

    def _read(self, key):
        slot = key.split("/")[-1].split(".")[0]
        if slot in self.missing:
            return None
        sharp = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        return Image.fromarray(flat if slot in self.blurry else sharp)

    def detect(self, imgs, views):
        out = list(imgs)
        for i, v in enumerate(views):
            if any(s for s, view in SLOTS.items() if view == v and s in self.undetected):
                out[i] = None
        return out

    def segment(self, imgs, views):
        return imgs

    def embed_batch(self, imgs):
        self.calls.append(("embed", len(imgs)))
        return np.ones((len(imgs), 4), dtype=np.float32)

    @property
    def bioclip(self):
        outer = self
        return types.SimpleNamespace(classify_batch=lambda crops: [outer.verdicts[outer._order.pop(0)] for _ in crops])

    def run(self, slots=("whole", "left", "right")):
        images = [{"slot": s, "view": SLOTS[s], "key": f"incoming/2026/{s}.jpg"} for s in slots]
        self._order = []
        # classify_batch pops verdicts in the order the surviving photos reach stage 3
        self._order = [s for s in slots if s in self.verdicts]
        pipeline.store = MagicMock()
        pipeline.store.save_sighting.return_value = "sid"
        self._match = lambda conn, job, sp, emb: {"status": "new", "turtle_id": "GT-1", "can_retry": False}
        result = self.identify("job1", images, "u")
        return result, pipeline.store


def good():
    return {s: verdict() for s in ("whole", "left", "right", "front", "shell", "rear")}


def deleted(fake):
    return [c.kwargs["Delete"]["Objects"] for c in fake.s3.delete_objects.call_args_list]


def test_happy_path_batches_both_faces_in_one_embed_call():
    f = Fake(good())
    result, store = f.run()
    assert result["identification"]["status"] == "new"
    assert f.calls == [("embed", 2)]  # left + right in ONE batch
    assert all(i["usable"] for i in result["images"])
    assert f.s3.copy_object.call_count == 3  # all three moved to accepted/


def test_required_photo_blurry_rejects_whole_request():
    f = Fake(good(), blurry=["left"])
    result, store = f.run()
    assert result["identification"]["status"] == "rejected" and result["identification"]["can_retry"]
    assert "left: too blurry" in result["identification"]["reason"]
    f.s3.copy_object.assert_not_called()            # nothing kept
    assert len(deleted(f)[0]) == 3                   # all photos deleted from incoming/
    store.save_sighting.assert_not_called()
    assert f.calls == []                             # no GPU work after a failed check


def test_optional_photo_blurry_is_dropped_not_fatal():
    f = Fake(good(), blurry=["rear"])
    result, _ = f.run(("whole", "left", "right", "rear"))
    assert result["identification"]["status"] == "new"
    rear = [i for i in result["images"] if i["slot"] == "rear"][0]
    assert rear["usable"] is False and rear["reason"] == "too_blurry"


def test_required_not_detected_rejects_optional_not_detected_falls_back():
    f = Fake(good(), undetected=["right"])
    result, _ = f.run()
    assert result["identification"]["status"] == "rejected"
    assert "right: no turtle found" in result["identification"]["reason"]

    f = Fake(good(), undetected=["shell"])
    result, _ = f.run(("whole", "left", "right", "shell"))
    assert result["identification"]["status"] == "new"
    assert [i for i in result["images"] if i["slot"] == "shell"][0]["detected"] is False


def test_any_non_turtle_rejects_whole_request_even_optional():
    v = good()
    v["rear"] = verdict(turtle=0.02)
    f = Fake(v)
    result, store = f.run(("whole", "left", "right", "rear"))
    assert result["identification"]["status"] == "no_sea_turtle"
    assert "rear: does not look like a sea turtle" in result["identification"]["reason"]
    f.s3.copy_object.assert_not_called()
    store.save_sighting.assert_not_called()


def test_leatherback_exits_early_without_embedding():
    v = {s: verdict(species="leatherback") for s in ("whole", "left", "right")}
    f = Fake(v)
    result, store = f.run()
    assert result["species"]["label"] == "leatherback"
    assert result["identification"]["status"] == "skipped"
    assert result["identification"]["turtle_id"] is None
    assert f.calls == []                             # no segment / DINOv2
    assert store.save_sighting.call_args.args[4] == "skipped"
    assert f.s3.copy_object.call_count == 3          # sighting is kept


def test_unknown_species_still_identifies_the_individual():
    f = Fake({s: verdict(species="unknown") for s in ("whole", "left", "right")})
    # make the species shares flat so no species wins
    for v in f.verdicts.values():
        v.scores = {n: 1 / len(gate.SPECIES) for n in gate.SPECIES}
    result, _ = f.run()
    assert result["species"]["label"] == "unknown"
    assert result["identification"]["status"] == "new"


def test_borderline_turtle_score_is_accepted_and_flagged():
    v = good()
    v["whole"] = verdict(turtle=0.5, flagged=True)
    result, _ = Fake(v).run()
    assert result["identification"]["status"] == "new" and result["identification"]["needs_review"] is True


def test_combine_weights_photos_by_turtle_score():
    a = verdict(turtle=0.99, species="hawksbill")
    b = verdict(turtle=0.31, species="loggerhead")
    label, conf, _ = gate.combine([a, b])
    assert label == "hawksbill" and conf > 0.5
