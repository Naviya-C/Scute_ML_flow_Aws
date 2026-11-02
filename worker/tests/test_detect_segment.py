"""detect() / segment() and their crop helpers, run with fake model outputs (no weights, no GPU).

    pytest worker/tests
"""
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

os.environ.update(S3_INCOMING_BUCKET="in", S3_ACCEPTED_BUCKET="acc")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for name in ("torch", "open_clip", "transformers", "boto3", "psycopg", "pgvector", "pgvector.psycopg",
             "psycopg.types", "psycopg.types.json", "store"):
    sys.modules.setdefault(name, MagicMock())
sys.modules["torch"].inference_mode = lambda: (lambda f: f)

import pipeline  # noqa: E402
from pipeline import best_box, best_mask, check_versions, mask_crop, pad_box  # noqa: E402


def img(w=400, h=300, value=200):
    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8))


def bare_pipeline(detector=None, segmenter=None):
    p = object.__new__(pipeline.Pipeline)  # skip __init__: no model loading
    p.device, p.gpu_lock, p.detector, p.segmenter = "cpu", threading.Lock(), detector, segmenter
    return p


class FakeModel:
    """Records calls and checks that every forward pass runs under the shared GPU lock."""

    def __init__(self, outputs):
        self.outputs, self.calls, self.lock = outputs, [], None

    def predict(self, images, **kw):
        assert self.lock.locked(), "forward pass must hold gpu_lock"
        self.calls.append((len(images), kw))
        return self.outputs


def with_model(p, attr, model):
    model.lock = p.gpu_lock
    setattr(p, attr, model)
    return model


def dets(xyxy, conf, cls=None):
    return SimpleNamespace(xyxy=np.array(xyxy, dtype=float).reshape(-1, 4), confidence=np.array(conf, dtype=float),
                           class_id=np.array(cls if cls is not None else [0] * len(conf)))


def seg_result(masks, conf):
    if masks is None:
        return SimpleNamespace(masks=None, boxes=SimpleNamespace(conf=np.array([])))
    return SimpleNamespace(masks=SimpleNamespace(data=np.array(masks, dtype=np.float32)),
                           boxes=SimpleNamespace(conf=np.array(conf, dtype=float)))


# ---- box helpers
def test_best_box_picks_highest_conf_turtle_above_threshold():
    box, conf = best_box([[0, 0, 10, 10], [5, 5, 50, 50], [1, 1, 2, 2]], [0.6, 0.9, 0.3], [0, 0, 0], 0.5)
    assert conf == 0.9 and list(box) == [5, 5, 50, 50]


def test_best_box_none_when_all_below_threshold_or_empty():
    assert best_box([[0, 0, 10, 10]], [0.49], [0], 0.5) is None
    assert best_box(np.zeros((0, 4)), [], [], 0.5) is None


def test_best_box_ignores_other_classes():
    box, conf = best_box([[0, 0, 10, 10], [0, 0, 20, 20]], [0.99, 0.7], [1, 0], 0.5)
    assert conf == 0.7 and list(box) == [0, 0, 20, 20]


def test_pad_box_grows_by_fraction_of_box_size():
    assert pad_box([100, 100, 200, 150], 0.10, 1000, 1000) == (90, 95, 210, 155)


def test_pad_box_clamps_to_image():
    assert pad_box([5, 2, 395, 298], 0.10, 400, 300) == (0, 0, 400, 300)


def test_pad_box_zero_pad_and_degenerate_box():
    assert pad_box([10.4, 20.6, 30.2, 40.9], 0.0, 100, 100) == (10, 20, 31, 41)
    assert pad_box([500, 500, 600, 600], 0.1, 100, 100) is None  # entirely outside the image


# ---- mask helpers
def test_best_mask_picks_highest_conf_and_none_when_empty():
    a, b = np.zeros((4, 4)), np.ones((4, 4))
    assert (best_mask(np.stack([a, b]), [0.3, 0.8]) == b).all()
    assert best_mask(None, []) is None
    assert best_mask(np.zeros((0, 4, 4)), []) is None


def test_mask_crop_blacks_background_and_crops_to_mask_bbox():
    image = img(10, 8, value=200)
    mask = np.zeros((8, 10), dtype=np.float32)
    mask[2:5, 3:7] = 1.0      # 3 rows x 4 cols
    mask[2, 3] = 0.0          # one background pixel inside the bbox
    out = mask_crop(image, mask)
    assert out.size == (4, 3)
    arr = np.asarray(out)
    assert (arr[0, 0] == 0).all()           # background inside the bbox is black
    assert (arr[1, 1] == 200).all()         # mask pixels keep their colour


def test_mask_crop_resizes_mask_to_image_and_empty_is_none():
    image = img(20, 10)
    small = np.zeros((5, 10), dtype=np.float32)
    small[:, 5:] = 1.0                      # right half, at half resolution
    assert mask_crop(image, small).size == (10, 10)
    assert mask_crop(image, np.zeros((10, 20))) is None


# ---- detect()
def test_detect_one_batched_call_crops_and_returns_none_without_box():
    p = bare_pipeline()
    fake = with_model(p, "detector", FakeModel([dets([[100, 100, 200, 150]], [0.95]),
                                                dets([[0, 0, 50, 50]], [0.2]),
                                                dets(np.zeros((0, 4)), [])]))
    out = p.detect([img(), img(), img()], ["whole", "left_face", "right_face"])
    assert len(fake.calls) == 1 and fake.calls[0][0] == 3
    assert out[0].size == (120, 60)          # (90,95)-(210,155)
    assert out[1] is None and out[2] is None


def test_detect_accepts_bare_detections_for_single_image():
    p = bare_pipeline()
    with_model(p, "detector", FakeModel(dets([[0, 0, 400, 300]], [0.9])))
    out = p.detect([img()], ["whole"])
    assert out[0].size == (400, 300)


def test_detect_passes_through_without_model():
    imgs = [img(), img()]
    assert bare_pipeline().detect(imgs, ["whole", "left_face"]) is imgs


# ---- segment()
def test_segment_only_runs_on_faces_and_crops_mask():
    p = bare_pipeline()
    mask = np.zeros((300, 400), dtype=np.float32)
    mask[50:150, 100:300] = 1
    fake = with_model(p, "segmenter", FakeModel([seg_result([mask, np.ones((300, 400))], [0.9, 0.4])]))
    whole = img()
    out = p.segment([whole, img()], ["whole", "left_face"])
    assert len(fake.calls) == 1 and fake.calls[0][0] == 1   # only the face went to the model
    assert out[0] is whole
    assert out[1].size == (200, 100)
    assert fake.calls[0][1]["retina_masks"] is True


def test_segment_no_mask_returns_detection_crop_and_warns(caplog):
    p = bare_pipeline()
    with_model(p, "segmenter", FakeModel([seg_result(None, []), seg_result([np.zeros((300, 400))], [0.5])]))
    left, right = img(), img()
    with caplog.at_level("WARNING", logger="turtle.worker"):
        out = p.segment([left, right], ["left_face", "right_face"])
    assert out[0] is left and out[1] is right
    assert sum("no scute mask" in r.message for r in caplog.records) == 2


def test_segment_passes_through_without_model():
    imgs = [img()]
    assert bare_pipeline().segment(imgs, ["left_face"]) is imgs


# ---- versions
def test_versions_segmented_needs_seg_embed_version():
    assert check_versions(True, True, "dinov2-base-seg-v1", "yolo26s-seg-v1") == ("dinov2-base-seg-v1", "yolo26s-seg-v1")
    with pytest.raises(RuntimeError):
        check_versions(True, True, "dinov2-base-v1", "yolo26s-seg-v1")     # would mix with full-photo vectors
    with pytest.raises(RuntimeError):
        check_versions(True, True, "dinov2-base-seg-v1", "none")


def test_versions_without_segmenter_must_not_claim_seg():
    assert check_versions(False, False, "dinov2-base-v1", "none") == ("dinov2-base-v1", "none")  # CI / dev
    with pytest.raises(RuntimeError):
        check_versions(False, False, "dinov2-base-seg-v1", "none")
    with pytest.raises(RuntimeError):
        check_versions(False, False, "dinov2-base-v1", "yolo26s-seg-v1")


def test_versions_detection_only_must_not_reuse_full_photo_version():
    assert check_versions(True, False, "dinov2-base-det-v1", "none") == ("dinov2-base-det-v1", "none")
    with pytest.raises(RuntimeError):
        check_versions(True, False, "dinov2-base-v1", "none")              # crops vs old full photos
