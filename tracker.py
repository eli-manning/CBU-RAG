"""
Whole-body person tracking for Lancer, running on the Mac.

The robot's own detector (YuNet) finds faces only, so it loses people who are
too close, turned away, or off to the side. YOLO sees the whole person, which
holds up in all three cases. The robot posts frames here and aims at what comes
back; if this service is unreachable it falls back to YuNet on-device.

Aim point: the head when the pose is confident enough to locate it, otherwise
the upper body -- which is what you want anyway when someone is very close.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_NAME = "yolo11n.pt"
PERSON_CLASS = 0
MIN_CONFIDENCE = 0.35
IMG_SIZE = 480
# Fraction down the person's box to aim at. A head sits near the top; aiming at
# the box centre makes Lancer stare at people's chests.
HEAD_FRACTION = 0.12

_model = None
_stats = {"frames": 0, "hits": 0, "last_ms": 0.0}


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(MODEL_NAME)
        logger.info("YOLO tracker loaded (%s)", MODEL_NAME)
    return _model


def _score(box, width: int, height: int) -> float:
    """Prefer the big, central person -- the one most likely talking to Lancer."""
    x1, y1, x2, y2 = box
    area = max(x2 - x1, 1) * max(y2 - y1, 1)
    cx = (x1 + x2) / 2
    centrality = 1.0 - abs(cx - width / 2) / (width / 2)
    return area / (width * height) + 0.35 * centrality


@router.post("/api/track")
async def track(file: UploadFile = File(...)) -> dict:
    """Locate the most relevant person in a frame and return an aim point."""
    started = time.perf_counter()
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return {"found": False, "error": f"decode failed: {e}"}

    frame = np.asarray(image)
    height, width = frame.shape[:2]

    try:
        results = _get_model().predict(
            frame, verbose=False, classes=[PERSON_CLASS],
            imgsz=IMG_SIZE, conf=MIN_CONFIDENCE,
        )
    except Exception as e:
        logger.warning("YOLO inference failed: %s", e)
        return {"found": False, "error": str(e)}

    boxes = results[0].boxes
    _stats["frames"] += 1
    _stats["last_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if boxes is None or len(boxes) == 0:
        return {"found": False, "width": width, "height": height,
                "ms": _stats["last_ms"]}

    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    best = max(range(len(xyxy)), key=lambda i: _score(xyxy[i], width, height))
    x1, y1, x2, y2 = xyxy[best]

    u = (x1 + x2) / 2.0
    v = y1 + (y2 - y1) * HEAD_FRACTION
    # Very close subjects get cropped at the top of frame; clamp so the aim
    # point stays inside the image instead of driving the head past the person.
    v = float(min(max(v, 0.0), height - 1))
    u = float(min(max(u, 0.0), width - 1))

    _stats["hits"] += 1
    return {
        "found": True,
        "u": u, "v": v,
        "x_norm": (u / width) * 2 - 1,
        "y_norm": (v / height) * 2 - 1,
        "bbox": [x1, y1, x2, y2],
        "confidence": float(confs[best]),
        "people": len(xyxy),
        "width": width, "height": height,
        "ms": _stats["last_ms"],
    }


@router.get("/api/track/stats")
async def track_stats() -> dict:
    return {**_stats, "model": MODEL_NAME, "loaded": _model is not None}
