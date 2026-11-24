"""Pixel-based AI-image classifier (Hugging Face image-classification model).

Known limit: public detectors are trained on specific generators and degrade on
newer ones, on edited/compressed images and on screenshots.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from config import AI_THRESHOLD, HUMAN_THRESHOLD, settings

AI_LABEL_KEYS = ("artificial", "fake", "ai", "generated", "synthetic")
HUMAN_LABEL_KEYS = ("human", "real", "natural")


@dataclass
class ImageDetection:
    ai_prob: float
    human_prob: float
    label: str
    confidence: str
    raw: list[dict]
    model: str


@lru_cache(maxsize=2)
def _load(model_name: str):
    from transformers import pipeline

    kwargs = {"token": settings.hf_token} if settings.hf_token else {}
    return pipeline("image-classification", model=model_name, **kwargs)


def _ai_prob(scores: list[dict]) -> float:
    by = {s["label"].lower(): float(s["score"]) for s in scores}
    for lab, v in by.items():
        if any(k == lab or lab.startswith(k) for k in AI_LABEL_KEYS):
            return v
    for lab, v in by.items():
        if any(k in lab for k in HUMAN_LABEL_KEYS):
            return 1 - v
    raise RuntimeError(f"Unrecognised detector labels: {list(by)}")


def detect_image(data: bytes) -> ImageDetection:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    raw = _load(settings.image_detector_model)(img, top_k=None)
    ai = _ai_prob(raw)
    if ai >= AI_THRESHOLD:
        label = "Potentially AI-generated"
    elif ai <= HUMAN_THRESHOLD:
        label = "Likely not AI-generated"
    else:
        label = "Uncertain"
    margin = abs(ai - 0.5) * 2
    conf = "High" if margin > 0.7 else "Medium" if margin > 0.4 else "Low"
    return ImageDetection(ai, 1 - ai, label, conf, raw, settings.image_detector_model)
