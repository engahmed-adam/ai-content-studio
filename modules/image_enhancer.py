"""Classical (non-generative) image enhancement with OpenCV.

Outputs are tagged in PNG metadata as enhanced and must not be presented as
original photographs.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

MAX_SIDE = 4096


def _to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def gray_world(bgr: np.ndarray) -> np.ndarray:
    means = bgr.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def clahe_lighting(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance(
    img: Image.Image,
    scale: int = 1,
    denoise: int = 0,
    sharpen: float = 0.0,
    auto_light: bool = False,
    auto_color: bool = False,
) -> Image.Image:
    bgr = _to_bgr(img)
    if denoise > 0:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, denoise, denoise, 7, 21)
    if auto_color:
        bgr = gray_world(bgr)
    if auto_light:
        bgr = clahe_lighting(bgr)
    if scale > 1:
        h, w = bgr.shape[:2]
        k = min(scale, MAX_SIDE / max(h, w))
        if k > 1:
            bgr = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_LANCZOS4)
    if sharpen > 0:
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
        bgr = cv2.addWeighted(bgr, 1 + sharpen, blur, -sharpen, 0)
    return _to_pil(bgr)


def to_png_bytes(img: Image.Image) -> bytes:
    meta = PngInfo()
    meta.add_text("Processing", "Enhanced with AI Content Studio (classical filters). Not an original photograph.")
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=meta)
    return buf.getvalue()
