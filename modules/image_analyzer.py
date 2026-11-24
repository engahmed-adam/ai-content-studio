"""Image metadata, provenance markers and simple forensic metrics.

Everything here is *evidence to inspect*, not a verdict. Missing metadata proves
nothing (social platforms strip it); present metadata can be edited.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageChops

AI_TOOL_KEYWORDS = [
    "midjourney", "stable diffusion", "stable-diffusion", "dall-e", "dall·e", "dalle",
    "firefly", "comfyui", "automatic1111", "sdxl", "novelai", "negative prompt",
    "openai", "imagen", "flux", "leonardo.ai", "ideogram", "runway", "sora",
]


@dataclass
class ImageAnalysis:
    info: dict
    exif: dict
    text_chunks: dict
    gps_present: bool
    camera_exif: bool
    ai_tool_hits: list[str]
    c2pa_marker: bool
    declared_ai_source: bool
    forensics: dict
    observations: list[tuple[str, str]]       # (strength, message)
    ela_image: Image.Image | None = None


def _exif(img: Image.Image) -> tuple[dict, bool]:
    out, gps = {}, False
    try:
        ex = img.getexif()
        items = dict(ex)
        try:
            items.update(ex.get_ifd(0x8769))
        except Exception:
            pass
        gps = 0x8825 in ex
        for tag, val in items.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            if name in {"GPSInfo", "MakerNote"}:
                continue
            if isinstance(val, bytes):
                val = val[:200].decode("utf-8", errors="replace")
            out[name] = str(val)[:300]
    except Exception:
        pass
    return out, gps


def _forensics(img: Image.Image) -> tuple[dict, Image.Image]:
    rgb = img.convert("RGB")
    rgb.thumbnail((1536, 1536))
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=90)
    buf.seek(0)
    diff = np.asarray(ImageChops.difference(rgb, Image.open(buf).convert("RGB")), dtype=np.float32)
    ela_vis = Image.fromarray(np.clip(diff * (255.0 / max(diff.max(), 1.0)), 0, 255).astype(np.uint8))

    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    noise_std = float((gray - cv2.GaussianBlur(gray, (5, 5), 0)).std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - 128, xx - 128)
    hf_ratio = float(mag[r > 64].sum() / max(mag.sum(), 1e-9))

    return {
        "ela_mean": round(float(diff.mean()), 2),
        "noise_std": round(noise_std, 2),
        "laplacian_variance": round(lap_var, 1),
        "high_freq_energy_ratio": round(hf_ratio, 3),
    }, ela_vis


def analyze_image(data: bytes) -> ImageAnalysis:
    img = Image.open(io.BytesIO(data))
    img.load()
    exif, gps = _exif(img)

    chunks = {}
    for k, v in img.info.items():
        if isinstance(v, (str, bytes)) and k not in {"icc_profile", "exif"}:
            s = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            chunks[str(k)] = s
    haystack = " ".join(list(chunks.keys()) + list(chunks.values()) + list(exif.values())).lower()
    hits = sorted({kw for kw in AI_TOOL_KEYWORDS if kw in haystack})
    if "parameters" in chunks and "steps" in chunks["parameters"].lower():
        hits.append("A1111-style generation parameters")

    raw = data
    c2pa = b"c2pa" in raw
    declared = b"trainedAlgorithmicMedia" in raw or b"compositeWithTrainedAlgorithmicMedia" in raw
    camera = bool(exif.get("Make") or exif.get("Model"))

    forensics, ela = _forensics(img)
    w, h = img.size
    obs: list[tuple[str, str]] = []
    if declared:
        obs.append(("strong", "IPTC/XMP metadata declares AI-generated media (trainedAlgorithmicMedia)."))
    if hits:
        obs.append(("strong", f"Metadata names generation tools or parameters: {', '.join(hits)}."))
    if c2pa:
        obs.append(("info", "A C2PA/Content Credentials marker is present. It is NOT cryptographically "
                            "verified here; check it with c2patool or contentcredentials.org/verify."))
    if camera:
        obs.append(("info", "Camera make/model EXIF is present. This can be edited, so it is not proof of authenticity."))
    else:
        obs.append(("weak", "No camera EXIF. Very common after social-media upload or export, so this proves nothing."))
    if w >= 512 and h >= 512 and w % 64 == 0 and h % 64 == 0:
        obs.append(("weak", f"Dimensions {w}×{h} are multiples of 64, typical of diffusion outputs but also common elsewhere."))
    if forensics["noise_std"] < 1.0:
        obs.append(("weak", "Very low high-frequency noise (unusually smooth). Seen in AI images and in heavily denoised or compressed photos."))
    obs.append(("info", "ELA, noise and frequency metrics are exploratory visual aids, not detectors."))

    return ImageAnalysis(
        info={"format": img.format, "size": f"{w}×{h}", "mode": img.mode, "file_kb": round(len(data) / 1024, 1)},
        exif=exif, text_chunks={k: v[:300] for k, v in chunks.items()}, gps_present=gps,
        camera_exif=camera, ai_tool_hits=hits, c2pa_marker=c2pa, declared_ai_source=declared,
        forensics=forensics, observations=obs, ela_image=ela,
    )
