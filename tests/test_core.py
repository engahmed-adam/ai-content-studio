import io

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules.file_processor import FileError, from_pasted, process_upload
from modules.image_analyzer import analyze_image
from modules.image_enhancer import enhance, to_png_bytes
from modules.text_detector import make_segments, stylometry, _ai_probability, _classify, Segment, LABEL_MIXED, LABEL_AI, LABEL_HUMAN
from modules import text_humanizer

LOREM = " ".join(f"This is sentence number {i} written for testing purposes only." for i in range(40))


def _png(w=128, h=128, text=None, extra=b""):
    arr = (np.random.default_rng(0).random((h, w, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    info = None
    if text:
        info = PngInfo()
        for k, v in text.items():
            info.add_text(k, v)
    Image.fromarray(arr).save(buf, "PNG", pnginfo=info)
    return buf.getvalue() + extra


# ---- file processing
def test_txt_and_md():
    assert process_upload("a.txt", b"hello world").text == "hello world"
    assert process_upload("a.md", "# مرحبا".encode()).kind == "text"


def test_docx_roundtrip():
    import docx
    d = docx.Document(); d.add_paragraph("Hello docx"); buf = io.BytesIO(); d.save(buf)
    assert "Hello docx" in process_upload("a.docx", buf.getvalue()).text


def test_pdf_roundtrip():
    import pymupdf as fitz
    doc = fitz.open(); doc.new_page().insert_text((72, 72), "Hello pdf"); data = doc.tobytes()
    assert "Hello pdf" in process_upload("a.pdf", data).text


def test_pdf_without_text_rejected():
    import pymupdf as fitz
    doc = fitz.open(); doc.new_page()
    with pytest.raises(FileError):
        process_upload("a.pdf", doc.tobytes())


@pytest.mark.parametrize("name,data", [
    ("a.exe", b"x"), ("a.txt", b""), ("a.pdf", b"not a pdf"), ("a.docx", b"not zip"),
    ("a.png", b"not an image"),
])
def test_bad_files(name, data):
    with pytest.raises(FileError):
        process_upload(name, data)


def test_size_limit():
    with pytest.raises(FileError):
        process_upload("a.txt", b"x" * (16 * 1024 * 1024))


def test_empty_paste():
    with pytest.raises(FileError):
        from_pasted("   ")


# ---- text logic (no model needed)
def test_segments_cover_text():
    segs = make_segments(LOREM)
    assert len(segs) > 1
    assert sum(len(s.split()) for s in segs) == len(LOREM.split())


def test_stylometry_uniform_text_low_variation():
    assert stylometry(LOREM)["sentence_len_variation"] < 0.1


def test_label_mapping():
    assert _ai_probability([{"label": "Human", "score": 0.8}, {"label": "ChatGPT", "score": 0.2}]) == pytest.approx(0.2)
    assert _ai_probability([{"label": "Real", "score": 0.1}, {"label": "Fake", "score": 0.9}]) == pytest.approx(0.9)


def test_classification_rules():
    hi = [Segment("a", 0.9, 50)] * 4
    lo = [Segment("a", 0.1, 50)] * 4
    assert _classify(0.9, hi, 200) == LABEL_AI
    assert _classify(0.1, lo, 200) == LABEL_HUMAN
    assert _classify(0.5, hi[:2] + lo[:2], 200) == LABEL_MIXED
    assert _classify(0.95, hi, 20) == LABEL_MIXED      # too short


# ---- image analysis
def test_image_metadata_detects_generator_params():
    data = _png(text={"parameters": "a cat\nNegative prompt: blur\nSteps: 20, Sampler: Euler, Model: SDXL"})
    a = analyze_image(data)
    assert a.ai_tool_hits and any(o[0] == "strong" for o in a.observations)


def test_image_declared_ai_and_c2pa_markers():
    a = analyze_image(_png(extra=b"trainedAlgorithmicMedia c2pa"))
    assert a.declared_ai_source and a.c2pa_marker


def test_no_exif_is_only_weak():
    a = analyze_image(_png())
    assert not a.declared_ai_source
    assert all(o[0] != "strong" for o in a.observations)
    assert a.ela_image is not None and "noise_std" in a.forensics


# ---- enhancement
def test_enhancer_and_tag():
    img = Image.open(io.BytesIO(_png(64, 48)))
    out = enhance(img, scale=2, denoise=5, sharpen=0.5, auto_light=True, auto_color=True)
    assert out.size == (128, 96)
    assert "Not an original photograph" in Image.open(io.BytesIO(to_png_bytes(out))).info["Processing"]


# ---- humanizer guards
def test_humanizer_requires_key():
    with pytest.raises(text_humanizer.HumanizerError):
        text_humanizer.humanize("hi", "Casual", api_key="")
