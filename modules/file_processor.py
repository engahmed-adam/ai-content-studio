"""Validation + extraction for uploaded files. Nothing is written to disk."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from config import ALLOWED_IMAGE_FORMATS, DOC_EXTS, IMAGE_EXTS, TEXT_EXTS, settings


class FileError(ValueError):
    """User-facing validation/extraction problem."""


@dataclass
class ProcessedFile:
    name: str
    kind: str                      # "text" | "image"
    text: str = ""
    image_bytes: bytes = b""
    notes: list[str] = field(default_factory=list)


def _clean(text: str, notes: list[str]) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise FileError("No readable text was found in this file.")
    if len(text) > settings.max_text_chars:
        text = text[: settings.max_text_chars]
        notes.append(f"Text truncated to the first {settings.max_text_chars:,} characters.")
    return text


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise FileError("Could not decode this text file.")


def _read_pdf(data: bytes, notes: list[str]) -> str:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:
        import fitz

    if not data.startswith(b"%PDF"):
        raise FileError("This file is not a valid PDF.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise FileError("The PDF is corrupted or unreadable.") from exc
    if doc.needs_pass:
        raise FileError("The PDF is password-protected.")
    pages = []
    total = 0
    for i, page in enumerate(doc):
        if i >= settings.max_pdf_pages:
            notes.append(f"Only the first {settings.max_pdf_pages} pages were read.")
            break
        t = page.get_text()
        pages.append(t)
        total += len(t)
        if total > settings.max_text_chars:
            break
    text = "\n\n".join(pages)
    if not text.strip():
        raise FileError("No extractable text found. This looks like a scanned PDF (OCR is not included).")
    return text


def _read_docx(data: bytes) -> str:
    import docx

    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise FileError("This file is not a valid DOCX.")
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise FileError("The DOCX is corrupted or unreadable.") from exc
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _check_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Image.DecompressionBombError as exc:
        raise FileError("The image has too many pixels to process safely.") from exc
    except Exception as exc:
        raise FileError("The image is corrupted or unreadable.") from exc
    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise FileError(f"Unsupported image format: {fmt}.")


def process_upload(name: str, data: bytes) -> ProcessedFile:
    ext = Path(name).suffix.lower()
    if not data:
        raise FileError("The file is empty.")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise FileError(f"File is larger than {settings.max_file_mb} MB.")
    notes: list[str] = []
    if ext in TEXT_EXTS:
        return ProcessedFile(name, "text", text=_clean(_decode(data), notes), notes=notes)
    if ext == ".pdf":
        return ProcessedFile(name, "text", text=_clean(_read_pdf(data, notes), notes), notes=notes)
    if ext == ".docx":
        return ProcessedFile(name, "text", text=_clean(_read_docx(data), notes), notes=notes)
    if ext in IMAGE_EXTS:
        _check_image(data)
        return ProcessedFile(name, "image", image_bytes=data, notes=notes)
    allowed = ", ".join(sorted(TEXT_EXTS | DOC_EXTS | IMAGE_EXTS))
    raise FileError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {allowed}")


def from_pasted(text: str) -> ProcessedFile:
    notes: list[str] = []
    return ProcessedFile("pasted_text", "text", text=_clean(text, notes), notes=notes)
