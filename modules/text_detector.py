"""AI-text detection using a Hugging Face sequence classifier.

The score comes ONLY from the classifier. Stylometric statistics are shown as
context for the user and are never mixed into the probability.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from config import AI_THRESHOLD, HUMAN_THRESHOLD, settings

LABEL_AI = "Likely AI-generated"
LABEL_HUMAN = "Likely Human-written"
LABEL_MIXED = "Uncertain / Mixed"
HUMAN_LABELS = {"human", "real", "label_0"}

_SENT_RE = re.compile(r"(?<=[.!?؟…])\s+|\n{2,}")


@dataclass
class Segment:
    text: str
    ai_prob: float
    words: int


@dataclass
class TextReport:
    ai_prob: float
    human_prob: float
    label: str
    segments: list[Segment]
    factors: list[str]
    stats: dict
    warnings: list[str]
    model: str


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]


def make_segments(text: str, min_words: int = 35, max_words: int = 160) -> list[str]:
    """Group sentences into passages; classifiers are unreliable on tiny inputs."""
    segs: list[str] = []
    cur: list[str] = []
    n = 0
    for s in split_sentences(text):
        w = len(s.split())
        if cur and n + w > max_words:
            segs.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
        if n >= min_words:
            segs.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if segs and n < min_words // 2:
            segs[-1] += " " + " ".join(cur)
        else:
            segs.append(" ".join(cur))
    return segs


def stylometry(text: str) -> dict:
    lens = [len(s.split()) for s in split_sentences(text) if s.split()]
    words = re.findall(r"\w+", text.lower())
    mean = statistics.mean(lens) if lens else 0.0
    sd = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    openers = [s.split()[0].lower() for s in split_sentences(text) if s.split()]
    rep = 1 - len(set(openers)) / len(openers) if openers else 0.0
    sample = words[:300]
    return {
        "words": len(words),
        "sentences": len(lens),
        "avg_sentence_len": round(mean, 1),
        "sentence_len_variation": round(sd / mean, 2) if mean else 0.0,
        "vocab_diversity_300w": round(len(set(sample)) / len(sample), 2) if sample else 0.0,
        "repeated_sentence_openers": round(rep, 2),
    }


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str):
    from transformers import pipeline

    kwargs = {"token": settings.hf_token} if settings.hf_token else {}
    return pipeline("text-classification", model=model_name, **kwargs)


def _ai_probability(scores: list[dict]) -> float:
    by = {s["label"].lower(): float(s["score"]) for s in scores}
    for lab, val in by.items():
        if lab in HUMAN_LABELS:
            return 1.0 - val
    for lab, val in by.items():
        if lab in {"label_1", "chatgpt", "fake", "ai", "machine"}:
            return val
    raise RuntimeError(f"Unrecognised detector labels: {list(by)}")


def _latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() for c in letters) / len(letters)


def _classify(ai: float, segments: list[Segment], words: int) -> str:
    if words < 50:
        return LABEL_MIXED
    flagged = sum(s.ai_prob >= AI_THRESHOLD for s in segments) / max(len(segments), 1)
    if ai >= AI_THRESHOLD and flagged >= 0.5:
        return LABEL_AI
    if ai <= HUMAN_THRESHOLD and flagged <= 0.25:
        return LABEL_HUMAN
    return LABEL_MIXED


def detect_text(
    text: str,
    on_progress: Optional[Callable[[float], None]] = None,
    batch_size: int = 8,
) -> TextReport:
    model = settings.text_detector_model
    pieces = make_segments(text)
    if not pieces:
        raise ValueError("Nothing to analyse.")
    clf = _load_pipeline(model)

    segments: list[Segment] = []
    for i in range(0, len(pieces), batch_size):
        batch = pieces[i : i + batch_size]
        outs = clf(batch, top_k=None, truncation=True, max_length=512, batch_size=batch_size)
        for t, scores in zip(batch, outs):
            segments.append(Segment(t, _ai_probability(scores), len(t.split())))
        if on_progress:
            on_progress(min(1.0, (i + len(batch)) / len(pieces)))

    total_words = sum(s.words for s in segments) or 1
    ai = sum(s.ai_prob * s.words for s in segments) / total_words
    stats = stylometry(text)
    label = _classify(ai, segments, stats["words"])

    warnings = [
        "AI detection is probabilistic. Treat this as one signal, never as proof of authorship."
    ]
    if stats["words"] < 50:
        warnings.append("Very short text: results are unreliable below ~50 words (ideally 250+).")
    if _latin_ratio(text) < 0.7:
        warnings.append(
            "The selected detector was trained on English text. Results for Arabic or other "
            "languages are not reliable."
        )
    warnings.append("Formal, edited, or non-native-English writing is often falsely flagged as AI.")

    flagged = sum(s.ai_prob >= AI_THRESHOLD for s in segments)
    factors = [
        f"Classifier ({model}) rated {flagged} of {len(segments)} passages as likely AI "
        f"(≥{int(AI_THRESHOLD * 100)}%); length-weighted average {ai:.0%}."
    ]
    cv = stats["sentence_len_variation"]
    if stats["sentences"] >= 8:
        if cv < 0.35:
            factors.append(f"Sentence lengths are very uniform (variation {cv}). Common in LLM output, "
                           "but also in formal or templated writing.")
        elif cv > 0.6:
            factors.append(f"Sentence lengths vary a lot (variation {cv}), which is typical of human writing.")
    if stats["repeated_sentence_openers"] > 0.35 and stats["sentences"] >= 8:
        factors.append("Many sentences start with the same word, a repetitive pattern.")
    factors.append("Style metrics are shown for context only; they are not part of the score.")

    return TextReport(ai, 1 - ai, label, segments, factors, stats, warnings, model)
