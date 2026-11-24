"""AI Content Studio - Streamlit front end. Detection/processing live in modules/."""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from config import IMAGE_EXTS, TEXT_EXTS, DOC_EXTS, settings
from modules import visualization as viz
from modules.file_processor import FileError, from_pasted, process_upload
from modules.image_analyzer import analyze_image
from modules.image_detector import detect_image
from modules.image_enhancer import enhance, to_png_bytes
from modules.text_detector import LABEL_AI, LABEL_HUMAN, detect_text
from modules.text_humanizer import STYLES, HumanizerError, humanize

st.set_page_config(page_title="AI Content Studio", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.hero h1{font-size:2.3rem;margin-bottom:.2rem;background:linear-gradient(90deg,#6366f1,#06b6d4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{opacity:.75;margin-top:0}
.card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:1rem 1.2rem;
background:rgba(128,128,128,.06);margin-bottom:.8rem}
.pill{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-weight:600;font-size:.9rem}
.pill.ai{background:rgba(248,81,73,.18);color:#f85149}
.pill.human{background:rgba(46,160,67,.18);color:#2ea043}
.pill.mixed{background:rgba(210,153,34,.18);color:#d29922}
</style>
""", unsafe_allow_html=True)


def pill(label: str) -> str:
    cls = "ai" if label.startswith(("Likely AI", "Potentially")) else \
          "human" if label.startswith(("Likely Human", "Likely not")) else "mixed"
    return f'<span class="pill {cls}">{label}</span>'


def sidebar() -> None:
    with st.sidebar:
        st.header("About")
        st.caption("Detectors give probabilities, not proof. Never use a score alone to accuse "
                   "someone of cheating or fabricating an image.")
        st.markdown(f"**Text model:** `{settings.text_detector_model}`")
        st.markdown(f"**Image model:** `{settings.image_detector_model}`")
        st.markdown(f"**Rewrite:** `{settings.llm_provider}` / `{settings.llm_model}`")
        if not settings.llm_key:
            st.warning(f"No API key for '{settings.llm_provider}': the humanizer is disabled (see .env.example).")
        st.divider()
        st.caption("Light/dark: use the ⋮ menu → Settings → Theme.")
        st.caption("Files are processed in memory and never saved to disk.")


# ------------------------------------------------------------------ text
def render_text(pf) -> None:
    key = hashlib.sha256(pf.text.encode()).hexdigest()[:16]
    for n in pf.notes:
        st.info(n)
    with st.expander("File / text preview", expanded=False):
        st.text(pf.text[:3000] + ("…" if len(pf.text) > 3000 else ""))

    if st.button("🔍 Analyze text", type="primary", key=f"an_{key}"):
        bar = st.progress(0.0, text="Loading model and scoring passages…")
        try:
            st.session_state[f"rep_{key}"] = detect_text(pf.text, lambda p: bar.progress(p))
        except Exception as exc:  # model download/inference problems
            st.error(f"Detection failed: {exc}")
        finally:
            bar.empty()

    rep = st.session_state.get(f"rep_{key}")
    if not rep:
        return

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(viz.gauge(rep.ai_prob, "AI likelihood"), use_container_width=True)
    with c2:
        st.markdown(f'<div class="card"><b>Classification</b><br>{pill(rep.label)}<br><br>'
                    f'AI: <b>{rep.ai_prob:.0%}</b> &nbsp;|&nbsp; Human: <b>{rep.human_prob:.0%}</b></div>',
                    unsafe_allow_html=True)
        for w in rep.warnings:
            st.caption(f"⚠️ {w}")

    st.subheader("Why this result")
    for f in rep.factors:
        st.markdown(f"- {f}")
    st.dataframe(pd.DataFrame([rep.stats]).T.rename(columns={0: "value"}), use_container_width=True)

    st.subheader("Suspicious passages")
    st.plotly_chart(viz.segment_bar(rep.segments), use_container_width=True)
    st.markdown(viz.highlight_html(rep.segments), unsafe_allow_html=True)

    st.divider()
    st.subheader("✍️ Humanizer")
    st.caption("Improves naturalness and variety. It does not guarantee any detector result.")
    if rep.label == LABEL_HUMAN:
        st.info("This text already reads as human-written. You can still rewrite it for style.")
    style = st.selectbox("Writing style", list(STYLES), index=1, key=f"st_{key}")
    if st.button("Rewrite", disabled=not settings.llm_key, key=f"hm_{key}"):
        with st.spinner("Rewriting…"):
            try:
                st.session_state[f"hum_{key}"] = humanize(pf.text, style)
            except HumanizerError as exc:
                st.error(str(exc))
    out = st.session_state.get(f"hum_{key}")
    if out:
        a, b = st.columns(2)
        a.markdown("**Before**")
        a.text_area("before", pf.text, height=320, disabled=True, label_visibility="collapsed")
        b.markdown("**After**")
        b.text_area("after", out, height=320, label_visibility="collapsed")
        st.download_button("⬇️ Download rewritten text", out, "rewritten.txt", "text/plain")
        if st.button("Re-check rewritten text", key=f"rc_{key}"):
            with st.spinner("Scoring rewritten text…"):
                r2 = detect_text(out)
            st.metric("AI likelihood after rewrite", f"{r2.ai_prob:.0%}", f"{(r2.ai_prob - rep.ai_prob):+.0%}",
                      delta_color="inverse")
            st.caption("A lower score reflects this one detector only; other tools may disagree.")


# ----------------------------------------------------------------- image
def render_image(pf) -> None:
    key = hashlib.sha256(pf.image_bytes).hexdigest()[:16]
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(pf.image_bytes)))
    left, right = st.columns([1, 1.2])
    left.image(img, caption=pf.name, use_container_width=True)

    if right.button("🔍 Analyze image", type="primary", key=f"ia_{key}"):
        with st.spinner("Running detector and forensics…"):
            try:
                st.session_state[f"idet_{key}"] = detect_image(pf.image_bytes)
            except Exception as exc:
                st.session_state[f"idet_{key}"] = None
                st.error(f"Detector unavailable: {exc}")
            st.session_state[f"iana_{key}"] = analyze_image(pf.image_bytes)

    det, ana = st.session_state.get(f"idet_{key}"), st.session_state.get(f"iana_{key}")
    if ana is None:
        return

    with right:
        if det:
            st.plotly_chart(viz.gauge(det.ai_prob, "AI-generation likelihood"), use_container_width=True)
            st.markdown(f'<div class="card">{pill(det.label)}<br>Confidence: <b>{det.confidence}</b></div>',
                        unsafe_allow_html=True)
            st.caption("⚠️ Pixel-based detectors miss many modern generators and are fooled by edits "
                       "and compression. This is not proof either way.")
    if ana.declared_ai_source or ana.ai_tool_hits:
        st.error("Provenance: metadata declares or names an AI generation source (see below).")

    t1, t2, t3 = st.tabs(["Provenance & metadata", "Visual forensics", "Enhance"])
    with t1:
        icon = {"strong": "🔴", "info": "🔵", "weak": "⚪"}
        for strength, msg in ana.observations:
            st.markdown(f"{icon[strength]} **{strength.title()}** — {msg}")
        st.json(ana.info)
        if ana.exif:
            with st.expander("EXIF"):
                st.json(ana.exif)
        if ana.text_chunks:
            with st.expander("Embedded text chunks"):
                st.json(ana.text_chunks)
        if ana.gps_present:
            st.caption("📍 GPS data is present in this file (not displayed).")
    with t2:
        st.dataframe(pd.DataFrame([ana.forensics]).T.rename(columns={0: "value"}), use_container_width=True)
        st.image(ana.ela_image, caption="Error Level Analysis (exploratory; bright areas = higher "
                 "recompression difference)", use_container_width=True)
    with t3:
        st.caption("Classical filters only. Output is tagged as enhanced and is not an original photograph.")
        c1, c2 = st.columns(2)
        scale = c1.select_slider("Upscale", [1, 2, 3, 4], value=1)
        denoise = c1.slider("Noise reduction", 0, 15, 0)
        sharp = c2.slider("Sharpness", 0.0, 2.0, 0.0, 0.1)
        light = c2.checkbox("Auto lighting (CLAHE)")
        color = c2.checkbox("Auto color balance")
        if st.button("Apply enhancements", key=f"en_{key}"):
            with st.spinner("Enhancing…"):
                st.session_state[f"enh_{key}"] = enhance(img, scale, denoise, sharp, light, color)
        res = st.session_state.get(f"enh_{key}")
        if res:
            a, b = st.columns(2)
            a.image(img, caption="Original", use_container_width=True)
            b.image(res, caption=f"Enhanced ({res.size[0]}×{res.size[1]})", use_container_width=True)
            st.download_button("⬇️ Download enhanced PNG", to_png_bytes(res), "enhanced.png", "image/png")


# ------------------------------------------------------------------ main
def main() -> None:
    sidebar()
    st.markdown('<div class="hero"><h1>AI Content Studio</h1>'
                '<p>Detect AI-generated text and images, rewrite text more naturally, inspect image provenance.</p></div>',
                unsafe_allow_html=True)

    up_tab, paste_tab = st.tabs(["📁 Upload file", "✍️ Paste text"])
    pf = None
    with up_tab:
        exts = sorted(e.lstrip(".") for e in TEXT_EXTS | DOC_EXTS | IMAGE_EXTS)
        f = st.file_uploader(f"Drag & drop (max {settings.max_file_mb} MB)", type=exts)
        if f:
            try:
                pf = process_upload(f.name, f.getvalue())
            except FileError as exc:
                st.error(str(exc))
    with paste_tab:
        pasted = st.text_area("Paste text", height=220, placeholder="Paste at least ~250 words for a meaningful result…")
        if pasted.strip() and pf is None:
            try:
                pf = from_pasted(pasted)
            except FileError as exc:
                st.error(str(exc))

    if pf is None:
        st.info("Upload a file or paste text to begin. Uploaded files take precedence over pasted text.")
    elif pf.kind == "text":
        render_text(pf)
    else:
        render_image(pf)


main()
