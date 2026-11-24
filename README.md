# AI Content Studio

Streamlit app: detect AI-generated **text** and **images**, rewrite text with an LLM, inspect image
provenance/forensics, and apply classical image enhancement. All scores come from real models.

## Architecture
```
app.py                    UI only (Streamlit)
config.py                 env-driven settings, thresholds
modules/file_processor    validation + TXT/MD/PDF/DOCX/image extraction (in memory)
modules/text_detector     HF classifier per passage -> length-weighted score (+ style stats for context)
modules/text_humanizer    LLM rewrite via OpenRouter (free models) / Hugging Face / Anthropic, 5 styles, chunked
modules/image_detector    HF image classifier
modules/image_analyzer    EXIF, PNG text chunks, AI-tool keywords, C2PA/IPTC markers, ELA/noise/FFT
modules/image_enhancer    OpenCV: upscale, denoise, unsharp, CLAHE, gray-world; PNG tagged "enhanced"
modules/visualization     Plotly gauges/bars, highlighted passages
```

## Models and honest accuracy
| Task | Default model | Reality check |
|---|---|---|
| Text | `Hello-SimpleAI/chatgpt-detector-roberta` | English only; trained on ChatGPT-era text; weak on newer LLMs, paraphrased text, short text; false positives on formal/non-native writing. |
| Image | `umm-maybe/AI-image-detector` | Trained on older generators; misses many modern ones; sensitive to compression/edits. |
| Provenance | metadata + C2PA marker scan | Strong only when metadata *declares* AI. Marker is **not** cryptographically verified. Missing metadata proves nothing. |
| Forensics (ELA/noise/FFT) | OpenCV/NumPy | Exploratory visuals, not part of any score. |

Swap models with `TEXT_DETECTOR_MODEL` / `IMAGE_DETECTOR_MODEL`. Benchmark any candidate on **your** data before trusting it.
Not included: OCR for scanned PDFs, Arabic-capable detector, generative "realism" refinement.

## Setup (Python 3.11+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENROUTER_API_KEY (humanizer only)
streamlit run app.py
```
First run downloads the models from Hugging Face (~0.5 GB, needs internet).

## Tests
```bash
pytest -q tests                                   # offline logic tests
RUN_MODEL_TESTS=1 pytest tests/test_models_integration.py -s   # real models, prints scores
```

## Deployment
- **Streamlit Community Cloud:** push to GitHub, set `ANTHROPIC_API_KEY` (and `HF_TOKEN`) in app secrets/env. Free tier RAM is tight for torch; consider the Docker route.
- **Docker:** `docker build -t ai-studio . && docker run -p 8501:8501 --env-file .env ai-studio`
- Keys stay server-side in env vars; uploads are processed in memory only.
