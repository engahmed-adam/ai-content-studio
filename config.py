"""Central configuration. Secrets come from environment variables only."""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

TEXT_EXTS = {".txt", ".md", ".markdown"}
DOC_EXTS = {".pdf", ".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}

AI_THRESHOLD = 0.70      # >= -> "Likely AI-generated"
HUMAN_THRESHOLD = 0.30   # <= -> "Likely Human-written"


# Rewriting (humanizer) provider: "openrouter" (free models), "huggingface", or "anthropic".
_PROVIDER = os.getenv("LLM_PROVIDER") or (
    "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY") else "openrouter"
)
_DEFAULT_MODEL = {"openrouter": "openrouter/free", "huggingface": "openai/gpt-oss-120b", "anthropic": "claude-sonnet-5-5"}
OPENAI_COMPAT_URLS = {"openrouter": "https://openrouter.ai/api/v1", "huggingface": "https://router.huggingface.co/v1"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _PROVIDER
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL") or _DEFAULT_MODEL.get(_PROVIDER, "openrouter/free")
    hf_token: str = os.getenv("HF_TOKEN", "")
    # English-only detector trained on ChatGPT-era text. Swap via env var.
    text_detector_model: str = os.getenv("TEXT_DETECTOR_MODEL", "Hello-SimpleAI/chatgpt-detector-roberta")
    # Trained on older generators; weak on newest models. Swap via env var.
    image_detector_model: str = os.getenv("IMAGE_DETECTOR_MODEL", "umm-maybe/AI-image-detector")
    max_file_mb: int = int(os.getenv("MAX_FILE_MB", "15"))
    max_text_chars: int = int(os.getenv("MAX_TEXT_CHARS", "50000"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "150"))

    @property
    def llm_key(self) -> str:
        return {"openrouter": self.openrouter_api_key, "huggingface": self.hf_token,
                "anthropic": self.anthropic_api_key}.get(self.llm_provider, "")


settings = Settings()
