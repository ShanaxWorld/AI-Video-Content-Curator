"""Central configuration, read from environment variables (optionally a .env file).

Every external service is optional. When a key is missing the corresponding stage
falls back to a free/no-key or fully offline implementation, so the whole pipeline
can always run end to end.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    # ---- storage -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("CURATOR_DATA_DIR", "data")).resolve())

    # ---- LLM (any OpenAI-compatible endpoint; first one with a key wins) --
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_model: str = field(
        default_factory=lambda: _env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    )
    # Local Ollama (fully open-source, no key). Enabled when OLLAMA_MODEL is set.
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL"))

    # ---- trend signals / video search ------------------------------------
    youtube_api_key: str = field(default_factory=lambda: _env("YOUTUBE_API_KEY"))
    youtube_region: str = field(default_factory=lambda: _env("YOUTUBE_REGION", "US"))
    # Restrict curation to Creative-Commons licensed videos (YouTube Data API only).
    creative_commons_only: bool = field(
        default_factory=lambda: _env("CREATIVE_COMMONS_ONLY", "true").lower() in {"1", "true", "yes"}
    )

    # ---- speech-to-text -------------------------------------------------
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))
    groq_whisper_model: str = field(default_factory=lambda: _env("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"))

    # ---- text-to-speech -------------------------------------------------
    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "auto"))
    edge_voice: str = field(default_factory=lambda: _env("EDGE_TTS_VOICE", "en-US-AriaNeural"))

    # ---- image generation -------------------------------------------------
    pollinations_api_key: str = field(default_factory=lambda: _env("POLLINATIONS_API_KEY"))
    pollinations_model: str = field(default_factory=lambda: _env("POLLINATIONS_MODEL", "flux"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))
    hf_image_model: str = field(default_factory=lambda: _env("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"))

    # ---- rendering --------------------------------------------------------
    width: int = 1080
    height: int = 1920
    fps: int = 30
    font_path: str = field(default_factory=lambda: _env("FONT_PATH"))
    max_clip_seconds: int = field(default_factory=lambda: int(_env("MAX_CLIP_SECONDS", "90")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "curator.db"

    def dir(self, name: str) -> Path:
        p = self.data_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def status(self) -> dict[str, str]:
        """Human-readable summary of which providers are active."""
        from . import llm, tts

        return {
            "LLM": llm.active_provider_name(self),
            "Trend signals": "YouTube Data API" if self.youtube_api_key else "LLM knowledge only",
            "Video search": "YouTube Data API" + (" (CC only)" if self.creative_commons_only else "")
            if self.youtube_api_key
            else "yt-dlp search (no key)",
            "Transcription": "captions -> "
            + ("Groq Whisper" if self.groq_api_key else "local faster-whisper"),
            "TTS": tts.describe(self),
            "Images": "Pollinations"
            if self.pollinations_api_key
            else ("HF Inference (FLUX)" if self.hf_token else "procedural (offline)"),
            "ffmpeg": "found" if shutil.which("ffmpeg") else "MISSING",
        }


settings = Settings()
