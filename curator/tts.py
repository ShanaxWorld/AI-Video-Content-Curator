"""Text-to-speech with graceful fallbacks.

auto order: edge-tts (free neural voices, no key) -> Kokoro-82M (open-source, local,
optional install) -> gTTS (free) -> espeak-ng (open-source, offline) -> silent track
timed at a natural speaking rate (keeps the video pipeline working fully offline).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

from .config import Settings, settings

log = logging.getLogger(__name__)

EDGE_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AndrewNeural",
    "en-GB-SoniaNeural",
    "en-IN-NeerjaNeural",
]

WORDS_PER_SECOND = 2.6  # ~155 wpm, typical narration pace


def _kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401

        return True
    except Exception:
        return False


def chain(cfg: Settings = settings) -> list[str]:
    if cfg.tts_provider and cfg.tts_provider != "auto":
        return [cfg.tts_provider, "silent"]
    order = ["edge", "kokoro", "gtts", "espeak", "silent"]
    if not _kokoro_available():
        order.remove("kokoro")
    if not shutil.which("espeak-ng"):
        order.remove("espeak")
    return order


def describe(cfg: Settings = settings) -> str:
    return " -> ".join(chain(cfg))


def probe(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- engines

def _edge(text: str, out: Path, voice: str) -> Path:
    import edge_tts

    async def run():
        await edge_tts.Communicate(text, voice).save(str(out))

    try:
        asyncio.run(run())
    except RuntimeError:  # already inside an event loop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run())
        loop.close()
    return out


def _kokoro(text: str, out: Path) -> Path:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    chunks = [audio for _, _, audio in pipe(text, voice="af_heart")]
    wav = out.with_suffix(".wav")
    sf.write(wav, np.concatenate(chunks), 24000)
    return wav


def _gtts(text: str, out: Path) -> Path:
    from gtts import gTTS

    gTTS(text=text, lang="en").save(str(out))
    return out


def _espeak(text: str, out: Path) -> Path:
    wav = out.with_suffix(".wav")
    subprocess.run(["espeak-ng", "-s", "165", "-w", str(wav), text], check=True, capture_output=True)
    return wav


def _silent(text: str, out: Path) -> Path:
    dur = max(1.5, len(text.split()) / WORDS_PER_SECOND + 0.4)
    wav = out.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", f"{dur:.2f}", str(wav)],
        check=True,
    )
    return wav


def synthesize(text: str, out: Path, voice: str | None = None, cfg: Settings = settings) -> tuple[Path, float, str]:
    """Return (audio_path, duration_seconds, engine_used)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    voice = voice or cfg.edge_voice
    engines = {
        "edge": lambda: _edge(text, out.with_suffix(".mp3"), voice),
        "kokoro": lambda: _kokoro(text, out),
        "gtts": lambda: _gtts(text, out.with_suffix(".mp3")),
        "espeak": lambda: _espeak(text, out),
        "silent": lambda: _silent(text, out),
    }
    last_err = None
    for name in chain(cfg):
        fn = engines.get(name)
        if not fn:
            continue
        try:
            path = fn()
            dur = probe(path)
            if dur > 0.2:
                return path, dur, name
        except Exception as e:
            last_err = e
            log.info("TTS engine %s failed: %s", name, e)
    raise RuntimeError(f"All TTS engines failed: {last_err}")
