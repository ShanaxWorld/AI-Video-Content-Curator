"""Stage 2b - transcription with a cost-aware fallback chain.

1. Existing YouTube captions (youtube-transcript-api)  - free, instant, no compute
2. Groq Whisper large-v3-turbo (free tier)             - fast, needs GROQ_API_KEY
3. Local faster-whisper (open-source, CPU int8)        - always available offline

All methods return the same shape: a list of segments
    {"start": float, "end": float, "text": str, "words": [{"start","end","word"}]?}
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

import requests

from . import sources, storage
from .config import Settings, settings

log = logging.getLogger(__name__)

Segment = dict


# --------------------------------------------------------------------------- helpers

def regroup(segments: list[Segment], max_chars: int = 220, max_dur: float = 15.0) -> list[Segment]:
    """Merge tiny caption fragments into sentence-sized, selectable segments."""
    out: list[Segment] = []
    cur: Segment | None = None
    for s in segments:
        text = re.sub(r"\s+", " ", s["text"].replace("\n", " ")).strip()
        if not text or re.fullmatch(r"\[[^\]]*\]", text):  # skip [Music], [Applause]
            continue
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": text, "words": list(s.get("words") or [])}
            continue
        ends_sentence = bool(re.search(r"[.!?]\"?$", cur["text"]))
        too_long = len(cur["text"]) + len(text) > max_chars or s["end"] - cur["start"] > max_dur
        if ends_sentence and len(cur["text"]) > 60 or too_long:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": text, "words": list(s.get("words") or [])}
        else:
            cur["text"] += " " + text
            cur["end"] = s["end"]
            cur["words"] += list(s.get("words") or [])
    if cur:
        out.append(cur)
    for s in out:
        if not s["words"]:
            s.pop("words")
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return out


def extract_audio(media: Path, cfg: Settings = settings) -> Path:
    """Re-encode to 16 kHz mono 32 kbps MP3 - small enough for Groq's 25 MB limit (~100 min)."""
    out = cfg.dir("media") / f"{media.stem}.16k.mp3"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "32k", str(out)],
            check=True,
        )
    return out


# --------------------------------------------------------------------------- backends

def from_youtube_captions(video_id: str) -> list[Segment]:
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"])
    return [{"start": s.start, "end": s.start + s.duration, "text": s.text} for s in fetched]


def from_groq_whisper(audio: Path, cfg: Settings = settings) -> list[Segment]:
    with open(audio, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
            files={"file": (audio.name, f, "audio/mpeg")},
            data={
                "model": cfg.groq_whisper_model,
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["segment", "word"],
                "language": "en",
            },
            timeout=600,
        )
    r.raise_for_status()
    data = r.json()
    words = data.get("words") or []
    segs = []
    for s in data.get("segments", []):
        seg = {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        w = [
            {"start": x["start"], "end": x["end"], "word": x["word"]}
            for x in words
            if x["start"] >= s["start"] - 0.01 and x["end"] <= s["end"] + 0.01
        ]
        if w:
            seg["words"] = w
        segs.append(seg)
    return segs


_WHISPER_CACHE: dict[str, object] = {}


def from_local_whisper(audio: Path, cfg: Settings = settings) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = _WHISPER_CACHE.get(cfg.whisper_model)
    if model is None:
        model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")
        _WHISPER_CACHE[cfg.whisper_model] = model
    segments, _info = model.transcribe(str(audio), vad_filter=True, word_timestamps=True)  # type: ignore[attr-defined]
    out = []
    for s in segments:
        seg = {"start": s.start, "end": s.end, "text": s.text.strip()}
        if s.words:
            seg["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]
        out.append(seg)
    return out


# --------------------------------------------------------------------------- orchestration

def transcribe_video(
    video_id: str,
    cfg: Settings = settings,
    force: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    progress = progress or (lambda *_: None)
    if not force:
        cached = storage.get_transcript(video_id)
        if cached:
            return cached

    errors: list[str] = []
    segs: list[Segment] | None = None
    method = ""

    if not video_id.startswith("local-"):
        progress(0.1, "Fetching existing YouTube captions")
        try:
            segs, method = from_youtube_captions(video_id), "youtube-captions"
        except Exception as e:
            errors.append(f"captions: {type(e).__name__}")

    if segs is None:
        progress(0.25, "Getting audio")
        kind = "video" if video_id.startswith("local-") else "audio"
        media = sources.download(video_id, kind=kind, cfg=cfg)
        audio = extract_audio(media, cfg)
        if cfg.groq_api_key:
            progress(0.5, "Transcribing with Groq Whisper")
            try:
                segs, method = from_groq_whisper(audio, cfg), f"groq:{cfg.groq_whisper_model}"
            except Exception as e:
                errors.append(f"groq: {e}")
        if segs is None:
            progress(0.5, f"Transcribing locally with faster-whisper ({cfg.whisper_model})")
            try:
                segs, method = from_local_whisper(audio, cfg), f"faster-whisper:{cfg.whisper_model}"
            except Exception as e:
                errors.append(f"local whisper: {e}")

    if segs is None:
        raise RuntimeError("All transcription methods failed: " + " | ".join(errors))

    segs = regroup(segs)
    storage.save_transcript(video_id, segs, method)
    progress(1.0, "Done")
    return storage.get_transcript(video_id)  # type: ignore[return-value]


def fmt_ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"
