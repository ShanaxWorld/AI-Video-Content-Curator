"""Stage 3 - rendering vertical (1080x1920) short videos with ffmpeg.

Two modes:
  * render_generated(): script scenes -> AI image + TTS per scene -> Ken Burns motion
    -> concat -> word-by-word highlighted captions (ASS/libass) + title bar.
  * render_source_clip(): cut a time range out of the original video, reframe 16:9 to
    9:16 (blurred-background or centre-crop), burn word-highlighted captions taken from
    the transcript timestamps.
"""
from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from . import images, tts
from .config import Settings, settings

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
HIGHLIGHT = "&H0000E5FF&"  # ASS colours are &HAABBGGRR -> #FFE500 (yellow)

Word = dict  # {"start": float, "end": float, "word": str}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:4])}...):\n{res.stderr[-1500:]}")


def probe_duration(path: Path | str) -> float:
    return tts.probe(Path(path))


def _ff_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filtergraph option."""
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


# --------------------------------------------------------------------------- captions (ASS)

def words_from_text(text: str, t0: float, t1: float) -> list[Word]:
    """Spread words over [t0, t1] proportionally to their length (for TTS / caption-only timing)."""
    ws = text.split()
    if not ws:
        return []
    weights = [len(w) + 2 for w in ws]
    total = sum(weights)
    out, t = [], t0
    for w, wt in zip(ws, weights):
        d = (t1 - t0) * wt / total
        out.append({"start": t, "end": t + d, "word": w})
        t += d
    return out


def chunk_words(words: list[Word], max_words: int = 4, max_chars: int = 24) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        text_len = len(" ".join(x["word"].strip() for x in cur + [w]))
        if cur and (len(cur) >= max_words or text_len > max_chars):
            chunks.append(cur)
            cur = []
        cur.append({**w, "word": token})
        if re.search(r"[.!?,;:]$", token) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(s: str) -> str:
    return s.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(
    words: list[Word],
    total: float,
    title: str = "",
    caption_margin_v: int = 560,
    font: str = "Poppins",
    uppercase: bool = False,
    W: int = 1080,
    H: int = 1920,
) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,{caption_margin_v},1
Style: Title,{font},58,&H00FFFFFF,&H00FFFFFF,&H28140F0A,&H00000000,-1,0,0,0,100,100,0,0,3,22,0,8,90,90,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        lines.append(
            f"Dialogue: 1,{_ass_time(0)},{_ass_time(total)},Title,,0,0,0,,{{\\fad(300,300)}}{_ass_escape(title)}"
        )
    chunks = chunk_words(words)
    for ci, chunk in enumerate(chunks):
        c_end = chunk[-1]["end"]
        if ci + 1 < len(chunks):
            nxt = chunks[ci + 1][0]["start"]
            if nxt - c_end < 0.35:
                c_end = nxt  # avoid flicker between consecutive chunks
        tokens = [_ass_escape(w["word"].upper() if uppercase else w["word"]) for w in chunk]
        for i, w in enumerate(chunk):
            s = w["start"]
            e = chunk[i + 1]["start"] if i + 1 < len(chunk) else c_end
            if e - s < 0.04:
                continue
            parts = [
                f"{{\\c{HIGHLIGHT}\\fscx110\\fscy110}}{tok}{{\\r}}" if j == i else tok
                for j, tok in enumerate(tokens)
            ]
            lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Caption,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"


def _font_family(cfg: Settings) -> tuple[str, Path | None]:
    """Prefer the bundled Poppins (OFL); fall back to whatever fontconfig has."""
    if cfg.font_path and Path(cfg.font_path).exists():
        return Path(cfg.font_path).stem.split("-")[0], Path(cfg.font_path).parent
    bundled = ASSETS / "fonts"
    if (bundled / "Poppins-Bold.ttf").exists():
        return "Poppins", bundled
    return "DejaVu Sans", None


def _subtitles_filter(ass_path: Path, cfg: Settings) -> str:
    _, fontsdir = _font_family(cfg)
    f = f"subtitles=filename='{_ff_path(ass_path)}'"
    if fontsdir:
        f += f":fontsdir='{_ff_path(fontsdir)}'"
    return f


# --------------------------------------------------------------------------- generated mode

KB_MOTIONS = [
    # (zoom expression, x expression, y expression) - zoom in, zoom out, pan up, pan right
    ("1+0.14*on/{N}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    ("1.14-0.14*on/{N}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    ("1.12", "iw/2-(iw/zoom/2)", "(ih-ih/zoom)*(1-on/{N})"),
    ("1.12", "(iw-iw/zoom)*on/{N}", "ih/2-(ih/zoom/2)"),
]


def _label(text: str) -> str:
    from .script import keywords

    kw = keywords(text, 1)
    return kw[0] if kw else ""


def render_scene(image: Path, audio: Path, dur: float, out: Path, motion: int, cfg: Settings) -> Path:
    W, H, fps = cfg.width, cfg.height, cfg.fps
    N = max(1, math.ceil(dur * fps))
    z, x, y = (e.replace("{N}", str(N)) for e in KB_MOTIONS[motion % len(KB_MOTIONS)])
    vf = (
        f"[0:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={N}:s={W}x{H}:fps={fps},"
        f"fade=t=in:st=0:d=0.25,format=yuv420p[v];"
        f"[1:a]aresample=44100,aformat=channel_layouts=stereo,apad[a]"
    )
    run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(image), "-i", str(audio), "-filter_complex", vf,
         "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "21", "-r", str(fps), "-c:a", "aac", "-b:a", "128k", str(out)]
    )
    return out


def render_generated(
    script: dict,
    out_path: Path,
    style: str = "Cinematic photo",
    voice: str | None = None,
    show_title: bool = True,
    use_ai_images: bool = True,
    cfg: Settings = settings,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, dict]:
    progress = progress or (lambda *_: None)
    work = cfg.dir("work") / f"gen-{int(time.time() * 1000)}"
    work.mkdir(parents=True)
    scenes = script["scenes"]
    timeline: list[Word] = []
    clips: list[Path] = []
    used = {"tts": set(), "images": set()}
    t = 0.0
    for i, sc in enumerate(scenes):
        frac = i / max(1, len(scenes))
        progress(0.05 + 0.8 * frac, f"Scene {i + 1}/{len(scenes)}: voice")
        audio, adur, engine = tts.synthesize(sc["narration"], work / f"s{i:02d}", voice=voice, cfg=cfg)
        used["tts"].add(engine)
        progress(0.05 + 0.8 * (frac + 0.3 / len(scenes)), f"Scene {i + 1}/{len(scenes)}: image")
        img, provider = images.generate(
            sc.get("image_prompt") or sc["narration"], work / f"s{i:02d}.png", style=style, seed=1000 + i,
            cfg=cfg, allow_remote=use_ai_images, label=sc.get("label") or _label(sc["narration"]),
        )
        used["images"].add(provider)
        dur = adur + 0.3  # small breath between scenes
        progress(0.05 + 0.8 * (frac + 0.6 / len(scenes)), f"Scene {i + 1}/{len(scenes)}: motion")
        clips.append(render_scene(img, audio, dur, work / f"s{i:02d}.mp4", i, cfg))
        timeline += words_from_text(sc["narration"], t + 0.05, t + adur)
        t += dur

    progress(0.9, "Burning captions")
    family, _ = _font_family(cfg)
    (work / "captions.ass").write_text(
        build_ass(timeline, t, title=script.get("title", "") if show_title else "", font=family), encoding="utf-8"
    )
    (work / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt",
         "-vf", _subtitles_filter(work / "captions.ass", cfg), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "21", "-c:a", "copy", "-movflags", "+faststart", str(out_path.resolve())],
        cwd=work,
    )
    shutil.rmtree(work, ignore_errors=True)
    progress(1.0, "Done")
    meta = {"duration": round(t, 2), "scenes": len(scenes), "tts": sorted(used["tts"]),
            "images": sorted(used["images"])}
    return out_path, meta


def concat_copy(parts: list[Path], out: Path, remove_parts: bool = True) -> Path:
    """Join same-encoded mp4 parts without re-encoding."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(out)])
    lst.unlink(missing_ok=True)
    if remove_parts:
        for p in parts:
            p.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- source-clip mode

def render_source_clip(
    src: Path,
    start: float,
    end: float,
    segments: list[dict],
    out_path: Path,
    title: str = "",
    layout: str = "blur",
    cfg: Settings = settings,
) -> tuple[Path, dict]:
    """Cut [start, end] from src, reframe to 9:16 and burn captions from transcript segments."""
    end = min(end, start + cfg.max_clip_seconds)
    dur = end - start
    if dur <= 0.5:
        raise ValueError("Clip is too short")
    W, H = cfg.width, cfg.height

    # caption words on the clip's own timeline
    words: list[Word] = []
    for s in segments:
        if s["end"] <= start or s["start"] >= end:
            continue
        if s.get("words"):
            ws = [{"start": w["start"], "end": w["end"], "word": w["word"]} for w in s["words"]]
        else:
            ws = words_from_text(s["text"], s["start"], s["end"])
        for w in ws:
            if start <= w["start"] < end:
                words.append({"start": w["start"] - start, "end": min(w["end"], end) - start, "word": w["word"]})

    work = cfg.dir("work") / f"clip-{int(time.time() * 1000)}"
    work.mkdir(parents=True)
    family, _ = _font_family(cfg)
    margin = 430 if layout == "blur" else 520
    (work / "captions.ass").write_text(build_ass(words, dur, title=title, caption_margin_v=margin, font=family),
                                       encoding="utf-8")
    subs = _subtitles_filter(work / "captions.ass", cfg)
    if layout == "crop":
        vf = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,{subs},format=yuv420p[v]"
    else:
        vf = (
            f"[0:v]split[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=28:2,eq=brightness=-0.12[bg];"
            f"[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs},format=yuv420p[v]"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src.resolve()),
         "-filter_complex", vf, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "21", "-r", str(cfg.fps), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         str(out_path.resolve())],
        cwd=work,
    )
    shutil.rmtree(work, ignore_errors=True)
    return out_path, {"duration": round(dur, 2), "start": start, "end": end, "layout": layout,
                      "caption_words": len(words)}
