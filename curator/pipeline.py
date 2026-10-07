"""High-level orchestration shared by the web UI and the CLI."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from . import script as scriptgen
from . import sources, storage, transcribe, video
from .config import Settings, settings

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


def _noop(*_):
    pass


def _out(name: str, cfg: Settings) -> Path:
    return cfg.dir("outputs") / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.mp4"


def curate_topic(topic_id: int, videos_per_topic: int = 3, do_transcribe: bool = True, cfg: Settings = settings) -> dict:
    """Search videos for one topic and (optionally) transcribe the top one(s)."""
    t = storage.get_topic(topic_id)
    if not t:
        raise ValueError(f"Unknown topic {topic_id}")
    vids, backend = sources.find_videos(t["search_query"] or t["title"], topic_id, videos_per_topic, cfg)
    transcribed, errors = 0, []
    if do_transcribe:
        for v in vids:
            try:
                transcribe.transcribe_video(v["id"], cfg)
                transcribed += 1
            except Exception as e:
                errors.append(f"{v['id']}: {e}")
    return {"topic": t["title"], "videos": len(vids), "transcribed": transcribed, "backend": backend,
            "errors": errors}


def batch_curate(
    limit: int = 10, videos_per_topic: int = 2, do_transcribe: bool = True, cfg: Settings = settings,
    progress: Progress = _noop,
) -> list[dict]:
    topics = storage.list_topics()[:limit]
    results = []
    for i, t in enumerate(topics):
        progress(i / max(1, len(topics)), f"Curating: {t['title']}")
        try:
            results.append(curate_topic(t["id"], videos_per_topic, do_transcribe, cfg))
        except Exception as e:
            log.warning("Curation failed for %s: %s", t["title"], e)
            results.append({"topic": t["title"], "videos": 0, "transcribed": 0, "backend": "", "errors": [str(e)]})
    progress(1.0, "Done")
    return results


def clip_from_transcript(
    video_id: str,
    segment_idx: list[int],
    mode: str = "original",
    title: str = "",
    layout: str = "blur",
    style: str = "Cinematic photo",
    voice: str | None = None,
    rewrite: bool = True,
    target_seconds: int = 45,
    cfg: Settings = settings,
    progress: Progress = _noop,
) -> tuple[Path, dict]:
    """mode='original': cut the real footage for the selected segments.
    mode='generated': turn the selected text into a brand-new AI-narrated, AI-illustrated short."""
    tr = storage.get_transcript(video_id)
    if not tr:
        raise ValueError("Transcribe this video first")
    segs = tr["segments"]
    idx = sorted(set(i for i in segment_idx if 0 <= i < len(segs)))
    if not idx:
        raise ValueError("Select at least one transcript segment")
    chosen = [segs[i] for i in idx]
    v = storage.get_video(video_id) or {}

    if mode == "original":
        progress(0.1, "Fetching source video")
        src = sources.download(video_id, "video", cfg)
        # contiguous runs of selected segments become separate cuts, joined in order
        runs: list[list[int]] = []
        for i in idx:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        out = _out("clip", cfg)
        parts, metas = [], []
        for r_i, run in enumerate(runs):
            progress(0.2 + 0.7 * r_i / len(runs), f"Cutting, reframing and captioning part {r_i + 1}/{len(runs)}")
            start, end = segs[run[0]]["start"], segs[run[-1]]["end"]
            target = out if len(runs) == 1 else out.with_name(f"{out.stem}.part{r_i}.mp4")
            p, m = video.render_source_clip(src, max(0, start - 0.15), end + 0.25, segs, target,
                                            title=title, layout=layout, cfg=cfg)
            parts.append(p)
            metas.append(m)
        if len(parts) > 1:
            video.concat_copy(parts, out)
        meta = {"duration": round(sum(m["duration"] for m in metas), 2), "cuts": [(m["start"], m["end"]) for m in metas],
                "layout": layout, "source": v.get("url") or v.get("title"), "license": v.get("license")}
        text = " ".join(s["text"] for s in chosen)
        storage.save_clip("original", str(out), text, video_id, meta)
        progress(1.0, "Done")
        return out, meta

    text = " ".join(s["text"] for s in chosen)
    progress(0.03, "Writing script")
    sc = scriptgen.make_script(text, topic=title or v.get("title", ""), rewrite=rewrite,
                               target_seconds=target_seconds, cfg=cfg)
    if title:
        sc["title"] = title
    out, meta = video.render_generated(sc, _out("short", cfg), style=style, voice=voice, cfg=cfg, progress=progress)
    meta.update({"script": sc, "source": v.get("url") or v.get("title")})
    storage.save_clip("generated", str(out), text, video_id, meta)
    return out, meta


def clip_from_text(
    text: str,
    title: str = "",
    style: str = "Cinematic photo",
    voice: str | None = None,
    rewrite: bool = False,
    target_seconds: int = 45,
    use_ai_images: bool = True,
    video_id: str | None = None,
    cfg: Settings = settings,
    progress: Progress = _noop,
) -> tuple[Path, dict]:
    """Custom (or user-edited transcript) text -> AI-narrated, AI-illustrated short."""
    progress(0.03, "Writing script")
    sc = scriptgen.make_script(text, topic=title, rewrite=rewrite, target_seconds=target_seconds, cfg=cfg)
    if title:
        sc["title"] = title
    out, meta = video.render_generated(sc, _out("short", cfg), style=style, voice=voice,
                                       use_ai_images=use_ai_images, cfg=cfg, progress=progress)
    meta["script"] = sc
    if video_id:
        v = storage.get_video(video_id) or {}
        meta["source"] = v.get("url") or v.get("title")
    storage.save_clip("generated" if video_id else "custom", str(out), text, video_id, meta)
    return out, meta
