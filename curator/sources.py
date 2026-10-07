"""Stage 2a - find popular existing videos for a topic, and fetch media.

Search backends:
  * YouTube Data API v3 (free 10k units/day). Supports `videoLicense=creativeCommon`,
    which matters here: we *repurpose* footage, so CC-BY sources are the safe default.
  * yt-dlp `ytsearch` (no key) as a fallback.
Local uploads are also supported so the tool works on footage you own, and on
cloud hosts where YouTube blocks downloads.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import requests

from . import storage
from .config import Settings, settings

log = logging.getLogger(__name__)

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def parse_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = YT_ID_RE.search(s)
    return m.group(1) if m else None


def _iso8601_duration(d: str) -> float:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0.0
    days, h, mi, s = (int(x or 0) for x in m.groups())
    return days * 86400 + h * 3600 + mi * 60 + s


# --------------------------------------------------------------------------- search

def search_youtube_api(query: str, limit: int, cfg: Settings) -> list[dict[str, Any]]:
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(limit * 2, 50),
        "relevanceLanguage": "en",
        "safeSearch": "strict",
        "videoEmbeddable": "true",
        "key": cfg.youtube_api_key,
    }
    if cfg.creative_commons_only:
        params["videoLicense"] = "creativeCommon"
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=20)
    r.raise_for_status()
    ids = [i["id"]["videoId"] for i in r.json().get("items", [])]
    if not ids:
        return []
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet,statistics,contentDetails,status", "id": ",".join(ids), "key": cfg.youtube_api_key},
        timeout=20,
    )
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        out.append(
            {
                "id": it["id"],
                "title": it["snippet"]["title"],
                "channel": it["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={it['id']}",
                "duration": _iso8601_duration(it["contentDetails"].get("duration", "")),
                "view_count": int(it.get("statistics", {}).get("viewCount", 0)),
                "license": it.get("status", {}).get("license", "youtube"),
            }
        )
    return out


def search_ytdlp(query: str, limit: int) -> list[dict[str, Any]]:
    import yt_dlp

    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist", "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit * 2}:{query}", download=False)
    out = []
    for e in info.get("entries", []) or []:
        if not e or not e.get("id"):
            continue
        out.append(
            {
                "id": e["id"],
                "title": e.get("title", ""),
                "channel": e.get("channel") or e.get("uploader", ""),
                "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
                "duration": float(e.get("duration") or 0),
                "view_count": int(e.get("view_count") or 0),
                "license": "unknown (check before reuse)",
            }
        )
    return out


def rank(videos: list[dict], min_s: float = 60, max_s: float = 1800) -> list[dict]:
    """Prefer explainer-length videos, then popularity."""
    ok = [v for v in videos if min_s <= (v.get("duration") or 0) <= max_s] or videos
    return sorted(ok, key=lambda v: v.get("view_count") or 0, reverse=True)


def find_videos(
    query: str, topic_id: int | None = None, limit: int = 5, cfg: Settings = settings
) -> tuple[list[dict], str]:
    backend = ""
    videos: list[dict] = []
    if cfg.youtube_api_key:
        try:
            videos = search_youtube_api(query, limit, cfg)
            backend = "YouTube Data API" + (" (Creative Commons)" if cfg.creative_commons_only else "")
        except Exception as e:
            log.warning("YouTube API search failed, falling back to yt-dlp: %s", e)
    if not videos:
        videos = search_ytdlp(query, limit)
        backend = "yt-dlp search"
    videos = rank(videos)[:limit]
    for v in videos:
        storage.save_video({**v, "topic_id": topic_id})
    return videos, backend


# --------------------------------------------------------------------------- media

def download(video_id: str, kind: str = "video", cfg: Settings = settings) -> Path:
    """Download a YouTube video (<=720p mp4) or just its audio. Cached on disk."""
    import yt_dlp

    v = storage.get_video(video_id)
    if v and v.get("local_path") and Path(v["local_path"]).exists():
        return Path(v["local_path"])  # local uploads and previously downloaded videos

    out_dir = cfg.dir("media")
    if kind == "audio":
        existing = list(out_dir.glob(f"{video_id}.audio.*"))
        if existing:
            return existing[0]
        opts = {"format": "bestaudio[ext=m4a]/bestaudio/best", "outtmpl": str(out_dir / f"{video_id}.audio.%(ext)s")}
    else:
        target = out_dir / f"{video_id}.mp4"
        if target.exists():
            return target
        opts = {
            "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        }
    opts.update({"quiet": True, "noplaylist": True})
    url = (v or {}).get("url") or f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if kind == "video":
        mp4 = path.with_suffix(".mp4")
        path = mp4 if mp4.exists() else path
        storage.set_video_local_path(video_id, str(path))
    return path


def register_local_video(file_path: str, title: str | None = None, cfg: Settings = settings) -> dict:
    """Copy an uploaded file into the media folder and register it as a source video."""
    src = Path(file_path)
    h = hashlib.sha1(src.read_bytes()[: 4 * 1024 * 1024]).hexdigest()[:10]
    vid = f"local-{h}"
    dst = cfg.dir("media") / f"{vid}{src.suffix.lower() or '.mp4'}"
    if not dst.exists():
        shutil.copy(src, dst)
    from .video import probe_duration

    v = {
        "id": vid,
        "title": title or src.stem,
        "channel": "local upload",
        "url": "",
        "duration": probe_duration(dst),
        "view_count": None,
        "license": "owned by uploader",
        "local_path": str(dst),
    }
    storage.save_video(v)
    return v
