"""Stage 1 - trending educational topic generation.

1. Collect *real* trend signals from free sources (all optional, each fails soft):
   - YouTube Data API `videos.list?chart=mostPopular` for Education / Science & Tech
   - Google Trends daily "trending now" RSS feed (no key)
   - Wikipedia most-viewed articles from yesterday (Wikimedia REST API, no key)
2. Ask an LLM to turn those signals + its own knowledge of what performs on
   YouTube / TikTok into N short-form-friendly educational topics, in batches
   per category so a 100-item list doesn't get truncated.
3. De-duplicate and persist. If no LLM is reachable, fall back to a bundled seed list.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import xml.etree.ElementTree as ET
from typing import Callable

import requests

from . import llm, storage
from .config import Settings, settings
from .seed_topics import seed_topics

log = logging.getLogger(__name__)

CATEGORIES = [
    "AI & Technology",
    "Science",
    "Space",
    "Personal Finance",
    "Health & Psychology",
    "History",
    "Math & Logic",
    "Productivity & Learning",
    "Environment & Earth",
    "Coding & Careers",
]

UA = {"User-Agent": "ai-video-curator/1.0 (educational demo)"}


# --------------------------------------------------------------------------- signals

def youtube_trending_titles(cfg: Settings = settings, per_category: int = 25) -> list[str]:
    if not cfg.youtube_api_key:
        return []
    titles: list[str] = []
    for cat_id in ("27", "28"):  # Education, Science & Technology
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "snippet",
                    "chart": "mostPopular",
                    "videoCategoryId": cat_id,
                    "regionCode": cfg.youtube_region,
                    "maxResults": per_category,
                    "key": cfg.youtube_api_key,
                },
                timeout=20,
            )
            r.raise_for_status()
            titles += [i["snippet"]["title"] for i in r.json().get("items", [])]
        except Exception as e:
            log.info("YouTube trending (cat %s) unavailable: %s", cat_id, e)
    return titles


def google_trends_titles(geo: str = "US") -> list[str]:
    try:
        r = requests.get(f"https://trends.google.com/trending/rss?geo={geo}", headers=UA, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        return [i.findtext("title", "") for i in root.iter("item")][:30]
    except Exception as e:
        log.info("Google Trends RSS unavailable: %s", e)
        return []


def wikipedia_top_articles(limit: int = 40) -> list[str]:
    day = dt.date.today() - dt.timedelta(days=1)
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
        f"{day:%Y/%m/%d}"
    )
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        arts = r.json()["items"][0]["articles"]
        skip = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Template:|Category:)")
        names = [a["article"].replace("_", " ") for a in arts if not skip.match(a["article"])]
        return names[:limit]
    except Exception as e:
        log.info("Wikipedia pageviews unavailable: %s", e)
        return []


def collect_signals(cfg: Settings = settings) -> dict[str, list[str]]:
    return {
        "youtube_most_popular_education": youtube_trending_titles(cfg),
        "google_trends_today": google_trends_titles(cfg.youtube_region),
        "wikipedia_most_viewed_yesterday": wikipedia_top_articles(),
    }


# --------------------------------------------------------------------------- generation

SYSTEM = (
    "You are a content strategist for an educational short-form video channel "
    "(YouTube Shorts, TikTok, Reels). You know which explainer formats go viral. "
    "Reply with JSON only."
)


def _batch_prompt(category: str, n: int, signals: dict[str, list[str]], avoid: list[str]) -> str:
    sig_lines = []
    for name, items in signals.items():
        if items:
            sig_lines.append(f"- {name}: " + "; ".join(items[:25]))
    signal_block = (
        "Real-time trend signals collected today (use them where they fit this category; "
        "ignore celebrity gossip, sports scores and anything not educational):\n" + "\n".join(sig_lines)
        if sig_lines
        else "No live trend data is available; rely on your knowledge of what currently performs well."
    )
    avoid_block = ("Do NOT repeat any of these existing topics: " + "; ".join(avoid[-60:])) if avoid else ""
    return f"""Today is {dt.date.today():%B %d, %Y}.
Generate {n} distinct trending educational video topics in the category "{category}".
Each must be explainable in a 30-90 second vertical video and be something people are
actively searching for or watching on YouTube/TikTok right now.

{signal_block}
{avoid_block}

Return JSON: {{"topics": [{{
  "title": "catchy but accurate topic title (max 70 chars)",
  "why_trending": "one sentence: why this is popular now",
  "search_query": "3-6 word YouTube search query to find great existing explainer videos",
  "hook": "a scroll-stopping first line for a short video"
}}]}}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def generate_topics(
    n: int = 100,
    cfg: Settings = settings,
    use_signals: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], str]:
    """Return (topics, source_description)."""
    progress = progress or (lambda *_: None)
    progress(0.02, "Collecting trend signals")
    signals = collect_signals(cfg) if use_signals else {}
    n_signals = sum(len(v) for v in signals.values())

    if not llm.providers(cfg):
        log.warning("No LLM configured - using offline seed topics")
        return seed_topics()[:n], "offline seed list (configure an LLM key for live generation)"

    per_cat = -(-n // len(CATEGORIES))  # ceil
    topics: list[dict] = []
    seen: set[str] = set()
    for i, cat in enumerate(CATEGORIES):
        progress(0.05 + 0.9 * i / len(CATEGORIES), f"Generating topics: {cat}")
        try:
            data = llm.chat_json(
                _batch_prompt(cat, per_cat + 2, signals, [t["title"] for t in topics]),
                system=SYSTEM,
                temperature=0.8,
                cfg=cfg,
            )
            items = data.get("topics", []) if isinstance(data, dict) else data
        except Exception as e:
            log.warning("Topic batch for %s failed: %s", cat, e)
            items = []
        added = 0
        for it in items or []:
            if not isinstance(it, dict) or not it.get("title"):
                continue
            key = _norm(it["title"])
            if key in seen:
                continue
            seen.add(key)
            topics.append(
                {
                    "title": it["title"].strip()[:120],
                    "category": cat,
                    "why_trending": it.get("why_trending", ""),
                    "search_query": it.get("search_query") or it["title"],
                    "hook": it.get("hook", ""),
                    "source": "llm+signals" if n_signals else "llm",
                }
            )
            added += 1
            if added >= per_cat:
                break

    # top up from the seed list if the LLM under-delivered
    if len(topics) < n:
        for s in seed_topics():
            if len(topics) >= n:
                break
            if _norm(s["title"]) not in seen:
                seen.add(_norm(s["title"]))
                topics.append(s)

    progress(1.0, "Done")
    src = f"{llm.active_provider_name(cfg)}; {n_signals} live trend signals"
    return topics[:n], src


def generate_and_store(n: int = 100, replace: bool = True, **kw) -> tuple[list[dict], str]:
    topics, src = generate_topics(n, **kw)
    storage.save_topics(topics, replace=replace)
    return storage.list_topics(), src
