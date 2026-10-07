"""Turn selected transcript text (or custom text) into a short-form video script:
a hook/title plus a list of scenes, each with narration and an image prompt."""
from __future__ import annotations

import logging
import re
from collections import Counter

from . import llm
from .config import Settings, settings

log = logging.getLogger(__name__)

STOP = set(
    """a an the and or but if then so of to in on at by for with from as is are was were be been being it its
this that these those i you he she we they me him her us them my your our their what which who whom whose
how why when where there here do does did done have has had not no yes can could would should will just
very really also about into over under than more most some such only own same too s t don now get got
like one two thing things way lot kind going gonna want know think say said make made see look let right
well even still much many any every each because while out up down off again further once""".split()
)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    out: list[str] = []
    for p in parts:
        words = p.split()
        # break run-on caption text (auto captions often lack punctuation)
        while len(words) > 28:
            out.append(" ".join(words[:22]))
            words = words[22:]
        if words:
            out.append(" ".join(words))
    return out


def keywords(text: str, k: int = 4) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text)]
    counts = Counter(w for w in words if w not in STOP)
    return [w for w, _ in counts.most_common(k)]


def heuristic_script(text: str, topic: str = "", max_words_per_scene: int = 24) -> dict:
    sents = split_sentences(text)
    scenes: list[dict] = []
    buf: list[str] = []
    for s in sents:
        if buf and len(" ".join(buf + [s]).split()) > max_words_per_scene:
            scenes.append({"narration": " ".join(buf)})
            buf = []
        buf.append(s)
    if buf:
        scenes.append({"narration": " ".join(buf)})
    ctx = topic or " ".join(keywords(text, 3))
    for sc in scenes:
        kw = ", ".join(keywords(sc["narration"], 4))
        sc["image_prompt"] = f"{kw}, illustrating {ctx}".strip(", ")
    title = topic or (" ".join(w.capitalize() for w in keywords(text, 3)) or "Quick Explainer")
    return {"title": title[:60], "scenes": scenes, "generator": "heuristic"}


SYSTEM = (
    "You write scripts for viral educational short-form videos (YouTube Shorts / TikTok). "
    "You are accurate: never add facts that are not supported by the source text. Reply with JSON only."
)


def llm_script(text: str, topic: str, target_seconds: int, cfg: Settings) -> dict:
    words = int(target_seconds * 2.5)
    prompt = f"""Rewrite the SOURCE into a punchy ~{target_seconds}-second vertical video script (about {words} words total).
Structure: a strong hook in scene 1, clear explanation, a memorable closing line.
Split it into 4-8 scenes. Each scene: 1-2 short spoken sentences, plus an image prompt that
describes a concrete, visual, text-free illustration for that moment.
{f'Topic: {topic}' if topic else ''}

SOURCE:
\"\"\"{text[:6000]}\"\"\"

Return JSON: {{"title": "on-screen title, max 6 words", "scenes": [{{"narration": "...", "image_prompt": "...", "label": "1-2 word key concept"}}]}}"""
    data = llm.chat_json(prompt, system=SYSTEM, temperature=0.6, cfg=cfg)
    scenes = [s for s in data.get("scenes", []) if isinstance(s, dict) and s.get("narration")]  # type: ignore[union-attr]
    if not scenes:
        raise ValueError("LLM returned no scenes")
    for s in scenes:
        s.setdefault("image_prompt", " ".join(keywords(s["narration"])))
    return {"title": str(data.get("title", topic))[:60], "scenes": scenes, "generator": "llm"}  # type: ignore[union-attr]


def make_script(
    text: str, topic: str = "", rewrite: bool = True, target_seconds: int = 45, cfg: Settings = settings
) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("No text provided")
    if rewrite and llm.providers(cfg):
        try:
            return llm_script(text, topic, target_seconds, cfg)
        except Exception as e:
            log.warning("LLM script rewrite failed, using text as-is: %s", e)
    return heuristic_script(text, topic)
