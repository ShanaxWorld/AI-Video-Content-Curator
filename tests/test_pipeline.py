"""Offline tests: no API keys or network needed (remote calls are mocked).

Run:  pytest -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from curator import config, llm, script, sources, storage, topics, transcribe, video

HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Fresh data dir + no provider keys for every test.

    All modules share the single `config.settings` object (also bound as default
    arguments), so patching its attributes in place isolates every stage."""
    s = config.settings
    monkeypatch.setattr(s, "data_dir", tmp_path / "data")
    for k in ("groq_api_key", "gemini_api_key", "openrouter_api_key", "ollama_model", "youtube_api_key",
              "pollinations_api_key", "hf_token"):
        monkeypatch.setattr(s, k, "")
    monkeypatch.setattr(s, "tts_provider", "silent")
    return s


# --------------------------------------------------------------------------- topics

def test_offline_topics_are_100_unique():
    ts, src = topics.generate_topics(100, use_signals=False)
    assert len(ts) == 100
    assert len({t["title"] for t in ts}) == 100
    assert "offline" in src


def test_llm_topics_are_parsed_and_deduped(monkeypatch):
    config.settings.groq_api_key = "test"
    calls = {"n": 0}

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        cat = json["messages"][1]["content"].split('category "')[1].split('"')[0]
        items = [{"title": f"{cat} topic {i}", "why_trending": "x", "search_query": "q", "hook": "h"}
                 for i in range(12)]
        items.append({"title": f"{cat} topic 0"})  # duplicate must be dropped
        content = "```json\n" + __import__("json").dumps({"topics": items}) + "\n```"
        return Resp({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    ts, src = topics.generate_topics(100, use_signals=False)
    assert len(ts) == 100
    assert calls["n"] == len(topics.CATEGORIES)
    assert len({t["title"] for t in ts}) == 100
    assert all(t["source"] == "llm" for t in ts)


def test_extract_json_handles_prose():
    assert llm.extract_json('Sure! Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}


def test_topics_persist():
    rows, _ = topics.generate_and_store(20, use_signals=False)
    assert len(rows) == 20 and storage.get_topic(rows[0]["id"])["title"] == rows[0]["title"]


# --------------------------------------------------------------------------- sources / transcripts

@pytest.mark.parametrize(
    "s,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("not a url", None),
    ],
)
def test_parse_video_id(s, expected):
    assert sources.parse_video_id(s) == expected


def test_rank_prefers_explainer_length_then_views():
    vids = [{"id": "a", "duration": 20, "view_count": 10**9}, {"id": "b", "duration": 300, "view_count": 5},
            {"id": "c", "duration": 600, "view_count": 50}]
    assert [v["id"] for v in sources.rank(vids)] == ["c", "b"]


def test_regroup_merges_caption_fragments():
    frags = [
        {"start": 0, "end": 1, "text": "[Music]"},
        {"start": 1, "end": 2, "text": "so today we're going to"},
        {"start": 2, "end": 3, "text": "talk about why the sky"},
        {"start": 3, "end": 4, "text": "is blue. It comes down to"},
        {"start": 4, "end": 5, "text": "Rayleigh scattering of sunlight in the atmosphere."},
    ]
    segs = transcribe.regroup(frags, max_chars=60)
    assert segs[0]["start"] == 1
    assert "[Music]" not in " ".join(s["text"] for s in segs)
    assert " ".join(s["text"] for s in segs).startswith("so today")
    assert len(segs) >= 2


# --------------------------------------------------------------------------- script + captions

def test_heuristic_script_splits_scenes():
    text = " ".join(f"This is sentence number {i} about photosynthesis in plants." for i in range(8))
    sc = script.heuristic_script(text, "Photosynthesis")
    assert sc["title"] == "Photosynthesis"
    assert len(sc["scenes"]) >= 3
    assert all(len(s["narration"].split()) <= 30 for s in sc["scenes"])
    assert all("photosynthesis" in s["image_prompt"].lower() for s in sc["scenes"])


def test_ass_captions_highlight_each_word():
    words = video.words_from_text("Light bends when it enters water.", 0.0, 3.0)
    ass = video.build_ass(words, 3.0, title="Refraction")
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue: 0")]
    assert len(dialogue) == 6  # one event per word
    assert all(video.HIGHLIGHT in l for l in dialogue)
    assert "Refraction" in ass
    assert words[-1]["end"] == pytest.approx(3.0)


def test_ass_time_format():
    assert video._ass_time(3725.456) == "1:02:05.46"


# --------------------------------------------------------------------------- rendering (needs ffmpeg)

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_from_custom_text(tmp_path):
    from curator import pipeline

    out, meta = pipeline.clip_from_text("Water boils at lower temperatures on mountains. Air pressure is lower up there.",
                                        title="Boiling point", rewrite=False)
    assert out.exists() and meta["scenes"] >= 1
    w, h = _dims(out)
    assert (w, h) == (1080, 1920)
    assert video.probe_duration(out) == pytest.approx(meta["duration"], abs=0.5)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_clip_from_local_video_transcript(tmp_path):
    from curator import pipeline

    src = tmp_path / "lecture.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=300", "-t", "16", "-c:v", "libx264", "-c:a", "aac",
                    str(src)], check=True)
    v = sources.register_local_video(str(src), "Lecture")
    segs = [{"start": 0, "end": 4, "text": "Intro to gravity."},
            {"start": 4, "end": 8, "text": "Mass bends spacetime."},
            {"start": 8, "end": 12, "text": "Objects follow the curves."},
            {"start": 12, "end": 16, "text": "That is what we feel as gravity."}]
    storage.save_transcript(v["id"], segs, "test")
    out, meta = pipeline.clip_from_transcript(v["id"], [0, 2, 3], mode="original", title="Gravity")
    assert out.exists()
    assert len(meta["cuts"]) == 2  # [0] and [2,3] are separate runs
    assert _dims(out) == (1080, 1920)
    assert video.probe_duration(out) == pytest.approx(4.4 + 8.4, abs=0.6)


def _dims(p: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", str(p)], capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]
