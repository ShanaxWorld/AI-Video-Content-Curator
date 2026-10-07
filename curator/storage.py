"""Tiny SQLite persistence layer. Each pipeline stage writes its results here so
stages are resumable and the UI / CLI share the same state."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    category TEXT,
    why_trending TEXT,
    search_query TEXT,
    hook TEXT,
    source TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,             -- YouTube id or 'local-<hash>'
    topic_id INTEGER REFERENCES topics(id),
    title TEXT,
    channel TEXT,
    url TEXT,
    duration REAL,
    view_count INTEGER,
    license TEXT,
    local_path TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT PRIMARY KEY REFERENCES videos(id),
    method TEXT,
    language TEXT,
    segments TEXT NOT NULL,          -- JSON list of {start, end, text}
    created_at REAL
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT,
    video_id TEXT,
    script TEXT,
    path TEXT,
    meta TEXT,
    created_at REAL
);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---- topics ---------------------------------------------------------------

def save_topics(topics: list[dict[str, Any]], replace: bool = False) -> int:
    with connect() as c:
        if replace:
            c.execute("DELETE FROM topics")
        n = 0
        for t in topics:
            cur = c.execute(
                "INSERT OR IGNORE INTO topics(title, category, why_trending, search_query, hook, source, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    t["title"],
                    t.get("category", ""),
                    t.get("why_trending", ""),
                    t.get("search_query") or t["title"],
                    t.get("hook", ""),
                    t.get("source", ""),
                    time.time(),
                ),
            )
            n += cur.rowcount
        return n


def list_topics() -> list[dict[str, Any]]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM topics ORDER BY id")]


def get_topic(topic_id: int) -> dict[str, Any] | None:
    with connect() as c:
        r = c.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        return dict(r) if r else None


# ---- videos ---------------------------------------------------------------

def save_video(v: dict[str, Any]) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO videos(id, topic_id, title, channel, url, duration, view_count, license, local_path, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET topic_id=COALESCE(excluded.topic_id, topic_id),"
            " title=excluded.title, channel=excluded.channel, url=excluded.url,"
            " duration=COALESCE(excluded.duration, duration), view_count=COALESCE(excluded.view_count, view_count),"
            " license=COALESCE(excluded.license, license), local_path=COALESCE(excluded.local_path, local_path)",
            (
                v["id"],
                v.get("topic_id"),
                v.get("title"),
                v.get("channel"),
                v.get("url"),
                v.get("duration"),
                v.get("view_count"),
                v.get("license"),
                v.get("local_path"),
                time.time(),
            ),
        )


def set_video_local_path(video_id: str, path: str) -> None:
    with connect() as c:
        c.execute("UPDATE videos SET local_path=? WHERE id=?", (path, video_id))


def list_videos(topic_id: int | None = None) -> list[dict[str, Any]]:
    q = (
        "SELECT v.*, t.title AS topic_title, (tr.video_id IS NOT NULL) AS has_transcript"
        " FROM videos v LEFT JOIN topics t ON t.id=v.topic_id LEFT JOIN transcripts tr ON tr.video_id=v.id"
    )
    args: tuple = ()
    if topic_id is not None:
        q += " WHERE v.topic_id=?"
        args = (topic_id,)
    q += " ORDER BY v.view_count DESC NULLS LAST, v.created_at DESC"
    with connect() as c:
        return [dict(r) for r in c.execute(q, args)]


def get_video(video_id: str) -> dict[str, Any] | None:
    with connect() as c:
        r = c.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        return dict(r) if r else None


# ---- transcripts ------------------------------------------------------------

def save_transcript(video_id: str, segments: list[dict], method: str, language: str = "en") -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO transcripts(video_id, method, language, segments, created_at) VALUES (?,?,?,?,?)",
            (video_id, method, language, json.dumps(segments), time.time()),
        )


def get_transcript(video_id: str) -> dict[str, Any] | None:
    with connect() as c:
        r = c.execute("SELECT * FROM transcripts WHERE video_id=?", (video_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["segments"] = json.loads(d["segments"])
        return d


# ---- clips ------------------------------------------------------------------

def save_clip(mode: str, path: str, script: str, video_id: str | None = None, meta: dict | None = None) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO clips(mode, video_id, script, path, meta, created_at) VALUES (?,?,?,?,?,?)",
            (mode, video_id, script, path, json.dumps(meta or {}), time.time()),
        )
        return int(cur.lastrowid)


def list_clips() -> list[dict[str, Any]]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM clips ORDER BY id DESC")]
