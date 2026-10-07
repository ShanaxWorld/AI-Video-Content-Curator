"""Command-line interface - run any stage, or the whole pipeline, headless.

Examples
  python cli.py status
  python cli.py topics --n 100
  python cli.py curate --limit 10 --per-topic 2
  python cli.py transcribe dQw4w9WgXcQ
  python cli.py add-file lecture.mp4
  python cli.py clip VIDEO_ID --segments 3-6 --mode original --title "Why the sky is blue"
  python cli.py text "Compound interest is..." --title "Compound interest" --rewrite
  python cli.py run --topics 100 --curate 5 --clips 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from curator import pipeline, sources, storage, topics, transcribe
from curator.config import settings


def _bar(frac: float, msg: str) -> None:
    n = int(frac * 30)
    sys.stderr.write(f"\r[{'#' * n}{'.' * (30 - n)}] {msg[:60]:<60}")
    if frac >= 1:
        sys.stderr.write("\n")
    sys.stderr.flush()


def parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI-Powered Video Content Curator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show which AI providers are active")

    p = sub.add_parser("topics", help="generate trending educational topics")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--no-signals", action="store_true")

    p = sub.add_parser("curate", help="find + transcribe videos for stored topics")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--per-topic", type=int, default=2)
    p.add_argument("--no-transcribe", action="store_true")

    p = sub.add_parser("transcribe", help="transcribe a YouTube id/URL or a stored video id")
    p.add_argument("video")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("add-file", help="register a local video file as a source")
    p.add_argument("path")
    p.add_argument("--title")

    p = sub.add_parser("show", help="print a stored transcript with segment numbers")
    p.add_argument("video_id")

    p = sub.add_parser("clip", help="make a short from transcript segments")
    p.add_argument("video_id")
    p.add_argument("--segments", required=True, help="e.g. 2-5 or 1,3,7")
    p.add_argument("--mode", choices=["original", "generated"], default="original")
    p.add_argument("--layout", choices=["blur", "crop"], default="blur")
    p.add_argument("--title", default="")
    p.add_argument("--style", default="Cinematic photo")
    p.add_argument("--no-rewrite", action="store_true")

    p = sub.add_parser("text", help="make a short from custom text")
    p.add_argument("text")
    p.add_argument("--title", default="")
    p.add_argument("--style", default="Cinematic photo")
    p.add_argument("--voice")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--seconds", type=int, default=45)

    p = sub.add_parser("run", help="end-to-end: topics -> curate -> auto clips")
    p.add_argument("--topics", type=int, default=100)
    p.add_argument("--curate", type=int, default=3, help="how many topics to curate")
    p.add_argument("--clips", type=int, default=2, help="how many auto clips to render")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if a.cmd == "status":
        for k, v in settings.status().items():
            print(f"{k:<15} {v}")

    elif a.cmd == "topics":
        rows, src = topics.generate_and_store(a.n, use_signals=not a.no_signals, progress=_bar)
        for t in rows:
            print(f"{t['id']:>3}. [{t['category']}] {t['title']}")
        print(f"\n{len(rows)} topics via {src}")

    elif a.cmd == "curate":
        for r in pipeline.batch_curate(a.limit, a.per_topic, not a.no_transcribe, progress=_bar):
            print(json.dumps(r))

    elif a.cmd == "transcribe":
        vid = sources.parse_video_id(a.video) or a.video
        if not storage.get_video(vid):
            storage.save_video({"id": vid, "title": vid, "url": f"https://www.youtube.com/watch?v={vid}"})
        tr = transcribe.transcribe_video(vid, force=a.force, progress=_bar)
        print(f"{len(tr['segments'])} segments via {tr['method']}")
        for i, s in enumerate(tr["segments"]):
            print(f"{i:>3} [{transcribe.fmt_ts(s['start'])}] {s['text']}")

    elif a.cmd == "add-file":
        v = sources.register_local_video(a.path, a.title)
        print(f"Registered {v['id']} ({v['duration']:.1f}s). Next: python cli.py transcribe {v['id']}")

    elif a.cmd == "show":
        tr = storage.get_transcript(a.video_id)
        if not tr:
            print("No transcript")
            return 1
        for i, s in enumerate(tr["segments"]):
            print(f"{i:>3} [{transcribe.fmt_ts(s['start'])}-{transcribe.fmt_ts(s['end'])}] {s['text']}")

    elif a.cmd == "clip":
        out, meta = pipeline.clip_from_transcript(
            a.video_id, parse_range(a.segments), mode=a.mode, title=a.title, layout=a.layout, style=a.style,
            rewrite=not a.no_rewrite, progress=_bar,
        )
        print(out)
        print(json.dumps(meta, indent=2, default=str))

    elif a.cmd == "text":
        out, meta = pipeline.clip_from_text(a.text, title=a.title, style=a.style, voice=a.voice, rewrite=a.rewrite,
                                            target_seconds=a.seconds, progress=_bar)
        print(out)
        print(json.dumps(meta, indent=2, default=str))

    elif a.cmd == "run":
        rows, src = topics.generate_and_store(a.topics, progress=_bar)
        print(f"{len(rows)} topics via {src}")
        results = pipeline.batch_curate(a.curate, 1, True, progress=_bar)
        made = 0
        for v in storage.list_videos():
            if made >= a.clips:
                break
            tr = storage.get_transcript(v["id"])
            if not tr or len(tr["segments"]) < 3:
                continue
            # auto-pick: the first ~40s window after the intro
            segs = tr["segments"]
            start = 1 if len(segs) > 4 else 0
            idx, t0 = [], segs[start]["start"]
            for i in range(start, len(segs)):
                if segs[i]["end"] - t0 > 40:
                    break
                idx.append(i)
            try:
                out, _ = pipeline.clip_from_transcript(v["id"], idx or [start], mode="original",
                                                       title=v.get("topic_title") or "", progress=_bar)
                print("clip:", out)
                made += 1
            except Exception as e:
                print(f"clip failed for {v['id']}: {e}")
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
