"""Gradio web UI for the AI-Powered Video Content Curator.

Run:  python app.py      (then open http://localhost:7860)
"""
from __future__ import annotations

import csv
import json
import logging
import os

import gradio as gr

from curator import pipeline, sources, storage, topics, transcribe
from curator.config import settings
from curator.images import STYLES
from curator.transcribe import fmt_ts
from curator.tts import EDGE_VOICES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TOPIC_COLS = ["id", "category", "title", "why_trending", "search_query"]
VIDEO_COLS = ["id", "title", "channel", "views", "duration", "license", "transcribed"]


# --------------------------------------------------------------------------- helpers

def _topic_rows():
    return [[t[c] for c in TOPIC_COLS] for t in storage.list_topics()]


def _topic_choices():
    return [(f"#{t['id']} · {t['title']}", t["id"]) for t in storage.list_topics()]


def _video_rows(topic_id=None):
    rows = []
    for v in storage.list_videos(topic_id):
        rows.append([
            v["id"], v["title"], v["channel"], f"{v['view_count']:,}" if v["view_count"] else "-",
            fmt_ts(v["duration"] or 0), v["license"] or "", "yes" if v["has_transcript"] else "",
        ])
    return rows


def _video_choices(only_transcribed=False):
    out = []
    for v in storage.list_videos():
        if only_transcribed and not v["has_transcript"]:
            continue
        mark = "📝 " if v["has_transcript"] else ""
        out.append((f"{mark}{v['title'][:70]} ({v['id']})", v["id"]))
    return out


def _segment_choices(video_id):
    tr = storage.get_transcript(video_id) if video_id else None
    if not tr:
        return []
    return [(f"[{fmt_ts(s['start'])}] {s['text']}", i) for i, s in enumerate(tr["segments"])]


def _status_md():
    rows = "\n".join(f"| {k} | {v} |" for k, v in settings.status().items())
    return f"| Stage | Active provider(s) |\n|---|---|\n{rows}"


def _progress_cb(progress: gr.Progress):
    return lambda frac, msg: progress(frac, desc=msg)


# --------------------------------------------------------------------------- tab 1: topics

def do_generate_topics(n, use_signals, progress=gr.Progress()):
    rows, src = topics.generate_and_store(int(n), replace=True, use_signals=use_signals,
                                          progress=_progress_cb(progress))
    return [[t[c] for c in TOPIC_COLS] for t in rows], f"Generated **{len(rows)}** topics via {src}.", \
        gr.Dropdown(choices=_topic_choices(), value=None)


def do_export_topics():
    path = settings.dir("outputs") / "topics.csv"
    ts = storage.list_topics()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ts[0].keys()) if ts else TOPIC_COLS)
        w.writeheader()
        w.writerows(ts)
    return str(path)


# --------------------------------------------------------------------------- tab 2: curate

def do_find_videos(topic_id, n):
    if topic_id is None:
        raise gr.Error("Pick a topic first (generate topics in tab 1).")
    t = storage.get_topic(int(topic_id))
    try:
        vids, backend = sources.find_videos(t["search_query"] or t["title"], t["id"], int(n))
    except Exception as e:
        raise gr.Error(f"Video search failed: {e}")
    msg = f"Found **{len(vids)}** videos for “{t['title']}” using {backend}."
    return _video_rows(t["id"]), msg, gr.Dropdown(choices=_video_choices(), value=vids[0]["id"] if vids else None)


def do_add_url(url):
    vid = sources.parse_video_id(url or "")
    if not vid:
        raise gr.Error("That doesn't look like a YouTube URL or video id.")
    storage.save_video({"id": vid, "title": f"YouTube video {vid}", "channel": "", "url": f"https://www.youtube.com/watch?v={vid}",
                        "license": "unknown (check before reuse)"})
    return _video_rows(), gr.Dropdown(choices=_video_choices(), value=vid)


def do_upload(file):
    if not file:
        raise gr.Error("Choose a video file to upload.")
    v = sources.register_local_video(file)
    return _video_rows(), gr.Dropdown(choices=_video_choices(), value=v["id"])


def do_transcribe(video_id, force, progress=gr.Progress()):
    if not video_id:
        raise gr.Error("Select a video to transcribe.")
    try:
        tr = transcribe.transcribe_video(video_id, force=force, progress=_progress_cb(progress))
    except Exception as e:
        raise gr.Error(str(e)[:500])
    rows = [[i, fmt_ts(s["start"]), fmt_ts(s["end"]), s["text"]] for i, s in enumerate(tr["segments"])]
    msg = f"Transcribed with **{tr['method']}** · {len(rows)} segments."
    return rows, msg, _video_rows(), gr.Dropdown(choices=_video_choices(True), value=video_id)


def show_transcript(video_id):
    tr = storage.get_transcript(video_id) if video_id else None
    if not tr:
        return [], "No transcript yet for this video."
    rows = [[i, fmt_ts(s["start"]), fmt_ts(s["end"]), s["text"]] for i, s in enumerate(tr["segments"])]
    return rows, f"Transcript method: **{tr['method']}** · {len(rows)} segments."


def do_batch(limit, per_topic, do_tr, progress=gr.Progress()):
    if not storage.list_topics():
        raise gr.Error("Generate topics first.")
    res = pipeline.batch_curate(int(limit), int(per_topic), do_tr, progress=_progress_cb(progress))
    rows = [[r["topic"], r["videos"], r["transcribed"], r["backend"], "; ".join(r["errors"])[:200]] for r in res]
    return rows, _video_rows(), gr.Dropdown(choices=_video_choices(True))


# --------------------------------------------------------------------------- tab 3: create

def on_select_video(video_id):
    v = storage.get_video(video_id) if video_id else None
    return gr.CheckboxGroup(choices=_segment_choices(video_id), value=[]), "", (v or {}).get("title", "")


def on_select_segments(video_id, idx):
    tr = storage.get_transcript(video_id) if video_id else None
    if not tr or not idx:
        return ""
    return " ".join(tr["segments"][i]["text"] for i in sorted(idx))


def do_create(source, video_id, seg_idx, selected_text, custom_text, clip_mode, title, style, voice,
              rewrite, target, layout, progress=gr.Progress()):
    cb = _progress_cb(progress)
    try:
        if source == "From transcript":
            if not video_id or not seg_idx:
                raise gr.Error("Pick a transcribed video and tick one or more transcript segments.")
            if clip_mode.startswith("Cut"):
                out, meta = pipeline.clip_from_transcript(video_id, seg_idx, mode="original", title=title,
                                                          layout="crop" if layout.startswith("Crop") else "blur",
                                                          progress=cb)
            elif selected_text.strip():  # user may have trimmed/edited the selected text
                out, meta = pipeline.clip_from_text(selected_text, title=title, style=style, voice=voice,
                                                    rewrite=rewrite, target_seconds=int(target),
                                                    video_id=video_id, progress=cb)
            else:
                out, meta = pipeline.clip_from_transcript(video_id, seg_idx, mode="generated", title=title,
                                                          style=style, voice=voice, rewrite=rewrite,
                                                          target_seconds=int(target), progress=cb)
        else:
            if not (custom_text or "").strip():
                raise gr.Error("Type or paste some text first.")
            out, meta = pipeline.clip_from_text(custom_text, title=title, style=style, voice=voice,
                                                rewrite=rewrite, target_seconds=int(target), progress=cb)
    except gr.Error:
        raise
    except Exception as e:
        logging.exception("render failed")
        raise gr.Error(f"Render failed: {str(e)[:400]}")
    script = meta.pop("script", None)
    script_md = ""
    if script:
        script_md = f"### {script.get('title', '')}\n*script by {script.get('generator')}*\n\n" + "\n".join(
            f"{i + 1}. {s['narration']}  \n   🖼️ _{s.get('image_prompt', '')}_" for i, s in enumerate(script["scenes"])
        )
    return str(out), meta, script_md


# --------------------------------------------------------------------------- tab 4: library

def library_rows():
    return [[c["id"], c["mode"], c["video_id"] or "", (c["script"] or "")[:90], os.path.basename(c["path"])]
            for c in storage.list_clips()]


def open_clip(evt: gr.SelectData):
    clips = storage.list_clips()
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row is None or row >= len(clips):
        return None, {}
    c = clips[row]
    return c["path"], json.loads(c["meta"] or "{}")


# --------------------------------------------------------------------------- layout

CSS = """
.gradio-container {max-width: 1280px !important}
#hero h1 {margin-bottom: 0}
#hero p {opacity: .8; margin-top: 4px}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="AI Video Content Curator") as demo:
        gr.Markdown(
            "# 🎬 AI Video Content Curator\n"
            "Trending educational topics → popular source videos → transcripts → new vertical shorts. "
            "Every stage uses free-tier APIs or open-source models, with offline fallbacks.",
            elem_id="hero",
        )
        with gr.Accordion("Active AI providers (configure via .env)", open=False):
            status = gr.Markdown(_status_md())

        # ---------------- Tab 1
        with gr.Tab("① Trending topics"):
            with gr.Row():
                n_topics = gr.Slider(10, 100, value=100, step=10, label="How many topics")
                use_signals = gr.Checkbox(True, label="Ground in live trend signals (YouTube / Google Trends / Wikipedia)")
            with gr.Row():
                gen_btn = gr.Button("✨ Generate topics", variant="primary")
                export_btn = gr.Button("Export CSV")
            topics_msg = gr.Markdown()
            topics_df = gr.Dataframe(headers=TOPIC_COLS, value=_topic_rows, wrap=True, interactive=False,
                                     max_height=520)
            export_file = gr.File(label="topics.csv", visible=True)

        # ---------------- Tab 2
        with gr.Tab("② Curate & transcribe"):
            with gr.Row():
                topic_dd = gr.Dropdown(choices=_topic_choices(), label="Topic", scale=3)
                per_topic = gr.Slider(1, 10, value=5, step=1, label="Videos to fetch", scale=1)
                find_btn = gr.Button("🔎 Find popular videos", variant="primary", scale=1)
            curate_msg = gr.Markdown()
            videos_df = gr.Dataframe(headers=VIDEO_COLS, value=_video_rows, interactive=False, wrap=True,
                                     max_height=300)
            with gr.Accordion("Add a video manually (YouTube URL or your own file)", open=False):
                with gr.Row():
                    url_in = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...", scale=3)
                    url_btn = gr.Button("Add URL", scale=1)
                with gr.Row():
                    upload = gr.File(label="Upload a video/audio file", file_types=["video", "audio"], scale=3)
                    upload_btn = gr.Button("Register upload", scale=1)
            with gr.Row():
                video_dd = gr.Dropdown(choices=_video_choices(), label="Video to transcribe", scale=3)
                force = gr.Checkbox(False, label="Re-transcribe", scale=1)
                tr_btn = gr.Button("🎙️ Transcribe", variant="primary", scale=1)
            tr_msg = gr.Markdown()
            transcript_df = gr.Dataframe(headers=["#", "start", "end", "text"], interactive=False, wrap=True,
                                         max_height=420)
            with gr.Accordion("Automate: run curation + transcription for the top N topics", open=False):
                with gr.Row():
                    b_limit = gr.Slider(1, 100, value=5, step=1, label="Topics")
                    b_per = gr.Slider(1, 5, value=1, step=1, label="Videos per topic")
                    b_tr = gr.Checkbox(True, label="Transcribe")
                    b_btn = gr.Button("▶ Run batch", variant="secondary")
                batch_df = gr.Dataframe(headers=["topic", "videos", "transcribed", "search backend", "errors"],
                                        interactive=False, wrap=True)

        # ---------------- Tab 3
        with gr.Tab("③ Create a short"):
            with gr.Row():
                with gr.Column(scale=5):
                    source = gr.Radio(["From transcript", "Custom text"], value="From transcript", label="Source")
                    with gr.Group(visible=True) as from_tr:
                        c_video = gr.Dropdown(choices=_video_choices(True), label="Transcribed video")
                        segs = gr.CheckboxGroup(choices=[], label="Select transcript segments")
                        sel_text = gr.Textbox(label="Selected text (editable — used for the AI-generated mode)",
                                              lines=4)
                        clip_mode = gr.Radio(
                            ["Cut original footage (captioned 9:16 clip)", "AI re-voiced & illustrated short"],
                            value="Cut original footage (captioned 9:16 clip)", label="Output",
                        )
                        layout = gr.Radio(["Blurred background (keeps full frame)", "Crop to fill"],
                                          value="Blurred background (keeps full frame)", label="Reframe")
                    with gr.Group(visible=False) as from_txt:
                        custom = gr.Textbox(label="Your text", lines=8,
                                            placeholder="Paste a paragraph, a script, or a few facts…")
                    title = gr.Textbox(label="On-screen title (optional)")
                    with gr.Accordion("AI generation options", open=True):
                        with gr.Row():
                            style = gr.Dropdown(list(STYLES), value="Cinematic photo", label="Image style")
                            voice = gr.Dropdown(EDGE_VOICES, value=settings.edge_voice, label="Voice (edge-tts)")
                        with gr.Row():
                            rewrite = gr.Checkbox(True, label="Let the LLM tighten it into a short-form script")
                            target = gr.Slider(15, 90, value=45, step=5, label="Target length (s)")
                    create_btn = gr.Button("🎬 Generate video", variant="primary", size="lg")
                with gr.Column(scale=4):
                    out_video = gr.Video(label="Result", height=640)
                    out_meta = gr.JSON(label="Render info")
                    out_script = gr.Markdown()

        # ---------------- Tab 4
        with gr.Tab("Library"):
            refresh_lib = gr.Button("Refresh")
            with gr.Row():
                lib_df = gr.Dataframe(headers=["id", "mode", "source video", "text", "file"], value=library_rows,
                                      interactive=False, wrap=True, scale=3)
                with gr.Column(scale=2):
                    lib_video = gr.Video(label="Preview (click a row)", height=520)
                    lib_meta = gr.JSON()

        # ---------------- wiring
        gen_btn.click(do_generate_topics, [n_topics, use_signals], [topics_df, topics_msg, topic_dd])
        export_btn.click(do_export_topics, None, export_file)

        find_btn.click(do_find_videos, [topic_dd, per_topic], [videos_df, curate_msg, video_dd])
        url_btn.click(do_add_url, url_in, [videos_df, video_dd])
        upload_btn.click(do_upload, upload, [videos_df, video_dd])
        video_dd.change(show_transcript, video_dd, [transcript_df, tr_msg])
        tr_btn.click(do_transcribe, [video_dd, force], [transcript_df, tr_msg, videos_df, c_video])
        b_btn.click(do_batch, [b_limit, b_per, b_tr], [batch_df, videos_df, c_video])

        source.change(lambda s: (gr.Group(visible=s == "From transcript"), gr.Group(visible=s != "From transcript")),
                      source, [from_tr, from_txt])
        c_video.focus(lambda v: gr.Dropdown(choices=_video_choices(True), value=v), c_video, c_video)
        c_video.change(on_select_video, c_video, [segs, sel_text, title])
        segs.change(on_select_segments, [c_video, segs], sel_text)
        create_btn.click(
            do_create,
            [source, c_video, segs, sel_text, custom, clip_mode, title, style, voice, rewrite, target, layout],
            [out_video, out_meta, out_script],
        ).then(library_rows, None, lib_df)

        refresh_lib.click(library_rows, None, lib_df)
        lib_df.select(open_clip, None, [lib_video, lib_meta])
        demo.load(_status_md, None, status)
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=2).launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
        allowed_paths=[str(settings.data_dir)],
    )
