"""Scene image generation.

Pollinations (free key, FLUX-family models) -> Hugging Face Inference (FLUX.1-schnell,
free monthly credits) -> procedural abstract background rendered locally with Pillow.
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import random
import urllib.parse
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter

from .config import Settings, settings

log = logging.getLogger(__name__)

STYLES = {
    "Cinematic photo": "cinematic photograph, dramatic lighting, shallow depth of field, highly detailed",
    "Flat illustration": "clean flat vector illustration, bold shapes, soft pastel palette, minimal",
    "3D render": "colorful 3D render, soft studio lighting, isometric, octane render",
    "Chalkboard": "hand-drawn chalk illustration on a dark green chalkboard, educational diagram style",
    "Anime": "anime style illustration, vibrant colors, detailed background",
}


def _full_prompt(prompt: str, style: str) -> str:
    suffix = STYLES.get(style, style)
    return f"{prompt}, {suffix}, vertical composition, no text, no watermark"


def _pollinations(prompt: str, w: int, h: int, seed: int, cfg: Settings) -> Image.Image:
    url = f"https://gen.pollinations.ai/image/{urllib.parse.quote(prompt)}"
    params = {"width": w, "height": h, "seed": seed, "model": cfg.pollinations_model, "nologo": "true"}
    r = requests.get(
        url, params=params, headers={"Authorization": f"Bearer {cfg.pollinations_api_key}"}, timeout=120
    )
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _hf(prompt: str, w: int, h: int, seed: int, cfg: Settings) -> Image.Image:
    r = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{cfg.hf_image_model}",
        headers={"Authorization": f"Bearer {cfg.hf_token}", "Accept": "image/png"},
        json={"inputs": prompt, "parameters": {"width": w, "height": h, "seed": seed, "num_inference_steps": 4}},
        timeout=180,
    )
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


PALETTES = [
    ["#0f172a", "#1e3a8a", "#3b82f6", "#22d3ee"],
    ["#1a1033", "#5b21b6", "#c026d3", "#f472b6"],
    ["#052e16", "#166534", "#22c55e", "#bef264"],
    ["#1c1917", "#9a3412", "#f97316", "#fde047"],
    ["#0c0a09", "#155e75", "#14b8a6", "#a7f3d0"],
    ["#172554", "#7c2d12", "#ea580c", "#fcd34d"],
]


def procedural(prompt: str, w: int, h: int, seed: int | None = None, label: str = "") -> Image.Image:
    """Deterministic abstract 'aurora' background - looks intentional, needs no network."""
    s = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) + (seed or 0)
    rnd = random.Random(s)
    pal = PALETTES[s % len(PALETTES)]
    small = (w // 4, h // 4)
    img = Image.new("RGB", small, pal[0])
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(9):
        color = Image.new("RGB", (1, 1), rnd.choice(pal[1:])).getpixel((0, 0))
        r = rnd.randint(small[0] // 4, small[0] // 1)
        cx, cy = rnd.randint(-r // 2, small[0] + r // 2), rnd.randint(0, small[1])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, rnd.randint(70, 150)))
    img = img.filter(ImageFilter.GaussianBlur(radius=small[0] / 7))
    # subtle diagonal grain lines for texture
    tex = ImageDraw.Draw(img, "RGBA")
    for i in range(0, small[0] + small[1], 9):
        tex.line([(i, 0), (i - small[1] * math.tan(0.35), small[1])], fill=(255, 255, 255, 6), width=1)
    img = img.resize((w, h), Image.LANCZOS)
    if label:
        _draw_label(img, label)
    return img


def _font(size: int):
    from PIL import ImageFont

    bundled = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Poppins-Bold.ttf"
    for p in (settings.font_path, str(bundled), "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def _draw_label(img: Image.Image, label: str) -> None:
    """Large 'concept card' keyword so offline scenes still carry meaning."""
    w, h = img.size
    label = label.upper()[:18]
    draw = ImageDraw.Draw(img, "RGBA")
    size = int(w * 0.16)
    font = _font(size)
    while draw.textlength(label, font=font) > w * 0.86 and size > 30:
        size = int(size * 0.9)
        font = _font(size)
    tw = draw.textlength(label, font=font)
    y = int(h * 0.36)
    draw.rounded_rectangle([w * 0.06, y - size * 0.45, w * 0.94, y + size * 1.55], radius=int(w * 0.04),
                           fill=(255, 255, 255, 28), outline=(255, 255, 255, 60), width=3)
    draw.text(((w - tw) / 2, y), label, font=font, fill=(255, 255, 255, 235))


def generate(
    prompt: str, out: Path, style: str = "Cinematic photo", seed: int = 7, cfg: Settings = settings,
    allow_remote: bool = True, label: str | None = None,
) -> tuple[Path, str]:
    """Return (image_path, provider_used). Generates at 768x1344 (9:16) for speed."""
    w, h = 768, 1344
    full = _full_prompt(prompt, style)
    attempts = []
    if allow_remote and cfg.pollinations_api_key:
        attempts.append(("pollinations", lambda: _pollinations(full, w, h, seed, cfg)))
    if allow_remote and cfg.hf_token:
        attempts.append(("hf-flux", lambda: _hf(full, w, h, seed, cfg)))
    for name, fn in attempts:
        try:
            img = fn()
            img.save(out)
            return out, name
        except Exception as e:
            log.info("Image provider %s failed: %s", name, e)
    label = prompt.split(",")[0].strip() if label is None else label
    procedural(prompt, w, h, seed=seed, label=label).save(out)
    return out, "procedural"
