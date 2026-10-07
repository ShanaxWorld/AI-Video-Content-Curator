"""LLM access through any OpenAI-compatible chat endpoint.

Groq, Google Gemini, OpenRouter and a local Ollama server all speak the same
`/chat/completions` protocol, so one small client covers every free option.
Providers are tried in order; if all fail (or none is configured) callers get an
`LLMUnavailable` error and use their own offline fallback.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import requests

from .config import Settings, settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


def providers(cfg: Settings = settings) -> list[Provider]:
    out: list[Provider] = []
    if cfg.groq_api_key:
        out.append(Provider("Groq", "https://api.groq.com/openai/v1", cfg.groq_api_key, cfg.groq_model))
    if cfg.gemini_api_key:
        out.append(
            Provider(
                "Gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                cfg.gemini_api_key,
                cfg.gemini_model,
            )
        )
    if cfg.openrouter_api_key:
        out.append(
            Provider("OpenRouter", "https://openrouter.ai/api/v1", cfg.openrouter_api_key, cfg.openrouter_model)
        )
    if cfg.ollama_model:
        out.append(Provider("Ollama", cfg.ollama_base_url, "ollama", cfg.ollama_model))
    return out


def active_provider_name(cfg: Settings = settings) -> str:
    ps = providers(cfg)
    if not ps:
        return "none (offline fallbacks)"
    return " -> ".join(f"{p.name} ({p.model})" for p in ps)


def chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    json_mode: bool = False,
    cfg: Settings = settings,
) -> str:
    errors = []
    for p in providers(cfg):
        body = {
            "model": p.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and p.name in {"Groq", "Gemini", "OpenRouter"}:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(
                f"{p.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            if content.strip():
                return content
            errors.append(f"{p.name}: empty response")
        except Exception as e:  # try the next provider
            log.warning("LLM provider %s failed: %s", p.name, e)
            errors.append(f"{p.name}: {e}")
    raise LLMUnavailable("; ".join(errors) or "no LLM provider configured")


def chat_json(prompt: str, system: str = "Reply with valid JSON only.", **kw) -> object:
    """Ask for JSON and parse it robustly (handles ```json fences and stray prose)."""
    text = chat(prompt, system=system, json_mode=True, **kw)
    return extract_json(text)


def extract_json(text: str) -> object:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost {...} or [...] block
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")
