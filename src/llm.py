"""LLM provider configuration (single place to change provider).

Supported keys, checked in this order:
  GEMINI_API_KEY / GOOGLE_API_KEY  -> Google Gemini (via Google's OpenAI-compatible endpoint)
  OPENAI_API_KEY                   -> OpenAI (or any OpenAI-compatible server via OPENAI_BASE_URL)
Optional: ASK_MODEL overrides the default model of the chosen provider.
No key at all -> llm_available() is False and callers fall back to the rule-based parser.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-flash-lite-latest"  # fast, always-current alias; bigger models were overloaded when tested
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
SECRET_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ASK_MODEL")


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    base_url: str | None
    model: str


def get_provider() -> Provider | None:
    gem = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gem:
        return Provider("Gemini", gem, GEMINI_BASE_URL, os.getenv("ASK_MODEL") or GEMINI_DEFAULT_MODEL)
    if os.getenv("OPENAI_API_KEY"):
        return Provider("OpenAI", os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_BASE_URL") or None,
                        os.getenv("ASK_MODEL") or OPENAI_DEFAULT_MODEL)
    return None


def llm_available() -> bool:
    return get_provider() is not None


def describe() -> str:
    p = get_provider()
    return f"ON ({p.name}: {p.model})" if p else "OFF - rule-based parser (no API key set)"


def chat(messages: list[dict], json_mode: bool = False) -> str:
    """Send a chat request to the configured provider and return the reply text."""
    from openai import OpenAI

    p = get_provider()
    if p is None:
        raise RuntimeError("no LLM API key configured")
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = OpenAI(api_key=p.api_key, base_url=p.base_url, timeout=20, max_retries=1).chat.completions.create(
        model=p.model, temperature=0, messages=messages, **kwargs)
    return resp.choices[0].message.content
