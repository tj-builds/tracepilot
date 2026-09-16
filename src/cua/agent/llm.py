"""LLM client abstraction.

Everything that needs a model (the router, the planner "actor", and the validator
"critic") talks to an `LLMClient`, never to a vendor SDK directly. There is exactly
one provider: **OpenAI**. There is deliberately no offline stand-in in the shipped
system -- routing, discovery, and validation are model-driven, and the system refuses
to run them without a real model rather than silently faking it. (Unit tests inject
their own explicit `LLMClient` test double; that fake lives under `tests/`, never in
the product.)

The `complete()` signature accepts arbitrary `**context` so a client may use structured
inputs; the OpenAI provider ignores it -- it only needs the rendered `system`/`user`
strings.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

from ..config import Config


class LLMClient(ABC):
    name = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, **context) -> str:
        """Return the model's raw text response (expected to be a JSON object)."""


# --------------------------------------------------------------------------- #
# OpenAI provider                                                             #
# --------------------------------------------------------------------------- #
def _retry_after_seconds(msg: str, default: float) -> float:
    m = re.search(r"try again in ([0-9.]+)\s*s", msg)
    return (float(m.group(1)) + 1.0) if m else default


def _create_with_backoff(client, **kwargs):
    """chat.completions.create with bounded backoff on rate limits (HTTP 429).

    Low free-tier RPM caps are common, so instead of crashing we honour the
    provider's "try again in Ns" hint (or a default) and retry a few times.
    """
    delay = 8.0
    attempts = 8
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if ("rate limit" in msg or "rate_limit" in msg or "429" in msg) \
                    and attempt < attempts - 1:
                # Honour the provider's "try again in Ns" hint; wait at least a few
                # seconds so we ride out a low per-minute (RPM) window rather than
                # hammering it. Free-tier RPM caps (e.g. 10/min) are common.
                time.sleep(max(_retry_after_seconds(msg, delay), 6.0))
                continue
            raise


def _openai_chat(client, model: str, system: str, user: str) -> str:
    """Call an OpenAI-compatible chat endpoint, tolerating models that reject our
    preferred parameters. We prefer JSON mode + temperature=0 (determinism), but some
    newer/reasoning models forbid a custom temperature and some served models don't
    support `response_format`; we progressively drop those and retry rather than fail.
    Rate limits are absorbed with backoff (see `_create_with_backoff`).
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    param_sets = [
        {"temperature": 0, "response_format": {"type": "json_object"}},
        {"response_format": {"type": "json_object"}},  # model forbids temperature
        {"temperature": 0},                             # provider lacks JSON mode
        {},                                             # bare minimum
    ]
    last: Exception | None = None
    for extra in param_sets:
        try:
            resp = _create_with_backoff(client, model=model, messages=messages, **extra)
            return resp.choices[0].message.content or "{}"
        except Exception as exc:  # only fall through on parameter-shape rejections
            last = exc
            msg = str(exc).lower()
            if any(w in msg for w in ("temperature", "response_format",
                                      "unsupported", "json_object")):
                continue
            raise
    raise last  # type: ignore[misc]


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def complete(self, system: str, user: str, **context) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.cfg.openai_api_key)
        return _openai_chat(client, self.cfg.openai_model, system, user)


def build_client(cfg: Config) -> LLMClient:
    """The one real provider. No key -> hard error (there is no offline fallback).

    Routing, discovery, and validation are genuinely model-driven; refusing to run
    without a model is safer and more honest than silently substituting a heuristic.
    """
    if not cfg.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. This system requires a real OpenAI model for "
            "routing, discovery, and validation; there is no offline fallback. Set "
            "OPENAI_API_KEY in your environment/.env and retry.")
    return OpenAIClient(cfg)
