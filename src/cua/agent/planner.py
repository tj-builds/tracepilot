"""The planner ("actor").

A single, application-agnostic planner drives discovery: it observes the current
screen, decides the next action, and returns a structured `Decision`. All model
access goes through an `LLMClient` (the OpenAI provider in production; an injected
test double under tests/), so the discovery loop is decoupled from the transport.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ..config import Config
from ..surface.base import Observation
from .decision import Decision
from .llm import LLMClient, build_client
from .prompts import SYSTEM_PROMPT, build_user_prompt


class Planner(ABC):
    name: str = "planner"

    @abstractmethod
    def decide(self, goal: str, params: dict, obs: Observation,
               history: list[str], objectives: dict | None = None) -> Decision: ...


class LLMPlanner(Planner):
    name = "llm"

    def __init__(self, client: LLMClient):
        self.client = client
        self.name = f"llm:{client.name}"

    def decide(self, goal, params, obs, history, objectives=None) -> Decision:
        user = build_user_prompt(goal, params, obs.digest(), history, objectives,
                                 page_text=obs.text)
        raw = self.client.complete(
            SYSTEM_PROMPT, user,
            mode="plan", goal=goal, params=params, observation=obs,
            history=history, objectives=objectives or {},
        )
        return _parse_decision(raw)


def _parse_decision(raw: str) -> Decision:
    return Decision(**_extract_json(raw))


def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()
    # Tolerate models that wrap JSON in fences.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def build_planner(cfg: Config) -> Planner:
    return LLMPlanner(build_client(cfg))
