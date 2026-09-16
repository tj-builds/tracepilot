"""The capability router -- the top of the agentic loop.

Given a natural-language request, the router decides, by *semantic meaning*, whether
an already-learned capability accomplishes it (INVOKE) or whether none fits and one
must be discovered first (CREATE). This is the "the system decides if the workflow is
present, or an agent creates it" layer that sits above discovery/replay.

It is deliberately thin and reuses the same `LLMClient` as the planner/validator, and
the same `CapabilityRegistry` catalog an external agent would see. In production this
role belongs to the calling agent's function-calling layer; here it is a real,
self-contained stand-in so the whole thing runs from one natural-language entrypoint.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..catalog.registry import CapabilityRegistry
from .llm import LLMClient
from .planner import _extract_json
from .prompts import ROUTER_SYSTEM_PROMPT, build_router_prompt


class RouteDecision(BaseModel):
    mode: str  # "invoke" | "create"
    capability: str | None = None
    args: dict = Field(default_factory=dict)
    reason: str = ""
    # Present when mode == "create":
    proposed_name: str | None = None
    proposed_description: str | None = None
    proposed_goal: str | None = None
    success_text: str | None = None
    outputs: list[dict] = Field(default_factory=list)
    mutating: bool = False


class CapabilityRouter:
    def __init__(self, client: LLMClient, registry: CapabilityRegistry | None = None):
        self.client = client
        self.registry = registry or CapabilityRegistry()

    def _catalog(self) -> list[dict]:
        cat = []
        for a in self.registry.list():
            cat.append({"name": a.name, "description": a.description, "goal": a.goal,
                        "params": [p.name for p in a.params if not p.sensitive]})
        return cat

    def route(self, request: str) -> RouteDecision:
        catalog = self._catalog()
        raw = self.client.complete(
            ROUTER_SYSTEM_PROMPT, build_router_prompt(request, catalog),
            mode="route", request=request, catalog=catalog)
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            return RouteDecision(mode="create", reason="router response unparseable")
        return self._normalize(data)

    @staticmethod
    def _normalize(data: dict) -> RouteDecision:
        proposed = data.get("proposed") or {}
        return RouteDecision(
            mode=data.get("mode", "create"),
            capability=data.get("capability"),
            args=data.get("args") or {},
            reason=data.get("reason", ""),
            proposed_name=proposed.get("name"),
            proposed_description=proposed.get("description"),
            proposed_goal=proposed.get("goal") or data.get("request"),
            success_text=proposed.get("success_text"),
            outputs=proposed.get("outputs") or [],
            mutating=bool(proposed.get("mutating", False)),
        )
