"""Test-only LLM double.

This is the deterministic, offline stand-in the unit tests inject wherever the code
expects an `LLMClient`. It deliberately lives under `tests/` and NOT in `src/`: the
shipped system requires a real OpenAI model (see `cua.agent.llm.build_client`) and has
no offline fallback. This fake exists only so the loop/parsing/wiring can be exercised
in CI without a network or a key.

It emits the SAME JSON contract the real model returns for every role
(route / plan / validate), reasoning generically over the accessibility observation
and the goal rather than hard-coding any specific app's screens.
"""
from __future__ import annotations

import json
import re

from cua.agent.llm import LLMClient
from cua.replay.errors import Condition, classify_page
from cua.surface.base import Observation

_INTERSTITIAL_WORDS = {"continue", "ok", "okay", "acknowledge", "proceed", "dismiss"}
_MUTATION_WORDS = {"create", "submit", "confirm", "delete", "remove", "transfer",
                   "pay", "withdraw", "send"}
_MUTATING_GOAL_WORDS = {"create", "open", "submit", "transfer", "delete", "update",
                        "change", "new", "add", "send", "withdraw", "pay"}
_SKIP_CONTROL_WORDS = {"back", "cancel", "logout", "sign out"}


def _tokens(s: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _norm_key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


class FakeLLMClient(LLMClient):
    """Deterministic, generic stand-in for a real model (tests only)."""

    name = "fake"

    def complete(self, system: str, user: str, **context) -> str:
        mode = context.get("mode", "plan")
        if mode == "route":
            return json.dumps(self._route(context))
        if mode == "validate":
            return json.dumps(self._validate(context))
        return json.dumps(self._plan(context))

    # ------------------------------------------------------------- routing #
    def _route(self, ctx: dict) -> dict:
        request: str = ctx.get("request", "")
        catalog: list[dict] = ctx.get("catalog", [])
        req_tokens = _tokens(request)

        best, best_score = None, 0.0
        for cap in catalog:
            hay = _tokens(cap.get("name", "")) | _tokens(cap.get("description", "")) \
                | _tokens(cap.get("goal", ""))
            score = len(req_tokens & hay)
            if score > best_score:
                best, best_score = cap, float(score)

        args: dict = {}
        m = re.search(r"\b(\d{5,7})\b", request)
        if m:
            args["member_id"] = m.group(1)

        if best and best_score >= 2:
            return {"mode": "invoke", "capability": best["name"], "args": args,
                    "reason": f"request overlaps existing capability '{best['name']}'"}

        mutating = bool(req_tokens & _MUTATING_GOAL_WORDS)
        proposed = {
            "name": re.sub(r"[^a-z0-9]+", "_", request.lower()).strip("_")[:40]
                    or "new_capability",
            "description": request,
            "goal": request,
            "success_text": "balance" if "balance" in req_tokens else "",
            "outputs": ([{"name": "balance", "label": "balance"}]
                        if "balance" in req_tokens else []),
        }
        return {"mode": "create", "capability": None, "args": args,
                "proposed": proposed,
                "reason": "no existing capability matched; discovery required",
                "mutating": mutating}

    # ------------------------------------------------------------- planning #
    def _plan(self, ctx: dict) -> dict:
        goal: str = ctx.get("goal", "")
        params: dict = ctx.get("params", {})
        obs: Observation = ctx["observation"]
        history: list[str] = ctx.get("history", [])
        objectives: dict = ctx.get("objectives") or {}
        text = obs.text or ""

        det = classify_page(obs.url, text)
        if det.condition in (Condition.BUSINESS, Condition.HARD):
            return {"action": "give_up", "reason": det.message or "unrecoverable state"}
        if det.condition == Condition.RECOVERABLE:
            btn = self._find_control(obs, "button", _INTERSTITIAL_WORDS)
            if btn:
                return {"action": "click", "ref": btn.ref, "risk": "safe",
                        "reason": "dismiss interstitial notice to proceed"}

        for e in obs.elements:
            if e.editable and not (e.value or "").strip():
                key = self._match_param(e, params)
                if not key:
                    continue
                if e.role in ("combobox", "listbox", "select"):
                    return {"action": "select", "ref": e.ref,
                            "value": "{{%s}}" % key, "risk": "reversible",
                            "reason": f"select value for '{e.name}'"}
                return {"action": "type", "ref": e.ref, "text": "{{%s}}" % key,
                        "risk": "reversible", "reason": f"enter '{e.name}'"}

        for o in objectives.get("outputs", []):
            label = o.get("label") or o.get("name")
            if not label:
                continue
            if label.lower() in text.lower() and not self._already_read(o["name"], history):
                return {"action": "read", "output_name": o["name"], "value": label,
                        "risk": "safe", "reason": f"extract '{o['name']}'"}

        if self._success_met(objectives, obs, history):
            return {"action": "done", "reason": "goal reached and outputs captured"}

        ctl = self._forward_control(obs, goal)
        if ctl:
            risk = "irreversible" if _tokens(ctl.name) & _MUTATION_WORDS else "safe"
            return {"action": "click", "ref": ctl.ref, "risk": risk,
                    "reason": f"advance via '{ctl.name}'"}

        return {"action": "give_up", "reason": "no actionable control advances the goal"}

    # ----------------------------------------------------------- validation #
    def _validate(self, ctx: dict) -> dict:
        goal: str = ctx.get("goal", "")
        decision: dict = ctx.get("decision", {})
        obs: Observation = ctx["observation"]
        action = decision.get("action")
        concerns: list[str] = []

        goal_mutates = bool(_tokens(goal) & _MUTATING_GOAL_WORDS)
        if decision.get("risk") == "irreversible" and not goal_mutates:
            return {"approved": False, "risk": "high",
                    "reason": "irreversible action proposed for a read-only goal",
                    "concerns": ["state change not implied by the goal"]}

        if action in ("click", "type", "select"):
            ref = decision.get("ref")
            if not ref or not any(e.ref == ref for e in obs.elements):
                return {"approved": False, "risk": "medium",
                        "reason": "target control is not present on the current screen",
                        "concerns": [f"unknown ref '{ref}'"]}

        if action in ("type", "select"):
            val = decision.get("text") or decision.get("value") or ""
            if not re.fullmatch(r"\{\{\s*\w+\s*\}\}", val.strip()):
                concerns.append("value is a literal, not a supplied parameter")

        return {"approved": True, "risk": "low" if concerns else "none",
                "reason": "action is on-goal and within scope", "concerns": concerns}

    # ---------------------------------------------------------------- utils #
    @staticmethod
    def _already_read(name: str, history: list[str]) -> bool:
        return any(h.startswith(f"read {name}") for h in history)

    @staticmethod
    def _match_param(elem, params: dict) -> str | None:
        name_key = _norm_key(elem.name)
        name_tokens = _tokens(elem.name)
        for k in params:
            if k.startswith("_"):
                continue
            if _norm_key(k) == name_key or _tokens(k) & name_tokens:
                return k
        return None

    @staticmethod
    def _find_control(obs: Observation, role: str, words: set[str]):
        for e in obs.elements:
            if e.role == role and _tokens(e.name) & words:
                return e
        return None

    @staticmethod
    def _forward_control(obs: Observation, goal: str):
        goal_tokens = _tokens(goal)
        best, best_score = None, -1.0
        for e in obs.elements:
            if e.role not in ("button", "link"):
                continue
            nl = _tokens(e.name)
            if nl & _SKIP_CONTROL_WORDS:
                continue
            score = float(len(nl & goal_tokens))
            if nl & {"search", "sign", "in", "submit", "continue", "create", "next"}:
                score += 0.5
            if score > best_score:
                best, best_score = e, score
        return best

    def _success_met(self, objectives: dict, obs: Observation,
                     history: list[str]) -> bool:
        success = objectives.get("success")
        if not success:
            return False
        typ = success.get("type", "text_present")
        expr = success.get("expression", "")
        if typ == "url_matches":
            ok = re.search(expr, obs.url or "") is not None
        else:
            ok = expr.lower() in (obs.text or "").lower()
        if not ok:
            return False
        for o in objectives.get("outputs", []):
            if not self._already_read(o["name"], history):
                return False
        return True
