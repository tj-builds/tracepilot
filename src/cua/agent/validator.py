"""The validator ("critic") -- a guardrail over every proposed action.

Each action the planner proposes is checked *before* it executes, in two layers:

  1. A deterministic layer that reuses the same `Policy` the rest of the system
     enforces (navigation allowlist + action-risk gating). This can never be talked
     out of a decision -- if policy denies, the action is denied.
  2. An LLM critic that judges the action *semantically*: is it on-goal, does it use
     only supplied data, is it about to take an irreversible step the goal never
     asked for, is it walking past a condition a human should adjudicate?

The verdict is advisory-plus-hard: the deterministic layer is authoritative for
policy violations; the model layer adds judgement the allowlist cannot express. This
is the "validate each action" guardrail the discovery loop consults on every step.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..config import Config
from ..safety.policy import Policy
from ..schema.artifact import ActionType, RiskClass
from ..surface.base import Observation
from .decision import Decision
from .llm import LLMClient, build_client
from .prompts import VALIDATOR_SYSTEM_PROMPT, build_validator_prompt

_ACTION_MAP = {
    "click": ActionType.CLICK,
    "type": ActionType.TYPE,
    "select": ActionType.SELECT,
    "press": ActionType.PRESS,
    "read": ActionType.READ,
}


class ValidationVerdict(BaseModel):
    approved: bool = True
    risk: str = "none"
    reason: str = ""
    concerns: list[str] = Field(default_factory=list)
    layer: str = "validator"  # which layer produced the verdict


class ActionValidator:
    def __init__(self, client: LLMClient, policy: Policy):
        self.client = client
        self.policy = policy

    def validate(self, goal: str, params: dict, obs: Observation,
                 decision: Decision) -> ValidationVerdict:
        # ---- Layer 1: deterministic policy (authoritative). ---- #
        policy_verdict = self._policy_check(decision)
        if policy_verdict is not None:
            return policy_verdict

        # ---- Layer 2: LLM critic (semantic judgement). ---- #
        payload = decision.model_dump(exclude_none=True)
        raw = self.client.complete(
            VALIDATOR_SYSTEM_PROMPT,
            build_validator_prompt(goal, params, obs.digest(), payload,
                                   page_text=obs.text),
            mode="validate", goal=goal, params=params, observation=obs,
            decision=payload,
        )
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # A malformed critic response must not silently approve.
            return ValidationVerdict(approved=False, risk="medium",
                                     reason="validator response was not parseable",
                                     layer="validator")
        return ValidationVerdict(layer="validator", **data)

    def _policy_check(self, decision: Decision) -> ValidationVerdict | None:
        if decision.action == "navigate":
            d = self.policy.check_navigation(decision.url or "")
            if not d.allowed:
                return ValidationVerdict(approved=False, risk="high",
                                         reason=d.reason, layer="policy")
            return None
        act = _ACTION_MAP.get(decision.action)
        if act is None:  # done / give_up need no policy gate
            return None
        risk = RiskClass(decision.risk)
        d = self.policy.check_action(act, risk)
        if not d.allowed:
            return ValidationVerdict(approved=False, risk="high",
                                     reason=d.reason, layer="policy")
        return None


def build_validator(cfg: Config, policy: Policy) -> ActionValidator:
    return ActionValidator(build_client(cfg), policy)
