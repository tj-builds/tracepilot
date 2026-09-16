"""The replay result contract.

The single most important distinction here (per the brief) is between:

  * BUSINESS_OUTCOME    -- a legitimate answer the caller needs ("no such member").
                           NOT a crash. Replay succeeded at *executing*; the app
                           gave a valid negative/branching result.
  * RECOVERED           -- a recoverable condition was detected & handled
                           (dismissed an interstitial, retried a transient load).
  * SUCCESS             -- goal reached, checkpoint verified, outputs returned.
  * ESCALATED           -- could not safely proceed; handed to a human.
  * RECOVERY_EXHAUSTED  -- a recoverable condition was retried up to the bound and
                           still did not clear; bounded, evidence-backed give-up.
  * HARD_FAILURE        -- unexpected/undebuggable; stop and surface detail.
  * BLOCKED_BY_POLICY   -- the safety allowlist refused the action.
  * NEEDS_INPUT         -- the request is incomplete: a required input the caller must
                           supply is missing. We refuse to fabricate it (no recorded
                           example, no hard-coded default), and ask the caller / a human
                           instead. Nothing was executed against the surface.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    RECOVERED = "recovered"
    ESCALATED = "escalated"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    RECOVERY_EXHAUSTED = "recovery_exhausted"  # tried to recover N times; still blocked
    NEEDS_INPUT = "needs_input"                # incomplete request; refuse to fabricate
    HARD_FAILURE = "hard_failure"


class StepResult(BaseModel):
    step_index: int
    action: str
    ok: bool
    detail: str = ""
    locator_used: str | None = None
    # Which rung of the locator ladder fired (0 = primary), and whether that means a
    # more-robust candidate had already failed (drift signal).
    rung: int | None = None
    used_fallback: bool = False
    recovered_from: str | None = None
    duration_ms: int | None = None


class ReplayResult(BaseModel):
    outcome: Outcome
    artifact_id: str
    artifact_version: int
    # Populated on SUCCESS (typed outputs the caller receives).
    outputs: dict[str, Any] = Field(default_factory=dict)
    # For BUSINESS_OUTCOME: a machine-readable code, e.g. "member_not_found".
    business_code: str | None = None
    # For NEEDS_INPUT: the required inputs the caller did not supply.
    missing_inputs: list[str] = Field(default_factory=list)
    message: str = ""
    # Debuggable failure context.
    failed_step: int | None = None
    expected: str | None = None
    observed: str | None = None
    # Trace of every step.
    steps: list[StepResult] = Field(default_factory=list)
    # Evidence pointers (log file, failure screenshot, etc.).
    evidence: dict[str, str] = Field(default_factory=dict)

    @property
    def is_terminal_success(self) -> bool:
        return self.outcome in {Outcome.SUCCESS, Outcome.BUSINESS_OUTCOME}
