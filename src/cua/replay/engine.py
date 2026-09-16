"""Deterministic replay engine -- the production execution path.

No LLM is in the decision loop. The engine walks the artifact's ordered steps,
resolves each control through the ordered locator candidates, verifies the success
checkpoint, and returns typed outputs. Every observed page is run through the error
taxonomy so the engine responds deliberately to runtime conditions instead of
blindly proceeding.
"""
from __future__ import annotations

import re
import time
import uuid

from ..config import Config
from ..evidence.logger import EvidenceLog
from ..safety.policy import Policy, redact
from ..schema.artifact import (
    ActionType,
    Artifact,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    RiskClass,
)
from ..schema.results import Outcome, ReplayResult, StepResult
from ..surface.web import WebSurface
from .errors import Condition, classify_page, verify_checkpoint

_TEMPLATE = re.compile(r"\{\{\s*([\w]+)\s*\}\}")

_CONTINUE_LOCATOR = Locator(
    description="interstitial Continue button",
    candidates=[LocatorCandidate(strategy=LocatorStrategy.ROLE, role="button",
                                 value="Continue")],
)


class MissingParam(Exception):
    pass


class ReplayEngine:
    def __init__(self, surface: WebSurface, policy: Policy, evidence: EvidenceLog,
                 cfg: Config, escalation=None,
                 escalate_codes: set[str] | None = None,
                 apply_variant_overrides: bool = True):
        self.surface = surface
        self.policy = policy
        self.ev = evidence
        self.cfg = cfg
        self.escalation = escalation
        # Conditions that should be routed to a human instead of auto-recovered.
        self.escalate_codes = escalate_codes or set()
        # When False, an active variant still selects the tenant base_url but its
        # locator/checkpoint overrides are ignored -- used to demonstrate that the
        # override is load-bearing (the run fails without it).
        self.apply_variant_overrides = apply_variant_overrides
        self._variant = None
        # Recovery is bounded: a recoverable condition is retried at most this many
        # times, and only counts if verified cleared afterwards.
        self.max_recovery_attempts = 2

    # ----------------------------------------------------------------- API #
    def replay(self, artifact: Artifact, params: dict) -> ReplayResult:
        result = ReplayResult(outcome=Outcome.HARD_FAILURE,
                              artifact_id=artifact.id,
                              artifact_version=artifact.version)
        try:
            self._check_params(artifact, params)
        except MissingParam as exc:
            result.message = str(exc)
            result.outcome = Outcome.HARD_FAILURE
            self.ev.finish("hard_failure", reason=str(exc))
            return result

        # Keep the artifact reachable so escalation requests can carry real context
        # (which capability + the original goal), not just the opaque artifact id.
        self._artifact = artifact
        # Resolve the active tenant variant (if any) once, up front.
        self._variant = self._active_variant(artifact, params)
        if self._variant:
            self.ev.event("variant_active", variant_id=self._variant.variant_id,
                          overrides_applied=self.apply_variant_overrides,
                          base_url=self._variant.base_url)

        self.ev.event("replay_started", capability=artifact.name,
                      params=_safe_params(artifact, params))

        # The login screen is an expected part of the flow until we pass it once;
        # only *after* that does landing back on login mean a session timeout.
        self._passed_login = False
        outputs: dict = {}
        for step in artifact.steps:
            sr = self._execute_step(artifact, step, params, outputs, result)
            result.steps.append(sr)
            if result.outcome in (Outcome.BUSINESS_OUTCOME, Outcome.BLOCKED_BY_POLICY,
                                  Outcome.ESCALATED,
                                  Outcome.RECOVERY_EXHAUSTED) and not sr.ok:
                self.ev.finish(result.outcome.value, business_code=result.business_code,
                               message=result.message)
                return result
            if result.outcome == Outcome.HARD_FAILURE and not sr.ok:
                self._attach_failure_evidence(result)
                return result

        # Drift signal: how many targeting steps needed a fallback rung. On a stable
        # UI this should be 0; a rising count is the early-warning that a tenant/vendor
        # relabelled a control and the artifact (or its variant) needs attention.
        targeting = [s for s in result.steps if s.rung is not None]
        fallbacks = [s for s in targeting if s.used_fallback]
        if targeting:
            self.ev.event("drift_summary", targeting_steps=len(targeting),
                          fallback_steps=len(fallbacks),
                          steps=[f"step {s.step_index}: {s.locator_used}"
                                 for s in fallbacks])
            if fallbacks:
                result.evidence["drift"] = (
                    f"{len(fallbacks)}/{len(targeting)} targeting steps used a "
                    f"fallback locator")

        # Final checkpoint verification (variant may relabel the asserted text).
        checkpoint = self._effective_checkpoint(artifact)
        url, text = self.surface.current_url(), self.surface.page_text()
        if self._checkpoint_holds(checkpoint, url, text):
            # Success requires BOTH the checkpoint AND that every declared output was
            # actually extracted with a non-empty value -- a checkpoint passing while a
            # declared output is missing is a failure, not a success with a null field.
            missing_out = [o.name for o in artifact.outputs
                           if str(outputs.get(o.name) or "").strip() == ""]
            if missing_out:
                result.outcome = Outcome.HARD_FAILURE
                result.message = ("checkpoint held but declared output(s) not "
                                  f"extracted: {', '.join(missing_out)}")
                result.expected = f"non-empty outputs: {', '.join(missing_out)}"
                result.observed = "one or more declared outputs were empty/None"
                self._attach_failure_evidence(result)
                return result
            result.outcome = Outcome.SUCCESS
            result.outputs = outputs
            result.message = "checkpoint verified; goal reached"
            self.ev.finish("success", outputs=_redact_outputs(artifact, outputs))
        else:
            result.outcome = Outcome.HARD_FAILURE
            result.message = "success checkpoint not satisfied"
            result.expected = checkpoint.expression
            result.observed = url
            self._attach_failure_evidence(result)
        return result

    # ------------------------------------------------------------- stepping #
    def _execute_step(self, artifact, step, params, outputs, result) -> StepResult:
        t0 = time.time()
        sr = StepResult(step_index=step.index, action=step.action.value, ok=True)
        locator = self._effective_locator(artifact, step, params)

        try:
            if step.action == ActionType.NAVIGATE:
                self._do_navigate(step, sr, result,
                                  route=self._effective_route(artifact, step))
            elif step.action == ActionType.TYPE:
                self._do_action(sr, result, step,
                                lambda: self.surface.type_text(
                                    locator, self._resolve(step.value, params)))
            elif step.action == ActionType.SELECT:
                self._do_action(sr, result, step,
                                lambda: self.surface.select(
                                    locator, self._resolve(step.value, params)))
            elif step.action == ActionType.CLICK:
                self._do_click(step, locator, sr, result)
            elif step.action == ActionType.PRESS:
                if self._gate(step, sr, result):
                    res = self.surface.press(step.key or "Enter")
                    sr.ok, sr.detail = res.ok, res.detail
            elif step.action == ActionType.READ:
                if self._gate(step, sr, result):
                    self._do_read(step, locator, sr, result, outputs)
            elif step.action in (ActionType.WAIT_FOR, ActionType.ASSERT):
                if self._gate(step, sr, result):
                    sr.detail = "noop"
        except Exception as exc:  # unexpected -> hard failure
            sr.ok = False
            sr.detail = f"exception: {exc}"
            result.outcome = Outcome.HARD_FAILURE
            result.failed_step = step.index
            result.observed = str(exc)

        # After the action, inspect the resulting page for runtime conditions.
        if sr.ok:
            self._post_step_detection(step, sr, result)

        sr.duration_ms = int((time.time() - t0) * 1000)
        self.ev.event("step", index=step.index, action=step.action.value,
                      ok=sr.ok, detail=sr.detail, locator=sr.locator_used,
                      recovered_from=sr.recovered_from)
        return sr

    def _post_step_detection(self, step, sr, result):
        url, text = self.surface.current_url(), self.surface.page_text()
        on_login = url.rstrip("/").endswith("/login")
        det = classify_page(url, text, expect_login=not self._passed_login)
        if not on_login:
            self._passed_login = True
        if det.condition == Condition.NONE:
            return

        if det.condition == Condition.BUSINESS:
            result.outcome = Outcome.BUSINESS_OUTCOME
            result.business_code = det.code
            result.message = det.message
            result.observed = url
            sr.ok = False
            sr.detail = f"business outcome: {det.code}"
            return

        if det.condition == Condition.RECOVERABLE:
            if det.code in self.escalate_codes and self.escalation:
                self._escalate(step, det.message, result, sr)
            else:
                self._auto_recover(step, det, sr, result)
            return

        if det.condition == Condition.SESSION_TIMEOUT:
            if self.escalation:
                self._escalate(step, det.message, result, sr)
            else:
                result.outcome = Outcome.ESCALATED
                result.message = det.message
                sr.ok = False
                sr.detail = "session timeout; no escalation handler configured"
            return

        # HARD
        if self.escalation:
            self._escalate(step, det.message, result, sr)
        else:
            result.outcome = Outcome.HARD_FAILURE
            result.message = det.message
            sr.ok = False

    # --------------------------------------------------------- action verbs #
    def _do_navigate(self, step, sr, result, route=None):
        url = route if route is not None else (step.route or "")
        dec = self.policy.check_navigation(url)
        if not dec.allowed:
            result.outcome = Outcome.BLOCKED_BY_POLICY
            result.message = dec.reason
            sr.ok, sr.detail = False, f"blocked: {dec.reason}"
            return
        res = self._with_retry(lambda: self.surface.navigate(url))
        sr.ok, sr.detail = res.ok, res.detail

    def _do_action(self, sr, result, step, fn):
        dec = self.policy.check_action(step.action, step.risk)
        if not dec.allowed:
            result.outcome = Outcome.BLOCKED_BY_POLICY
            result.message = dec.reason
            sr.ok, sr.detail = False, f"blocked: {dec.reason}"
            return
        res = self._with_retry(fn)
        sr.ok, sr.detail, sr.locator_used = res.ok, res.detail, res.locator_used
        sr.rung = res.rung
        sr.used_fallback = bool(res.rung)
        if not res.ok:
            result.failed_step = step.index
            result.expected = "control resolvable & actionable"
            result.observed = res.detail

    def _gate(self, step, sr, result) -> bool:
        """Action-type allowlist check for the verbs that don't have their own gate
        (READ / PRESS / WAIT_FOR / ASSERT). Returns True if the action may proceed;
        otherwise sets BLOCKED_BY_POLICY on the result. This makes the "every action
        passes through the policy" guarantee actually true, not just for the mutating
        verbs."""
        dec = self.policy.check_action(step.action, step.risk)
        if not dec.allowed:
            result.outcome = Outcome.BLOCKED_BY_POLICY
            result.message = dec.reason
            sr.ok, sr.detail = False, f"blocked: {dec.reason}"
            return False
        return True

    def _do_read(self, step, locator, sr, result, outputs):
        """Extract a declared output. A read whose locator does not resolve (or whose
        cell is empty) is a hard failure at this step, not a silent None that later
        masquerades as success."""
        val = self.surface.read(locator)
        sr.locator_used = _loc_str(locator)
        name = step.output_name or "value"
        if val is None or str(val).strip() == "":
            sr.ok = False
            result.outcome = Outcome.HARD_FAILURE
            result.failed_step = step.index
            result.expected = f"non-empty value for output '{name}'"
            result.observed = "locator did not resolve or extracted an empty value"
            sr.detail = f"read '{name}' returned no value"
            return
        outputs[name] = val
        sr.detail = f"read {name}"

    def _do_click(self, step, locator, sr, result):
        dec = self.policy.check_action(ActionType.CLICK, step.risk)
        if not dec.allowed:
            result.outcome = Outcome.BLOCKED_BY_POLICY
            result.message = dec.reason
            sr.ok, sr.detail = False, f"blocked: {dec.reason}"
            return
        if dec.requires_confirmation:
            self.ev.event("irreversible_confirmed", step=step.index)
        res = self._with_retry(lambda: self.surface.click(locator))
        sr.ok, sr.detail, sr.locator_used = res.ok, res.detail, res.locator_used
        sr.rung = res.rung
        sr.used_fallback = bool(res.rung)
        if not res.ok:
            result.failed_step = step.index
            result.expected = "clickable control"
            result.observed = res.detail

    # ----------------------------------------------------------- recovery #
    def _auto_recover(self, step, det, sr, result):
        """Bounded, evidence-backed recovery.

        A recovery only *counts* if the blocking condition is actually gone
        afterwards. We re-observe and re-classify after each attempt; if the same
        condition keeps returning past `max_recovery_attempts`, we stop with
        RECOVERY_EXHAUSTED (or escalate) instead of looping or falsely proceeding.
        """
        code = det.code
        for attempt in range(1, self.max_recovery_attempts + 1):
            if not self._recovery_action(det):
                break
            url, text = self.surface.current_url(), self.surface.page_text()
            redet = classify_page(url, text, expect_login=not self._passed_login)
            if redet.code != code:  # condition verified cleared
                sr.recovered_from = code
                sr.detail = f"recovered from '{code}' (verified after {attempt} attempt(s))"
                self.ev.event("recovered", condition=code, attempts=attempt,
                              verified=True)
                return
            self.ev.event("recovery_retry", condition=code, attempt=attempt)

        # Still blocked after the bound: bounded give-up (or escalate if available).
        self.ev.event("recovery_exhausted", condition=code,
                      attempts=self.max_recovery_attempts)
        if self.escalation:
            self._escalate(step, f"recovery for '{code}' exhausted", result, sr)
            return
        result.outcome = Outcome.RECOVERY_EXHAUSTED
        result.message = (f"recovery for '{code}' did not resolve after "
                          f"{self.max_recovery_attempts} attempts")
        sr.ok = False
        # Overwrite the stale action detail so the trace explains *why* ok flipped to
        # false, rather than leaving the misleading "clicked" from the action itself.
        sr.detail = (f"recovery for '{code}' exhausted after "
                     f"{self.max_recovery_attempts} verified-failed attempts")

    def _recovery_action(self, det) -> bool:
        """Perform the recovery for a known recoverable condition. Returns whether
        an action was attempted (unknown conditions are not silently 'recovered')."""
        if det.code == "interstitial_notice":
            self.surface.click(_CONTINUE_LOCATOR)
            return True
        return False

    def _escalate(self, step, reason, result, sr):
        req_id = f"intv-{uuid.uuid4().hex[:8]}"
        shot = self.ev.save_screenshot(self.surface, "escalation")
        from ..escalation.handoff import InterventionRequest
        art = getattr(self, "_artifact", None)
        req = InterventionRequest(
            request_id=req_id,
            capability=art.name if art else result.artifact_id,
            goal=art.goal if art else "",
            step_index=step.index, reason=reason,
            current_url=self.surface.current_url(), screenshot=shot,
            user_data_dir=self.surface.user_data_dir,
            cdp_endpoint=self.surface.cdp_endpoint(),
        )
        self.ev.event("escalation_raised", request_id=req_id, reason=reason,
                      step=step.index, screenshot=shot)
        handoff = self.escalation.escalate(req, self.surface, self.policy)
        self.ev.event("handoff_result", controller=handoff.controller,
                      resolved=handoff.resolved, actions=handoff.actions)
        result.evidence["escalation_request"] = req_id
        if handoff.resolved:
            # A reopened session starts blank; reposition to where the operator
            # left off. Session state (cookies/auth/ack) persists in the profile.
            self.surface.navigate(req.current_url)
            sr.recovered_from = "escalation"
            sr.detail = f"human/{handoff.controller} intervened: {handoff.actions}"
        else:
            result.outcome = Outcome.ESCALATED
            result.message = f"unresolved escalation: {handoff.note}"
            sr.ok = False

    # ------------------------------------------------------------- helpers #
    def _checkpoint_holds(self, cp, url, text) -> bool:
        """Verify a checkpoint, resolving ELEMENT_VISIBLE against the live surface."""
        from ..schema.artifact import CheckpointType
        if cp.type == CheckpointType.ELEMENT_VISIBLE and cp.locator is not None:
            pl, _ = self.surface._resolve(cp.locator)
            try:
                return pl is not None and pl.is_visible()
            except Exception:
                return False
        return verify_checkpoint(cp, url, text)

    # ------------------------------------------------------ tenant variants #
    def _active_variant(self, artifact, params):
        vid = params.get("_variant")
        if not vid:
            return None
        return next((v for v in artifact.variants if v.variant_id == vid), None)

    def _effective_route(self, artifact, step) -> str:
        """A variant selects its tenant `base_url` for the entry navigation. This is
        independent of `apply_variant_overrides`: the base_url picks *which tenant*
        we drive; the overrides are the locator/checkpoint specializations."""
        route = step.route or ""
        v = self._variant
        if v and v.base_url and route == artifact.target.base_url:
            return v.base_url
        return route

    def _effective_locator(self, artifact, step, params) -> Locator | None:
        if step.locator is None:
            return None
        v = self._variant
        if v and self.apply_variant_overrides and step.index in v.locator_overrides:
            return v.locator_overrides[step.index]
        return step.locator

    def _effective_checkpoint(self, artifact):
        v = self._variant
        if v and self.apply_variant_overrides and v.success_checkpoint is not None:
            return v.success_checkpoint
        return artifact.success_checkpoint

    def _resolve(self, text, params) -> str:
        def sub(m):
            key = m.group(1)
            if key not in params:
                raise MissingParam(f"missing parameter '{key}'")
            return str(params[key])
        return _TEMPLATE.sub(sub, text or "")

    def _with_retry(self, fn, attempts: int = 2, backoff: float = 0.4):
        last = None
        for i in range(attempts):
            last = fn()
            if getattr(last, "ok", False):
                return last
            time.sleep(backoff * (i + 1))
        return last

    def _check_params(self, artifact, params):
        for p in artifact.params:
            if p.required and p.name not in params:
                raise MissingParam(f"required parameter '{p.name}' not supplied")

    def _attach_failure_evidence(self, result):
        shot = self.ev.save_screenshot(self.surface, "failure")
        if shot:
            result.evidence["failure_screenshot"] = shot
        self.ev.finish("hard_failure", failed_step=result.failed_step,
                       expected=result.expected, observed=result.observed,
                       message=result.message, screenshot=shot)


# --------------------------------------------------------------- redaction #
def _loc_str(loc: Locator | None) -> str | None:
    """Format a locator the same way `_ok` tags an action's resolved rung.

    A READ resolves through the ordered candidate list and uses the first that
    resolves; read locators are single, label-anchored candidates, so this reports
    the anchor candidate in the identical `[rung/total:strategy] value` shape the
    click/type steps use, rather than a raw enum repr.
    """
    if not loc or not loc.candidates:
        return None
    c = loc.candidates[0]
    return f"[1/{len(loc.candidates)}:{c.strategy.value}] {c.value}"


def _safe_params(artifact: Artifact, params: dict) -> dict:
    sensitive = {p.name for p in artifact.params if p.sensitive}
    return {k: ("<redacted>" if k in sensitive else redact(str(v)))
            for k, v in params.items() if not k.startswith("_")}


def _redact_outputs(artifact: Artifact, outputs: dict) -> dict:
    return {k: redact(str(v)) for k, v in outputs.items()}
