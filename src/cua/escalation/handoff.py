"""Human-in-the-loop escalation and same-session control transfer.

Control-transfer model
----------------------
There is exactly one live session, identified by the browser's persistent profile
directory (`user_data_dir`). At most one party holds it at a time:

    AUTOMATION --(pause: release profile)--> HUMAN --(resume: release profile)--> AUTOMATION

An `InterventionRequest` carries enough context to act (capability, step, current
URL, screenshot, reason). The automation *releases* the session, a human operates
the *same* profile (same cookies/auth/state), then hands it back and the run
resumes. We record who held control and what they did.

Two handlers implement the same interface:
  * FileEscalationHandler -- writes the request to a queue dir and blocks until a
    human signals resume (real operator, possibly via the mock operator console).
  * AutoOperatorHandler   -- a scripted stand-in operator that performs the manual
    fix on the same profile so the whole loop is demonstrable without a person.
"""
from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT
from ..safety.policy import Policy, PolicyDecision
from ..schema.artifact import ActionType, RiskClass
from ..surface.web import WebSurface

ESCALATION_DIR = REPO_ROOT / "evidence" / "_escalations"


@dataclass
class OperatorAction:
    """One action a human operator took during handoff, as policed + recorded."""

    action: str
    target: str
    risk: str
    authorized: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "authorized" if self.authorized else f"REFUSED ({self.detail})"
        return f"{self.action} '{self.target}' [{self.risk}] {status}"


class HandoffControl:
    """Mediates a human operator's actions on the *same live session* through the
    policy gate. This is the point of the control-transfer model: handing the session
    to a person does not suspend the guardrails. Every action is policy-checked and
    recorded; an IRREVERSIBLE action requires a one-shot authorization grant that the
    policy consumes, so a human cannot silently perform an unbounded irreversible
    action. Typed values are never recorded (regulated-data redaction).
    """

    def __init__(self, page, policy: Policy | None):
        self.page = page
        self.policy = policy
        self.actions: list[OperatorAction] = []

    def _check(self, action: ActionType, risk: RiskClass,
               grant: str | None) -> PolicyDecision:
        if self.policy is None:
            return PolicyDecision(True, "no policy bound")
        return self.policy.check_action(action, risk, grant=grant)

    def click(self, name: str, role: str = "button",
              risk: RiskClass = RiskClass.SAFE, grant: str | None = None) -> bool:
        dec = self._check(ActionType.CLICK, risk, grant)
        if not dec.allowed:
            self.actions.append(OperatorAction("click", name, risk.value, False, dec.reason))
            return False
        self.page.get_by_role(role, name=name).click()
        self.actions.append(OperatorAction("click", name, risk.value, True, dec.reason))
        return True

    def fill(self, label: str, value: str,
             risk: RiskClass = RiskClass.REVERSIBLE) -> bool:
        dec = self._check(ActionType.TYPE, risk, None)
        if not dec.allowed:
            self.actions.append(OperatorAction("type", label, risk.value, False, dec.reason))
            return False
        self.page.get_by_label(label).fill(value)
        # NB: value intentionally omitted from the record (may be a credential/PII).
        self.actions.append(OperatorAction("type", label, risk.value, True))
        return True

    def summary(self) -> list[str]:
        return [str(a) for a in self.actions]

    def all_authorized(self) -> bool:
        return bool(self.actions) and all(a.authorized for a in self.actions)


@dataclass
class InterventionRequest:
    request_id: str
    capability: str
    goal: str
    step_index: int
    reason: str
    current_url: str
    screenshot: str | None = None
    user_data_dir: str | None = None
    # The live CDP endpoint an operator attaches to, to drive the SAME session.
    cdp_endpoint: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        # Never persist the absolute profile path -- it embeds the OS username. The
        # basename is enough to correlate the session in evidence.
        if d.get("user_data_dir"):
            d["user_data_dir"] = Path(d["user_data_dir"]).name
        return d


@dataclass
class HandoffResult:
    resolved: bool
    controller: str                 # "human" | "auto-operator"
    actions: list[str] = field(default_factory=list)
    note: str = ""


def raise_authorization_request(capability: str, goal: str, reason: str) -> str:
    """Pre-execution escalation: queue a human-authorization request for an operation
    the system may NOT perform autonomously (a human-only intent, or an unknown/new
    operation). No live session exists yet -- nothing has been executed -- so this asks
    a human to take/authorize the operation rather than handing off a live session.
    Returns the request id (written to the same queue the operator console reads)."""
    ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
    req_id = f"intv-{uuid.uuid4().hex[:8]}"
    req = InterventionRequest(request_id=req_id, capability=capability, goal=goal,
                              step_index=-1, reason=reason, current_url="")
    (ESCALATION_DIR / f"{req_id}.json").write_text(
        json.dumps(req.to_dict(), indent=2), encoding="utf-8")
    return req_id


class EscalationHandler(ABC):
    @abstractmethod
    def escalate(self, req: InterventionRequest, surface: WebSurface,
                 policy: Policy | None = None) -> HandoffResult: ...


class FileEscalationHandler(EscalationHandler):
    """Route to a real human: write the request, wait for a resume signal.

    In a production co-browse integration the human's actions would flow through the
    same `HandoffControl` gate; here the resume signal may carry the actions (and, for
    an irreversible step, a one-shot grant token) the operator performed.
    """

    def __init__(self, poll_seconds: float = 2.0, timeout_seconds: float = 600):
        self.poll = poll_seconds
        self.timeout = timeout_seconds
        ESCALATION_DIR.mkdir(parents=True, exist_ok=True)

    def escalate(self, req, surface, policy=None) -> HandoffResult:
        req_path = ESCALATION_DIR / f"{req.request_id}.json"
        resume_path = ESCALATION_DIR / f"{req.request_id}.resume"
        req_path.write_text(json.dumps(req.to_dict(), indent=2), encoding="utf-8")

        # Release the live session so a human can take control of the same profile.
        surface.to_headed_for_handoff()

        waited = 0.0
        while waited < self.timeout:
            if resume_path.exists():
                actions = json.loads(resume_path.read_text() or "{}").get("actions", [])
                surface.resume_after_handoff()
                return HandoffResult(True, "human", actions, "resumed by operator")
            time.sleep(self.poll)
            waited += self.poll
        return HandoffResult(False, "human", note="timed out waiting for operator")


class AutoOperatorHandler(EscalationHandler):
    """A scripted stand-in operator so the full handoff loop runs unattended.

    It takes control of the *same* persistent profile, performs a minimal fix, then
    releases it. Currently it can: re-authenticate after a session timeout, and
    dismiss an interstitial 'Continue' notice.
    """

    def __init__(self, base_url: str, username: str | None = None,
                 password: str | None = None, headless: bool = True):
        from ..config import CONFIG
        self.base_url = base_url
        self.username = username or CONFIG.app_username
        self.password = password or CONFIG.app_password
        self.headless = headless
        ESCALATION_DIR.mkdir(parents=True, exist_ok=True)

    def escalate(self, req, surface, policy=None) -> HandoffResult:
        (ESCALATION_DIR / f"{req.request_id}.json").write_text(
            json.dumps(req.to_dict(), indent=2), encoding="utf-8")

        surface.to_headed_for_handoff()  # automation yields control
        page = surface.control_page()    # the SAME live page/session
        ctl = HandoffControl(page, policy)  # every operator action is policed
        body = (page.inner_text("body") or "").lower()

        fixed = False
        if "sign in" in body and page.get_by_label("Username").count():
            ctl.fill("Username", self.username)
            ctl.fill("Password", self.password)
            fixed = ctl.click("Sign in", risk=RiskClass.SAFE)
        elif page.get_by_role("button", name="Continue").count():
            fixed = ctl.click("Continue", risk=RiskClass.SAFE)

        surface.resume_after_handoff()  # automation takes control back
        # Resolved only if a fix was applied AND every action passed the policy gate.
        # NOTE: resume decision is "advance" (the escalated step already ran; the
        # engine continues from the NEXT step), so a human confirm cannot double-fire.
        resolved = fixed and ctl.all_authorized()
        note = "same-session handoff completed" if resolved else "no policed fix applied"
        return HandoffResult(resolved, "auto-operator", ctl.summary(), note)
