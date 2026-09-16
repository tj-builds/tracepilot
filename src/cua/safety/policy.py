"""Safety guardrails: allowlist, action-risk gating, and redaction.

The policy is enforced *before every action*, in both discovery and replay. It is
the single choke point through which all navigation and interaction passes, so an
LLM (during discovery) cannot wander outside the sanctioned surface, and a replay
cannot perform an irreversible action that wasn't approved.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..schema.artifact import ActionType, RiskClass

# Patterns that look like secrets / regulated data -> redacted from logs & artifacts.
_REDACT_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<ssn>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<pan>"),          # card numbers
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[=:]\s*\S+"),
     r"\1=<redacted>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
]


def redact(text: str | None) -> str:
    if not text:
        return ""
    out = text
    for pat, repl in _REDACT_PATTERNS:
        out = pat.sub(repl, out)
    return out


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False


@dataclass
class Grant:
    """A one-shot authorization for a single irreversible action.

    Issued to a human operator during handoff so that even a person's irreversible
    click is policed by the same gate. The grant is *spent* by the action it
    authorizes, so it cannot be replayed for a second irreversible step.
    """

    token: str
    action: ActionType
    risk: RiskClass
    spent: bool = False


@dataclass
class Policy:
    """Configurable allowlist. Everything not explicitly allowed is denied."""

    allowed_hosts: set[str] = field(default_factory=set)
    allowed_route_patterns: list[re.Pattern] = field(default_factory=list)
    allowed_actions: set[ActionType] = field(
        default_factory=lambda: set(ActionType)
    )
    # Irreversible actions are gated: blocked, or require explicit confirmation.
    allow_irreversible: bool = False
    require_confirmation_for_irreversible: bool = True
    # One-shot authorization grants (used during human handoff).
    grants: list[Grant] = field(default_factory=list)

    @classmethod
    def from_allowlist(
        cls,
        base_url: str,
        route_patterns: list[str],
        allow_irreversible: bool = False,
    ) -> "Policy":
        host = urlparse(base_url).netloc
        compiled = [re.compile(_route_to_regex(r)) for r in route_patterns]
        return cls(
            allowed_hosts={host},
            allowed_route_patterns=compiled,
            allow_irreversible=allow_irreversible,
        )

    def check_navigation(self, url: str) -> PolicyDecision:
        parsed = urlparse(url)
        host = parsed.netloc
        if host and self.allowed_hosts and host not in self.allowed_hosts:
            return PolicyDecision(False, f"host '{host}' not in allowlist")
        path = parsed.path or "/"
        if self.allowed_route_patterns and not any(
            p.match(path) for p in self.allowed_route_patterns
        ):
            return PolicyDecision(False, f"route '{path}' not in allowlist")
        return PolicyDecision(True, "navigation permitted")

    def issue_grant(self, action: ActionType, risk: RiskClass) -> str:
        """Mint a one-shot authorization for a single irreversible action."""
        g = Grant(token=uuid.uuid4().hex[:12], action=action, risk=risk)
        self.grants.append(g)
        return g.token

    def check_action(self, action: ActionType, risk: RiskClass,
                     grant: str | None = None) -> PolicyDecision:
        if action not in self.allowed_actions:
            return PolicyDecision(False, f"action '{action}' not permitted")
        if risk == RiskClass.IRREVERSIBLE:
            # A one-shot grant authorizes exactly one irreversible action and is then
            # spent. This is what lets a *human* act during handoff without the gate
            # simply rubber-stamping anything they click.
            if grant is not None:
                g = next((g for g in self.grants
                          if g.token == grant and not g.spent
                          and g.action == action and g.risk == risk), None)
                if g is None:
                    return PolicyDecision(
                        False, "invalid, mismatched, or already-spent grant")
                g.spent = True
                return PolicyDecision(True, "authorized by one-shot grant")
            if not self.allow_irreversible:
                return PolicyDecision(
                    False, "irreversible action blocked by policy"
                )
            if self.require_confirmation_for_irreversible:
                return PolicyDecision(
                    True, "irreversible action requires confirmation",
                    requires_confirmation=True,
                )
        return PolicyDecision(True, "action permitted")


def _route_to_regex(pattern: str) -> str:
    """Convert a canonical route ('/member/:id') to an anchored regex."""
    parts = []
    for seg in pattern.split("/"):
        if seg.startswith(":"):
            parts.append(r"[^/]+")
        else:
            parts.append(re.escape(seg))
    return "^" + "/".join(parts) + "/?$"
