"""Runtime error / exceptional-state taxonomy for deterministic replay.

The core idea from the brief: since the UI is stable, the interesting failures are
*runtime conditions*, not layout drift. We classify what we observe into four buckets
and respond deliberately:

  BUSINESS         -> a legitimate result the caller needs (member_not_found,
                      permission_denied, validation_error). NOT a crash.
  RECOVERABLE      -> handle and continue (dismiss interstitial, retry slow load).
  SESSION_TIMEOUT  -> a special recoverable: re-authenticate if creds available.
  HARD             -> stop and surface a debuggable error.

Rules are data, not code, so per-tenant/vendor overrides can extend them without
touching the engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..schema.artifact import Checkpoint, CheckpointType


class Condition(str, Enum):
    NONE = "none"
    BUSINESS = "business"
    RECOVERABLE = "recoverable"
    SESSION_TIMEOUT = "session_timeout"
    HARD = "hard"


@dataclass
class Detection:
    condition: Condition
    code: str = ""
    message: str = ""


@dataclass
class Rule:
    pattern: re.Pattern
    condition: Condition
    code: str
    message: str


# Default rule set. `text` is the visible page text; `url` the current route.
DEFAULT_RULES: list[Rule] = [
    Rule(re.compile(r"no such member", re.I), Condition.BUSINESS,
         "member_not_found", "The member ID was not found."),
    Rule(re.compile(r"permission denied", re.I), Condition.BUSINESS,
         "permission_denied", "The operator is not authorized for this member."),
    Rule(re.compile(r"is required", re.I), Condition.BUSINESS,
         "validation_error", "A required field was rejected by the app."),
    Rule(re.compile(r"pending compliance review|^\s*notice\b", re.I),
         Condition.RECOVERABLE, "interstitial_notice",
         "An interstitial notice must be acknowledged."),
    Rule(re.compile(r"internal error|unexpected error|system error", re.I),
         Condition.HARD, "app_error",
         "The application returned an internal error."),
]

_LOGIN_HINT = re.compile(r"\bsign in\b", re.I)


def classify_page(url: str, text: str, rules: list[Rule] | None = None,
                  expect_login: bool = False) -> Detection:
    """Classify the current page into a runtime condition.

    Ordering matters. The login screen is the discriminator for an important
    ambiguity: the *same* "we can't proceed" state means a session timeout when a
    login form is present, but a hard app error when it is not. So we check the
    login-form / session-timeout case FIRST; only then do we scan the rule set
    (where the HARD `app_error` rule lives). This is why an "internal error" page
    is a hard failure unless we were bounced to login, in which case it's a timeout.
    """
    rules = rules or DEFAULT_RULES
    # Session timeout: bounced back to the login screen when we didn't expect it.
    if not expect_login and (url.rstrip("/").endswith("/login")
                             or (_LOGIN_HINT.search(text or "")
                                 and "member" not in (text or "").lower())):
        return Detection(Condition.SESSION_TIMEOUT, "session_timeout",
                         "Session expired; bounced to the login screen.")
    for r in rules:
        if r.pattern.search(text or ""):
            return Detection(r.condition, r.code, r.message)
    return Detection(Condition.NONE)


def verify_checkpoint(cp: Checkpoint, url: str, text: str) -> bool:
    if cp.type == CheckpointType.URL_MATCHES:
        return re.search(cp.expression, url or "") is not None
    if cp.type == CheckpointType.TEXT_PRESENT:
        return cp.expression.lower() in (text or "").lower()
    if cp.type == CheckpointType.TEXT_ABSENT:
        return cp.expression.lower() not in (text or "").lower()
    if cp.type == CheckpointType.ELEMENT_VISIBLE:
        # Element visibility can only be confirmed against a live surface via locator
        # resolution, which this text/url-level helper does not have. The engine must
        # verify this checkpoint type through the surface; failing closed here avoids
        # a false "success" when the check was never actually performed.
        return False
    return False
