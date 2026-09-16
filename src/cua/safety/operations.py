"""Operation-level allowlist -- the *semantic* safety layer.

Below this there are two other guardrails: the route/host allowlist (`policy.py`) and
the per-action risk gate (safe / reversible / irreversible). Those decide *where* the
agent may go and *whether a single UI action* is permitted. This layer decides
something higher-level and prior to execution: **is the system even allowed to attempt
this whole operation autonomously, or must a human take it?**

The model is an allowlist, not a blocklist:

  * ALLOW    -- the operation is explicitly trusted for autonomous execution.
  * ESCALATE -- the operation must be handled by a human in the loop. This covers two
                cases: (a) a high-blast-radius / irreversible *intent* (close, transfer,
                wire, delete, freeze ...) that we never do unattended even if a
                capability for it exists, and (b) any operation NOT on the allowlist
                (including learning a brand-new capability) -- unknown means human.

Everything is data here so a tenant/institution can widen or narrow what its agents may
do without touching the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationDecision(str, Enum):
    ALLOW = "allow"        # autonomous execution permitted
    ESCALATE = "escalate"  # must be handled by a human in the loop


# Operations the system is trusted to perform autonomously. These are capability
# names (the router resolves a request to one of these). Anything not listed here is
# escalated to a human by default.
ALLOWED_OPERATIONS: set[str] = {
    "lookup_member_balance",   # read-only: fetch a member's savings balance
    "open_subaccount",         # state-changing but bounded; gated again at replay
}

# Intents that ALWAYS require a human, even if a capability exists -- money movement
# and account-lifecycle changes are too high-blast-radius to run unattended. Matched as
# substrings against the request text and the routed operation name.
HUMAN_ONLY_INTENTS: dict[str, str] = {
    "close": "closing an account",
    "delete": "deleting a record",
    "remove": "removing a record",
    "transfer": "transferring funds",
    "wire": "wiring funds",
    "withdraw": "withdrawing funds",
    "payout": "issuing a payout",
    "freeze": "freezing an account",
    "reopen": "reopening a closed account",
    "charge off": "charging off a balance",
    "chargeoff": "charging off a balance",
    "write off": "writing off a balance",
}


@dataclass
class OperationVerdict:
    decision: OperationDecision
    reason: str = ""
    matched: str = ""  # the operation name or the human-only keyword that matched


def classify_operation(operation: str | None, request_text: str = "") -> OperationVerdict:
    """Decide whether an operation may run autonomously or must go to a human.

    `operation` is the routed capability name (or a proposed new-capability name);
    `request_text` is the original natural-language request. We check the human-only
    intents first (they win regardless of the allowlist), then the allowlist.
    """
    hay = f"{operation or ''} {request_text}".lower()
    for kw, desc in HUMAN_ONLY_INTENTS.items():
        if kw in hay:
            return OperationVerdict(OperationDecision.ESCALATE,
                                    f"'{desc}' is a human-only operation", matched=kw)
    if operation and operation in ALLOWED_OPERATIONS:
        return OperationVerdict(OperationDecision.ALLOW,
                                "operation is on the autonomous allowlist",
                                matched=operation)
    return OperationVerdict(OperationDecision.ESCALATE,
                            "operation is not on the autonomous allowlist "
                            "(unknown/new operations require human authorization)",
                            matched=operation or "")
