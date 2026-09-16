"""Application service layer.

The reusable operations the CLI and the agentic router both call, kept out of the
CLI so they are not coupled to argument parsing:

  * discover_capability(...) -- run an LLM-driven discovery and save the artifact.
  * invoke_capability(...)   -- deterministically replay a saved capability by name.
  * run_request(...)         -- the agentic entrypoint: route a natural-language
                                request to invoke-an-existing or create-then-invoke.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent.llm import build_client
from .agent.loop import DiscoveryAgent, DiscoveryResult
from .agent.planner import build_planner
from .agent.router import CapabilityRouter, RouteDecision
from .agent.validator import build_validator
from .catalog.registry import CapabilityRegistry
from .config import CONFIG
from .escalation.handoff import (
    ESCALATION_DIR,
    AutoOperatorHandler,
    raise_authorization_request,
)
from .evidence.logger import EvidenceLog
from .replay.engine import ReplayEngine
from .safety.operations import OperationDecision, classify_operation
from .safety.policy import Policy
from .schema.artifact import Artifact, Checkpoint, CheckpointType, TargetSpec
from .schema.results import Outcome, ReplayResult
from .surface.web import WebSurface

APP_SIGNATURE = "corecu-servicing"
ROUTE_PATTERNS = [
    "/", "/login", "/search", "/member",
    "/member/:member_id",
    "/member/:member_id/ack",
    "/member/:member_id/sub-account/new",
    "/member/:member_id/sub-account/confirm",
]


def creds() -> dict:
    return {"username": CONFIG.app_username, "password": CONFIG.app_password}


def target_spec(base_url: str | None = None) -> TargetSpec:
    return TargetSpec(app_signature=APP_SIGNATURE,
                      base_url=base_url or CONFIG.target_base_url,
                      allowed_routes=ROUTE_PATTERNS)


# --------------------------------------------------------------------------- #
# Input completeness (never fabricate a required input)                       #
# --------------------------------------------------------------------------- #
def caller_required_inputs(art: Artifact) -> list[str]:
    """Required params the CALLER must supply for an invocation.

    Sensitive params (credentials) are excluded: they come from secure config, not
    from the request. Everything else that is `required` must be provided explicitly
    -- we never fall back to the recorded `example` or a hard-coded default, because
    guessing the *subject* of an action (which member, how much) is a safety hole,
    not a convenience.
    """
    return [p.name for p in art.params if p.required and not p.sensitive]


def missing_inputs(art: Artifact, args: dict | None) -> list[str]:
    """Required caller inputs that were not supplied with a non-empty value."""
    supplied = {k for k, v in (args or {}).items()
                if v is not None and str(v).strip() != ""}
    return [name for name in caller_required_inputs(art) if name not in supplied]


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #
def discover_capability(goal: str, capability: str, description: str,
                        checkpoint: Checkpoint, outputs_spec: list[dict], *,
                        member_id: str, extra_params: dict | None = None,
                        no_validator: bool = False, base_url: str | None = None,
                        run_id: str | None = None) -> tuple[DiscoveryResult, EvidenceLog]:
    CONFIG.ensure_dirs()
    ev = EvidenceLog("discovery", run_id=run_id)
    # The target (app/entry point) is an explicit input to the agent loop, defaulting
    # to the configured app when not supplied.
    base = base_url or CONFIG.target_base_url
    policy = Policy.from_allowlist(base, ROUTE_PATTERNS,
                                   allow_irreversible=True)  # discovery is supervised
    params = {"member_id": member_id, **creds()}
    if extra_params:
        params.update(extra_params)

    planner = build_planner(CONFIG)
    validator = None if no_validator else build_validator(CONFIG, policy)
    surface = WebSurface(headless=CONFIG.headless, timeout_ms=CONFIG.step_timeout_ms)
    surface.start()
    try:
        agent = DiscoveryAgent(surface, planner, policy, ev, CONFIG, validator=validator)
        res = agent.run(goal, target_spec(base), params, capability, description,
                        checkpoint, outputs_spec=outputs_spec)
    finally:
        surface.stop()

    if res.status == "success" and res.artifact:
        CapabilityRegistry().save(res.artifact)
    return res, ev


# --------------------------------------------------------------------------- #
# Replay / invoke                                                             #
# --------------------------------------------------------------------------- #
def invoke_capability(name: str, args: dict | None = None, *,
                      allow_irreversible: bool = False, escalate: bool = False,
                      tenant: str | None = None, inject: str | None = None,
                      run_id: str | None = None,
                      strict_inputs: bool = True) -> tuple[ReplayResult, EvidenceLog]:
    CONFIG.ensure_dirs()
    reg = CapabilityRegistry()
    art = reg.load(name)
    ev = EvidenceLog("replay", run_id=run_id)
    args = dict(args or {})

    # Completeness gate (before any browser work): if the caller omitted a required
    # input we refuse to invent it. We return NEEDS_INPUT and surface exactly what is
    # missing, so the request can be clarified by the user or routed to a human --
    # nothing is executed against the live surface.
    if strict_inputs:
        gaps = missing_inputs(art, args)
        if gaps:
            result = ReplayResult(outcome=Outcome.NEEDS_INPUT, artifact_id=art.id,
                                  artifact_version=art.version, missing_inputs=gaps,
                                  message="incomplete request; missing required "
                                          f"input(s): {', '.join(gaps)}")
            ev.event("needs_input", capability=art.name, missing=gaps)
            ev.finish("needs_input", missing=gaps)
            ev.write_json("replay_result.json", result.model_dump())
            return result, ev

    policy = Policy.from_allowlist(CONFIG.target_base_url, art.target.allowed_routes,
                                   allow_irreversible=allow_irreversible)

    # Assemble params WITHOUT fabricating required inputs. Recorded `example` values
    # may seed OPTIONAL params only (a safe convenience); required inputs come solely
    # from the caller's args. Credentials are always supplied fresh from config.
    params: dict[str, Any] = {p.name: p.example for p in art.params
                              if p.example is not None and not p.sensitive
                              and not p.required}
    if not strict_inputs:
        # Legacy/opt-out path: allow required params to fall back to examples too.
        for p in art.params:
            if p.example is not None and not p.sensitive and p.name not in params:
                params[p.name] = p.example
    for k, v in args.items():
        if v is not None:
            params[k] = str(v)
    params.update(creds())
    if tenant:
        params["_variant"] = tenant

    escalation, escalate_codes = None, set()
    cdp_port = None
    if escalate:
        escalation = AutoOperatorHandler(CONFIG.target_base_url,
                                         headless=CONFIG.headless, **creds())
        escalate_codes = {"interstitial_notice", "session_timeout"}
        # Expose a CDP port so a human could attach to the same live session.
        cdp_port = CONFIG.handoff_cdp_port

    surface = WebSurface(headless=CONFIG.headless, timeout_ms=CONFIG.step_timeout_ms,
                         cdp_port=cdp_port)
    surface.start()
    try:
        if inject:
            surface.navigate(f"{CONFIG.target_base_url}/_simulate/inject/{inject}")
            ev.event("fault_injected", condition=inject)
        engine = ReplayEngine(surface, policy, ev, CONFIG, escalation=escalation,
                              escalate_codes=escalate_codes)
        result = engine.replay(art, params)
    finally:
        surface.stop()

    ev.write_json("replay_result.json", result.model_dump())
    return result, ev


# --------------------------------------------------------------------------- #
# Agentic entrypoint                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class RunOutcome:
    decision: RouteDecision
    # invoke | created_and_invoked | would_create | create_failed | needs_input
    phase: str
    result: ReplayResult | None = None
    discovery: DiscoveryResult | None = None
    evidence: str = ""
    error: str = ""
    missing_inputs: list[str] = field(default_factory=list)


def run_request(request: str, *, autocreate: bool = True,
                allow_irreversible: bool = False, escalate: bool = False,
                run_id: str | None = None) -> RunOutcome:
    """Route a natural-language request, then execute it.

    - If an existing capability semantically fits -> invoke (deterministic replay).
    - If none fits -> (optionally) discover a new capability from the request, then
      invoke it. This is the "reuse or create the workflow" decision, made by meaning.
    """
    if not CONFIG.has_llm:
        return RunOutcome(RouteDecision(mode="create"), "error",
                          error="OPENAI_API_KEY is not set; routing requires a real "
                                "model (there is no offline fallback).")

    reg = CapabilityRegistry()
    router = CapabilityRouter(build_client(CONFIG), reg)
    decision = router.route(request)

    # Operation-level allowlist (BEFORE any execution or discovery): may we act
    # autonomously, or must this go to a human? Human-only intents (close/transfer/
    # delete/...) and any operation not on the allowlist -- including learning a new
    # capability -- are routed to a human in the loop.
    op = decision.capability if decision.mode == "invoke" else decision.proposed_name
    verdict = classify_operation(op, request)
    if verdict.decision == OperationDecision.ESCALATE:
        req_id = raise_authorization_request(
            capability=op or "(new capability)",
            goal=decision.proposed_goal or request, reason=verdict.reason)
        return RunOutcome(decision, "escalated",
                          error=f"routed to a human operator: {verdict.reason}",
                          evidence=str(ESCALATION_DIR / f"{req_id}.json"))

    # INVOKE an existing capability (guard against a hallucinated name).
    known = {a.name for a in reg.list()}
    if decision.mode == "invoke" and decision.capability in known:
        result, ev = invoke_capability(decision.capability, decision.args,
                                       allow_irreversible=allow_irreversible,
                                       escalate=escalate, run_id=run_id)
        # An incomplete request is not an execution failure -- it's a clarification.
        phase = "needs_input" if result.outcome == Outcome.NEEDS_INPUT else "invoke"
        return RunOutcome(decision, phase, result=result, evidence=str(ev.dir),
                          missing_inputs=result.missing_inputs)

    # CREATE then invoke.
    if not autocreate:
        return RunOutcome(decision, "would_create")

    # Before spending an LLM discovery run, refuse to proceed on an incomplete
    # request: every capability here operates on a specific member, so a missing
    # member id means we'd be acting on an unspecified (or defaulted) subject. Ask
    # instead of guessing -- this is the "wrong person" safety hole made explicit.
    if not str(decision.args.get("member_id") or "").strip():
        return RunOutcome(decision, "needs_input", missing_inputs=["member_id"],
                          error="request does not specify the member to act on; "
                                "add the member id (e.g. 'for member 100001') "
                                "and retry.")
    if not decision.success_text:
        return RunOutcome(decision, "create_failed",
                          error="router did not specify a success condition to verify")

    checkpoint = Checkpoint(type=CheckpointType.TEXT_PRESENT,
                            expression=decision.success_text,
                            description="router-proposed success condition")
    disc, dev = discover_capability(
        goal=decision.proposed_goal or request,
        capability=decision.proposed_name or "new_capability",
        description=decision.proposed_description or request,
        checkpoint=checkpoint, outputs_spec=decision.outputs,
        member_id=str(decision.args["member_id"]))  # guaranteed present by the guard
    if disc.status != "success":
        return RunOutcome(decision, "create_failed", discovery=disc,
                          evidence=disc.evidence_dir,
                          error=f"discovery did not complete: {disc.reason}")

    result, ev = invoke_capability(decision.proposed_name, decision.args,
                                   allow_irreversible=allow_irreversible,
                                   escalate=escalate)
    return RunOutcome(decision, "created_and_invoked", result=result,
                      discovery=disc, evidence=str(ev.dir))
