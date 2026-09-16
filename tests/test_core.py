"""Unit tests for the load-bearing, browser-free pieces:
schema round-trip, safety policy, redaction, error taxonomy, locator synthesis,
and template resolution.
"""
from __future__ import annotations

import re

import pytest

from cua.agent.recorder import Recorder, read_locator_for_label, synthesize_locator
from cua.replay.errors import Condition, classify_page, verify_checkpoint
from cua.safety.policy import Policy, redact
from cua.schema.artifact import (
    ActionType,
    Artifact,
    Checkpoint,
    CheckpointType,
    LocatorStrategy,
    RiskClass,
    TargetSpec,
)
from cua.surface.base import ElementDigest


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #
def _artifact() -> Artifact:
    r = Recorder(
        goal="g", capability_name="cap", description="d",
        target=TargetSpec(app_signature="sig", base_url="http://h",
                          allowed_routes=["/member/:member_id"]),
        provided_params={"member_id": "100001", "username": "u", "password": "p"},
    )
    r.record_navigate("http://h")
    r.record_type(ElementDigest(ref="e0", role="textbox", name="Member ID",
                                editable=True, attrs={"id": "mid"}), "{{member_id}}")
    r.record_read("savings_balance", "Savings balance")
    return r.finalize(Checkpoint(type=CheckpointType.TEXT_PRESENT,
                                 expression="Savings balance"))


def test_artifact_roundtrip():
    art = _artifact()
    js = art.to_json()
    back = Artifact.model_validate_json(js)
    assert back.name == "cap"
    assert back.steps[1].value == "{{member_id}}"
    assert back.outputs[0].name == "savings_balance"


def test_credentials_marked_sensitive_and_no_example():
    # Record a login flow so username/password become real params.
    r = Recorder(
        goal="g", capability_name="cap", description="d",
        target=TargetSpec(app_signature="sig", base_url="http://h",
                          allowed_routes=["/login"]),
        provided_params={"member_id": "100001", "username": "u", "password": "p"},
    )
    r.record_type(ElementDigest(ref="e0", role="textbox", name="Username",
                                editable=True, attrs={}), "{{username}}")
    r.record_type(ElementDigest(ref="e1", role="textbox", name="Password",
                                editable=True, attrs={}), "{{password}}")
    r.record_type(ElementDigest(ref="e2", role="textbox", name="Member ID",
                                editable=True, attrs={"id": "mid"}), "{{member_id}}")
    art = r.finalize(Checkpoint(type=CheckpointType.TEXT_PRESENT, expression="x"))
    names = {p.name: p for p in art.params}
    # Credentials are sensitive and carry NO example (never persisted).
    assert names["username"].sensitive is True
    assert names["username"].example is None
    assert names["password"].sensitive is True
    assert names["password"].example is None
    # Non-sensitive params keep an example for the calling agent.
    assert names["member_id"].sensitive is False
    assert names["member_id"].example == "100001"


def test_artifact_contract_validation():
    art = _artifact()
    assert art.validate_contract() == []  # a well-formed artifact passes

    # A READ step that names an undeclared output must be flagged.
    bad = _artifact()
    bad.steps[-1].output_name = "not_declared"
    issues = bad.validate_contract()
    assert any("READ does not name a declared output" in i for i in issues)

    # A templated value referencing an undeclared parameter must be flagged.
    bad2 = _artifact()
    bad2.steps[1].value = "{{ghost}}"
    assert any("undeclared parameter 'ghost'" in i for i in bad2.validate_contract())


# --------------------------------------------------------------------------- #
# Locator synthesis                                                           #
# --------------------------------------------------------------------------- #
def test_locator_candidates_ordered_by_robustness():
    loc = synthesize_locator(ElementDigest(
        ref="e0", role="textbox", name="Member ID", editable=True,
        attrs={"id": "mid"}))
    strategies = [c.strategy for c in loc.candidates]
    assert strategies[0] == LocatorStrategy.ROLE            # most robust first
    assert LocatorStrategy.CSS in strategies                # id fallback present
    assert loc.candidates[0].confidence > loc.candidates[-1].confidence


def test_read_locator_is_label_anchored():
    loc = read_locator_for_label("Savings balance")
    assert loc.candidates[0].strategy == LocatorStrategy.XPATH
    assert "Savings balance" in loc.candidates[0].value


# --------------------------------------------------------------------------- #
# Safety policy + redaction                                                   #
# --------------------------------------------------------------------------- #
def test_allowlist_blocks_foreign_host_and_route():
    pol = Policy.from_allowlist("http://good.test", ["/member/:id"])
    assert pol.check_navigation("http://good.test/member/1").allowed
    assert not pol.check_navigation("http://evil.test/member/1").allowed
    assert not pol.check_navigation("http://good.test/admin").allowed


def test_irreversible_gated_by_default():
    pol = Policy.from_allowlist("http://h", ["/"])
    d = pol.check_action(ActionType.CLICK, RiskClass.IRREVERSIBLE)
    assert not d.allowed
    pol2 = Policy.from_allowlist("http://h", ["/"], allow_irreversible=True)
    d2 = pol2.check_action(ActionType.CLICK, RiskClass.IRREVERSIBLE)
    assert d2.allowed and d2.requires_confirmation


def test_redaction():
    assert "<ssn>" in redact("ssn 123-45-6789")
    assert "password=<redacted>" in redact("password=hunter2")
    assert "<email>" in redact("user@example.com")


# --------------------------------------------------------------------------- #
# Error taxonomy                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected_code", [
    ("No such member. Member ID 000000 was not found.", "member_not_found"),
    ("Permission denied. You are not authorized.", "permission_denied"),
    ("Account type is required.", "validation_error"),
    ("Notice pending compliance review. Continue anyway?", "interstitial_notice"),
])
def test_classify_business_and_recoverable(text, expected_code):
    det = classify_page("http://h/member/1", text)
    assert det.code == expected_code


def test_classify_session_timeout_only_after_login():
    login_text = "Sign in"
    # Expected during the login phase -> not a timeout.
    assert classify_page("http://h/login", login_text,
                         expect_login=True).condition == Condition.NONE
    # Unexpected later -> timeout.
    assert classify_page("http://h/login", login_text,
                         expect_login=False).condition == Condition.SESSION_TIMEOUT


def test_verify_checkpoint():
    cp = Checkpoint(type=CheckpointType.TEXT_PRESENT, expression="Savings balance")
    assert verify_checkpoint(cp, "http://h/member/1", "... Savings balance ...")
    assert not verify_checkpoint(cp, "http://h/member/1", "No such member")
    cp2 = Checkpoint(type=CheckpointType.URL_MATCHES, expression=r"/member/\d+")
    assert verify_checkpoint(cp2, "http://h/member/100001", "")


def test_element_visible_checkpoint_fails_closed_without_surface():
    # ELEMENT_VISIBLE cannot be judged from url/text alone -> must not report success.
    cp = Checkpoint(type=CheckpointType.ELEMENT_VISIBLE, expression="ignored")
    assert verify_checkpoint(cp, "http://h/member/1", "anything") is False


# --------------------------------------------------------------------------- #
# LLM stand-in (planner) + validator (critic)                                 #
# --------------------------------------------------------------------------- #
from cua.agent.decision import Decision
from cua.agent.validator import ActionValidator
from cua.surface.base import Observation
from fakes import FakeLLMClient


def _obs(url, elements, text=""):
    return Observation(url=url, title="t", elements=elements, text=text)


def _login_obs():
    return _obs("http://h/login", [
        ElementDigest(ref="e0", role="textbox", name="Username", editable=True),
        ElementDigest(ref="e1", role="textbox", name="Password", editable=True),
        ElementDigest(ref="e2", role="button", name="Sign in"),
    ], text="Sign in to the console")


def test_fake_planner_fills_matching_field_then_advances():
    client = FakeLLMClient()
    params = {"username": "op", "password": "pw", "member_id": "100001"}
    obs = _login_obs()
    d = Decision(**_decode(client.complete("s", "u", mode="plan", goal="look up member",
                                           params=params, observation=obs, history=[],
                                           objectives={})))
    assert d.action == "type" and d.ref == "e0" and d.text == "{{username}}"

    # With all inputs filled, the planner advances via the goal-matching control.
    obs.elements[0].value = "op"
    obs.elements[1].value = "pw"
    d2 = Decision(**_decode(client.complete("s", "u", mode="plan", goal="sign in",
                                            params=params, observation=obs, history=[],
                                            objectives={})))
    assert d2.action == "click" and d2.ref == "e2"


def test_fake_planner_never_invents_unmatched_values():
    client = FakeLLMClient()
    obs = _obs("http://h/x", [
        ElementDigest(ref="e0", role="textbox", name="Nickname", editable=True),
        ElementDigest(ref="e1", role="button", name="Continue"),
    ], text="page")
    d = Decision(**_decode(client.complete("s", "u", mode="plan", goal="proceed",
                                           params={"member_id": "1"}, observation=obs,
                                           history=[], objectives={})))
    # No param matches 'Nickname', so it must not type into it.
    assert d.action != "type"


def test_validator_rejects_irreversible_for_readonly_goal():
    client = FakeLLMClient()
    pol = Policy.from_allowlist("http://h", ["/"], allow_irreversible=True)
    v = ActionValidator(client, pol)
    obs = _obs("http://h/", [ElementDigest(ref="e0", role="button", name="Create")],
               text="x")
    dec = Decision(action="click", ref="e0", risk="irreversible", reason="create")
    verdict = v.validate("look up member balance", {}, obs, dec)
    assert verdict.approved is False
    assert "irreversible" in verdict.reason.lower()


def test_validator_policy_layer_blocks_offsite_navigation():
    client = FakeLLMClient()
    pol = Policy.from_allowlist("http://good.test", ["/member/:id"])
    v = ActionValidator(client, pol)
    obs = _obs("http://good.test/", [], text="x")
    dec = Decision(action="navigate", url="http://evil.test/member/1", reason="go")
    verdict = v.validate("goal", {}, obs, dec)
    assert verdict.approved is False and verdict.layer == "policy"


def test_validator_approves_ongoal_typed_action():
    client = FakeLLMClient()
    pol = Policy.from_allowlist("http://h", ["/"])
    v = ActionValidator(client, pol)
    obs = _obs("http://h/", [ElementDigest(ref="e0", role="textbox", name="Member ID",
                                           editable=True)], text="x")
    dec = Decision(action="type", ref="e0", text="{{member_id}}", risk="reversible",
                   reason="enter")
    verdict = v.validate("look up member", {"member_id": "100001"}, obs, dec)
    assert verdict.approved is True


def _decode(raw: str) -> dict:
    import json
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Agentic router: reuse-vs-create by meaning (injected LLM test double)        #
# --------------------------------------------------------------------------- #
from cua.agent.router import CapabilityRouter
from cua.catalog.registry import CapabilityRegistry


def _named_capability():
    return _artifact().model_copy(update={
        "name": "lookup_member_balance",
        "goal": "look up member and read savings balance",
        "description": "Look up a member and read their current savings balance."})


def test_router_invokes_semantically_matching_capability(tmp_path):
    reg = CapabilityRegistry(directory=tmp_path)
    reg.save(_named_capability())
    router = CapabilityRouter(FakeLLMClient(), reg)
    # Different wording, same meaning -> should reuse the existing capability.
    d = router.route("read the current savings balance for member 100001")
    assert d.mode == "invoke"
    assert d.capability == "lookup_member_balance"
    assert d.args.get("member_id") == "100001"


def test_router_proposes_creation_when_nothing_matches(tmp_path):
    reg = CapabilityRegistry(directory=tmp_path)  # empty catalog
    router = CapabilityRouter(FakeLLMClient(), reg)
    d = router.route("open a brand new certificate account for member 424242")
    assert d.mode == "create"
    assert d.proposed_name  # a concrete capability name was proposed
    assert d.args.get("member_id") == "424242"


# --------------------------------------------------------------------------- #
# Multi-tenant variant resolution (engine helpers, browser-free)              #
# --------------------------------------------------------------------------- #
from cua.replay.engine import ReplayEngine
from cua.schema.artifact import Locator as _Loc, LocatorCandidate, VariantOverride


def _variant_artifact():
    r = Recorder(
        goal="g", capability_name="cap", description="d",
        target=TargetSpec(app_signature="sig", base_url="http://h",
                          allowed_routes=["/member/:member_id"]),
        provided_params={"member_id": "100001"},
    )
    r.record_navigate("http://h")
    r.record_read("savings_balance", "Savings balance")
    art = r.finalize(Checkpoint(type=CheckpointType.TEXT_PRESENT,
                                expression="Savings balance"))
    read_idx = art.steps[-1].index
    art.variants = [VariantOverride(
        variant_id="pioneer", base_url="http://h/?tenant=pioneer",
        locator_overrides={read_idx: _Loc(
            description="value for 'Ledger balance'",
            candidates=[LocatorCandidate(strategy=LocatorStrategy.XPATH,
                                         value="//td[.='Ledger balance']/td")])},
        success_checkpoint=Checkpoint(type=CheckpointType.TEXT_PRESENT,
                                      expression="Ledger balance"),
    )]
    return art, read_idx


def _engine(apply_overrides: bool, variant):
    e = object.__new__(ReplayEngine)
    e.apply_variant_overrides = apply_overrides
    e._variant = variant
    return e


# --------------------------------------------------------------------------- #
# Bounded, evidence-backed recovery (engine, browser-free)                    #
# --------------------------------------------------------------------------- #
from cua.replay.errors import Detection
from cua.schema.results import Outcome, ReplayResult, StepResult


class _NullEv:
    def event(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


class _RecSurface:
    """Fake surface: the interstitial notice clears after `clear_after` clicks
    (None = never clears, forcing exhaustion)."""

    def __init__(self, clear_after):
        self.clicks = 0
        self.clear_after = clear_after

    def click(self, loc):
        self.clicks += 1

    def current_url(self):
        return "http://h/member/1"

    def page_text(self):
        if self.clear_after is not None and self.clicks >= self.clear_after:
            return "Member detail Savings balance"
        return "Notice pending compliance review"


def _rec_engine(surface):
    e = object.__new__(ReplayEngine)
    e.surface = surface
    e.ev = _NullEv()
    e.escalation = None
    e.max_recovery_attempts = 2
    e._passed_login = True
    return e


def _rec_inputs():
    det = Detection(Condition.RECOVERABLE, "interstitial_notice", "notice")
    sr = StepResult(step_index=5, action="click", ok=True)
    res = ReplayResult(outcome=Outcome.HARD_FAILURE, artifact_id="x",
                       artifact_version=1)
    return det, sr, res


def test_recovery_counts_only_when_condition_clears():
    e = _rec_engine(_RecSurface(clear_after=1))
    det, sr, res = _rec_inputs()
    e._auto_recover(object(), det, sr, res)
    assert sr.recovered_from == "interstitial_notice"
    assert sr.ok is True
    assert res.outcome != Outcome.RECOVERY_EXHAUSTED


def test_recovery_exhausts_when_condition_persists():
    e = _rec_engine(_RecSurface(clear_after=None))
    det, sr, res = _rec_inputs()
    e._auto_recover(object(), det, sr, res)
    assert res.outcome == Outcome.RECOVERY_EXHAUSTED
    assert sr.ok is False
    assert e.surface.clicks == e.max_recovery_attempts  # bounded, not infinite


def test_variant_selected_by_param():
    art, _ = _variant_artifact()
    e = _engine(True, None)
    assert e._active_variant(art, {"_variant": "pioneer"}).variant_id == "pioneer"
    assert e._active_variant(art, {}) is None
    assert e._active_variant(art, {"_variant": "unknown"}) is None


def test_variant_override_applied_and_bypassed():
    art, read_idx = _variant_artifact()
    variant = art.variants[0]
    read_step = next(s for s in art.steps if s.index == read_idx)

    applied = _engine(True, variant)._effective_locator(art, read_step, {})
    assert "Ledger balance" in applied.candidates[0].value

    # With overrides bypassed, the base locator (Savings balance) is used instead.
    bypassed = _engine(False, variant)._effective_locator(art, read_step, {})
    assert "Savings balance" in bypassed.candidates[0].value


# --------------------------------------------------------------------------- #
# Policed human handoff: one-shot authorization grants                        #
# --------------------------------------------------------------------------- #
from cua.escalation.handoff import HandoffControl


class _FakeLocator:
    def __init__(self, page, kind, name):
        self.page, self.kind, self.name = page, kind, name

    def count(self):
        return 1

    def click(self):
        self.page.clicks.append((self.kind, self.name))

    def fill(self, v):
        self.page.fills.append((self.name, v))


class _FakePage:
    def __init__(self):
        self.clicks, self.fills = [], []

    def get_by_role(self, role, name=None):
        return _FakeLocator(self, "role:" + role, name)

    def get_by_label(self, label):
        return _FakeLocator(self, "label", label)

    def inner_text(self, sel):
        return ""


def test_one_shot_grant_authorizes_exactly_one_irreversible():
    pol = Policy.from_allowlist("http://h", ["/"])  # allow_irreversible=False
    from cua.schema.artifact import ActionType as AT
    # Blocked without a grant.
    assert not pol.check_action(AT.CLICK, RiskClass.IRREVERSIBLE).allowed
    tok = pol.issue_grant(AT.CLICK, RiskClass.IRREVERSIBLE)
    # Allowed once with the grant...
    assert pol.check_action(AT.CLICK, RiskClass.IRREVERSIBLE, grant=tok).allowed
    # ...and refused again (the grant is spent).
    assert not pol.check_action(AT.CLICK, RiskClass.IRREVERSIBLE, grant=tok).allowed


def test_grant_must_match_action_and_risk():
    pol = Policy.from_allowlist("http://h", ["/"])
    from cua.schema.artifact import ActionType as AT
    tok = pol.issue_grant(AT.CLICK, RiskClass.IRREVERSIBLE)
    assert not pol.check_action(AT.TYPE, RiskClass.IRREVERSIBLE, grant=tok).allowed


def test_handoff_control_polices_and_redacts():
    from cua.schema.artifact import ActionType as AT
    pol = Policy.from_allowlist("http://h", ["/"])
    page = _FakePage()
    ctl = HandoffControl(page, pol)
    # A human's irreversible click is REFUSED without a grant, and not executed.
    assert ctl.click("Confirm", risk=RiskClass.IRREVERSIBLE) is False
    assert page.clicks == []
    assert ctl.actions[-1].authorized is False
    # With a one-shot grant it executes exactly once.
    tok = pol.issue_grant(AT.CLICK, RiskClass.IRREVERSIBLE)
    assert ctl.click("Confirm", risk=RiskClass.IRREVERSIBLE, grant=tok) is True
    assert ("role:button", "Confirm") in page.clicks
    # Typed values are never recorded (regulated-data redaction).
    ctl.fill("Password", "supersecret")
    assert all("supersecret" not in str(a) for a in ctl.actions)


def test_variant_checkpoint_and_base_url():
    art, _ = _variant_artifact()
    variant = art.variants[0]
    nav_step = art.steps[0]
    # base_url is swapped for the entry navigation regardless of override toggle.
    assert _engine(False, variant)._effective_route(art, nav_step) == "http://h/?tenant=pioneer"
    # checkpoint override only when overrides are applied.
    assert _engine(True, variant)._effective_checkpoint(art).expression == "Ledger balance"
    assert _engine(False, variant)._effective_checkpoint(art).expression == "Savings balance"


# --------------------------------------------------------------------------- #
# Template resolution                                                         #
# --------------------------------------------------------------------------- #
def test_template_resolution_and_missing_param():
    from cua.replay.engine import ReplayEngine, MissingParam

    resolve = ReplayEngine._resolve
    fake = object.__new__(ReplayEngine)
    assert resolve(fake, "{{member_id}}", {"member_id": "5"}) == "5"
    with pytest.raises(MissingParam):
        resolve(fake, "{{missing}}", {})


# --------------------------------------------------------------------------- #
# Input completeness gate (never fabricate a required input)                  #
# --------------------------------------------------------------------------- #
from cua.service import caller_required_inputs, missing_inputs


def test_caller_required_inputs_and_missing_detection():
    art = _artifact()  # declares one required, non-sensitive param: member_id
    assert caller_required_inputs(art) == ["member_id"]
    # Absent, empty, or whitespace-only inputs all count as missing.
    assert missing_inputs(art, {}) == ["member_id"]
    assert missing_inputs(art, {"member_id": "  "}) == ["member_id"]
    assert missing_inputs(art, {"member_id": None}) == ["member_id"]
    # A real value satisfies the contract.
    assert missing_inputs(art, {"member_id": "100001"}) == []


def test_credentials_are_not_caller_supplied_inputs():
    # Sensitive params (credentials) come from secure config, never the request, so
    # they must NOT appear in the set the caller is required to supply.
    r = Recorder(
        goal="g", capability_name="cap", description="d",
        target=TargetSpec(app_signature="sig", base_url="http://h",
                          allowed_routes=["/login"]),
        provided_params={"member_id": "100001", "username": "u", "password": "p"},
    )
    r.record_type(ElementDigest(ref="e0", role="textbox", name="Username",
                                editable=True, attrs={}), "{{username}}")
    r.record_type(ElementDigest(ref="e1", role="textbox", name="Password",
                                editable=True, attrs={}), "{{password}}")
    r.record_type(ElementDigest(ref="e2", role="textbox", name="Member ID",
                                editable=True, attrs={"id": "mid"}), "{{member_id}}")
    art = r.finalize(Checkpoint(type=CheckpointType.TEXT_PRESENT, expression="x"))
    req = caller_required_inputs(art)
    assert "member_id" in req
    assert "username" not in req and "password" not in req
    # An invocation supplying only creds is still missing the member id.
    assert missing_inputs(art, {}) == ["member_id"]


# --------------------------------------------------------------------------- #
# Operation allowlist (semantic safety layer)                                 #
# --------------------------------------------------------------------------- #
from cua.safety.operations import OperationDecision, classify_operation


def test_allowlisted_operation_runs_autonomously():
    v = classify_operation("lookup_member_balance",
                           "look up the savings balance for member 100001")
    assert v.decision == OperationDecision.ALLOW


def test_human_only_intent_is_escalated_even_if_capability_exists():
    # Money movement always goes to a human, regardless of the routed capability.
    v = classify_operation("open_subaccount",
                           "transfer 500 from member 100001 to member 100002")
    assert v.decision == OperationDecision.ESCALATE
    assert v.matched == "transfer"


def test_unknown_operation_is_escalated_not_run():
    v = classify_operation("generate_statement",
                           "generate an account statement for member 100001")
    assert v.decision == OperationDecision.ESCALATE
    assert "not on the autonomous allowlist" in v.reason


# --------------------------------------------------------------------------- #
# Replay hardening: policy-gated verbs + output extraction validation         #
# --------------------------------------------------------------------------- #
from cua.replay.engine import ReplayEngine
from cua.schema.artifact import Step


def test_gate_blocks_disallowed_action_type():
    # A policy that only permits NAVIGATE must block a READ (the action-type allowlist
    # now covers read/press/wait/assert, not just the mutating verbs).
    e = object.__new__(ReplayEngine)
    e.policy = Policy(allowed_actions={ActionType.NAVIGATE})
    sr = StepResult(step_index=1, action="read", ok=True)
    res = ReplayResult(outcome=Outcome.SUCCESS, artifact_id="x", artifact_version=1)
    step = Step(index=1, action=ActionType.READ, risk=RiskClass.SAFE)
    assert e._gate(step, sr, res) is False
    assert res.outcome == Outcome.BLOCKED_BY_POLICY
    assert sr.ok is False


def test_read_that_extracts_nothing_is_hard_failure():
    # A read whose locator doesn't resolve returns None -> must be a hard failure at
    # that step, not a silent None that later passes as success.
    class _NoValueSurface:
        def read(self, locator):
            return None

    e = object.__new__(ReplayEngine)
    e.surface = _NoValueSurface()
    sr = StepResult(step_index=6, action="read", ok=True)
    res = ReplayResult(outcome=Outcome.SUCCESS, artifact_id="x", artifact_version=1)
    step = Step(index=6, action=ActionType.READ, output_name="savings_balance")
    outputs: dict = {}
    e._do_read(step, None, sr, res, outputs)
    assert sr.ok is False
    assert res.outcome == Outcome.HARD_FAILURE
    assert res.failed_step == 6
    assert "savings_balance" not in outputs
