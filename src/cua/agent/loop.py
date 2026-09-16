"""The LLM-driven discovery loop: observe -> decide -> VALIDATE -> act, then record.

Discovery is human-initiated and supervised, so its policy is permissive enough to
complete the goal (including recording an irreversible step), while every action
passes through two guardrails before it runs: the deterministic allowlist choke
point and the LLM `ActionValidator` critic. The output is a structured Artifact --
the raw model transcript is kept only as redacted evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Config
from ..evidence.logger import EvidenceLog
from ..safety.policy import Policy
from ..schema.artifact import ActionType, Artifact, Checkpoint, RiskClass, TargetSpec
from ..surface.base import ElementDigest, Surface
from .planner import Planner
from .recorder import Recorder, read_locator_for_label, synthesize_locator
from .validator import ActionValidator

_TEMPLATE = re.compile(r"\{\{\s*([\w]+)\s*\}\}")
# How many consecutive validator rejections before we stop replanning and give up.
_MAX_REJECTS = 3
# How many consecutive un-actionable decisions (e.g. unresolved refs) before we stop.
_MAX_NOOPS = 3


def _resolve_template(text: str, params: dict) -> str:
    def sub(m):
        return str(params.get(m.group(1), m.group(0)))
    return _TEMPLATE.sub(sub, text or "")


@dataclass
class DiscoveryResult:
    status: str                      # success | gave_up | blocked | max_steps
    artifact: Artifact | None = None
    outputs: dict = field(default_factory=dict)
    reason: str = ""
    evidence_dir: str = ""
    steps_taken: int = 0


class DiscoveryAgent:
    def __init__(self, surface: Surface, planner: Planner, policy: Policy,
                 evidence: EvidenceLog, cfg: Config,
                 validator: ActionValidator | None = None):
        self.surface = surface
        self.planner = planner
        self.policy = policy
        self.ev = evidence
        self.cfg = cfg
        self.validator = validator

    def run(self, goal: str, target: TargetSpec, params: dict,
            capability_name: str, description: str,
            success_checkpoint: Checkpoint,
            outputs_spec: list[dict] | None = None) -> DiscoveryResult:
        recorder = Recorder(goal, capability_name, description, target, params)
        outputs: dict = {}
        objectives = {
            "outputs": outputs_spec or [],
            "success": {"type": success_checkpoint.type.value,
                        "expression": success_checkpoint.expression},
        }

        self.ev.event("goal", goal=goal, capability=capability_name,
                      planner=self.planner.name)

        nav = self.policy.check_navigation(target.base_url)
        if not nav.allowed:
            self.ev.finish("blocked", reason=nav.reason)
            return DiscoveryResult("blocked", reason=nav.reason,
                                   evidence_dir=str(self.ev.dir))
        self.surface.navigate(target.base_url)
        recorder.record_navigate(target.base_url)

        history: list[str] = []
        rejects = 0
        noops = 0
        for step_no in range(self.cfg.max_steps):
            obs = self.surface.observe()
            self.ev.event("observe", url=obs.url, title=obs.title,
                          n_controls=len(obs.elements))

            nav = self.policy.check_navigation(obs.url)
            if not nav.allowed:
                self.ev.finish("blocked", reason=nav.reason, url=obs.url)
                return DiscoveryResult("blocked", reason=nav.reason,
                                       evidence_dir=str(self.ev.dir),
                                       steps_taken=step_no)

            decision = self.planner.decide(goal, params, obs, history, objectives)
            self.ev.event("decide", action=decision.action, ref=decision.ref,
                          reason=decision.reason, risk=decision.risk)

            # ---- Guardrail: validate the proposed action before acting. ---- #
            if self.validator and decision.action not in ("done", "give_up"):
                verdict = self.validator.validate(goal, params, obs, decision)
                self.ev.event("validate", approved=verdict.approved, layer=verdict.layer,
                              risk=verdict.risk, reason=verdict.reason,
                              concerns=verdict.concerns)
                if not verdict.approved:
                    rejects += 1
                    history.append(
                        f"validator[{verdict.layer}] rejected {decision.action}: "
                        f"{verdict.reason}")
                    if rejects >= _MAX_REJECTS:
                        shot = self.ev.save_screenshot(self.surface, "rejected")
                        self.ev.finish("gave_up", reason="validator rejected repeatedly",
                                       screenshot=shot)
                        return DiscoveryResult("gave_up",
                                               reason="validator rejected repeatedly",
                                               evidence_dir=str(self.ev.dir),
                                               steps_taken=step_no)
                    continue
                rejects = 0

            if decision.action == "done":
                # Strict success gate: the checkpoint must verify AND every declared
                # output must have actually been extracted with a non-empty value. A
                # bare "done" (checkpoint text present but nothing read) does NOT pass.
                missing = [o["name"] for o in (outputs_spec or [])
                           if not str(outputs.get(o["name"]) or "").strip()]
                if self._verify(success_checkpoint, obs) and not missing:
                    art = recorder.finalize(
                        success_checkpoint, self.ev.rel_dir,
                        planner=self.planner.name, model=self._model_id())
                    self.ev.event("goal_reached")
                    self.ev.finish("success", capability=art.name)
                    return DiscoveryResult("success", artifact=art, outputs=outputs,
                                           evidence_dir=str(self.ev.dir),
                                           steps_taken=step_no)
                reason = ("checkpoint not satisfied" if missing == [] else
                          f"declared outputs not captured: {missing}")
                self.ev.event("done_rejected", reason=reason)
                history.append(f"done requested but {reason}")
                continue

            if decision.action == "give_up":
                shot = self.ev.save_screenshot(self.surface, "give_up")
                self.ev.finish("gave_up", reason=decision.reason, screenshot=shot)
                return DiscoveryResult("gave_up", reason=decision.reason,
                                       evidence_dir=str(self.ev.dir),
                                       steps_taken=step_no)

            elem = self._find(obs.elements, decision.ref)

            if decision.action == "navigate":
                noops = 0
                self.surface.navigate(decision.url)
                recorder.record_navigate(decision.url)
                history.append(f"navigated to {decision.url}")

            elif decision.action == "type" and elem:
                noops = 0
                real = _resolve_template(decision.text or "", params)
                self.surface.type_text(synthesize_locator(elem), real)
                recorder.record_type(elem, decision.text or "")
                history.append(f"typed into '{elem.name}'")

            elif decision.action == "select" and elem:
                noops = 0
                real = _resolve_template(decision.value or "", params)
                self.surface.select(synthesize_locator(elem), real)
                recorder.record_select(elem, decision.value or "")
                history.append(f"selected in '{elem.name}'")

            elif decision.action == "click" and elem:
                noops = 0
                risk = RiskClass(decision.risk)
                if risk == RiskClass.IRREVERSIBLE:
                    self.ev.event("irreversible_confirmed", control=elem.name)
                self.surface.click(synthesize_locator(elem))
                recorder.record_click(elem, risk)
                history.append(f"clicked '{elem.name}'")

            elif decision.action == "press":
                noops = 0
                self.surface.press(decision.key or "Enter")
                recorder.record_press(decision.key or "Enter")
                history.append(f"pressed {decision.key}")

            elif decision.action == "read":
                noops = 0
                label = decision.value or decision.output_name or "value"
                raw = self.surface.read(read_locator_for_label(label))
                outputs[decision.output_name or "value"] = raw
                recorder.record_read(decision.output_name or "value", label)
                self.ev.event("read", output=decision.output_name, value=raw or "")
                history.append(f"read {decision.output_name}")
            else:
                # The chosen action couldn't be carried out (e.g. the ref didn't
                # resolve). Bound this: don't burn LLM budget spinning on a screen we
                # can't act on -- after a few consecutive no-ops, stop.
                noops += 1
                history.append(f"no-op ({decision.action}); ref '{decision.ref}' "
                               f"not found ({noops}/{_MAX_NOOPS})")
                if noops >= _MAX_NOOPS:
                    shot = self.ev.save_screenshot(self.surface, "stuck")
                    self.ev.finish("gave_up", reason="repeated no-op: unresolved refs",
                                   screenshot=shot)
                    return DiscoveryResult(
                        "gave_up", reason="repeated no-op (unresolvable control refs)",
                        evidence_dir=str(self.ev.dir), steps_taken=step_no)

        shot = self.ev.save_screenshot(self.surface, "max_steps")
        self.ev.finish("max_steps", screenshot=shot)
        return DiscoveryResult("max_steps", reason="step budget exhausted",
                               evidence_dir=str(self.ev.dir),
                               steps_taken=self.cfg.max_steps)

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _find(elements: list[ElementDigest], ref: str | None) -> ElementDigest | None:
        if not ref:
            return None
        # Models often echo the digest's "#e0" display form; normalize to "e0".
        norm = ref.strip().lstrip("#").strip()
        return next((e for e in elements if e.ref == norm), None)

    def _verify(self, cp: Checkpoint, obs) -> bool:
        from ..replay.errors import verify_checkpoint
        return verify_checkpoint(cp, obs.url, obs.text)

    def _model_id(self) -> str:
        return self.cfg.model_label
