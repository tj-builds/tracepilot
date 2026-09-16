"""Turns a successful discovery trace into a typed, reusable Artifact.

Two jobs matter here:
  1. Synthesize *robust, ordered* locator candidates from each element digest, so
     replay does not depend on a single brittle selector.
  2. Generalize concrete values into typed parameters ("100001" -> {{member_id}})
     and mark credentials as sensitive so they are never persisted.
"""
from __future__ import annotations

import hashlib
import json
import uuid

from ..schema.artifact import (
    ActionType,
    Artifact,
    Checkpoint,
    Locator,
    LocatorCandidate,
    LocatorStrategy,
    OutputSpec,
    ParamSpec,
    ParamType,
    Provenance,
    RiskClass,
    Step,
    TargetSpec,
)
from ..surface.base import ElementDigest

_SENSITIVE = {"username", "password", "pin", "ssn", "token"}


def synthesize_locator(elem: ElementDigest) -> Locator:
    cands: list[LocatorCandidate] = []
    role = elem.role
    name = elem.name or ""

    if name:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.ROLE, role=role, value=name, confidence=0.9,
            rationale="Accessibility role + accessible name; survives markup churn "
                      "and maps onto the a11y tree (works on desktop too).",
        ))
    if elem.editable and name:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.LABEL, value=name, confidence=0.8,
            rationale="Associated <label> text; stable for form fields.",
        ))
    ph = elem.attrs.get("placeholder")
    if ph:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.PLACEHOLDER, value=ph, confidence=0.6,
            rationale="Placeholder text fallback.",
        ))
    el_id = elem.attrs.get("id")
    if el_id:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.CSS, value=f"#{el_id}", confidence=0.5,
            rationale="Element id; brittle if ids are generated, so ranked below "
                      "semantic strategies.",
        ))
    if role in ("button", "link") and name:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.TEXT, value=name, confidence=0.4,
            rationale="Visible text fallback for buttons/links.",
        ))
    if not cands:
        cands.append(LocatorCandidate(
            strategy=LocatorStrategy.TEXT, value=name or role, confidence=0.2,
            rationale="Last-resort text match.",
        ))
    return Locator(description=name or role, candidates=cands)


def read_locator_for_label(label: str) -> Locator:
    """Robust locator for a value shown next to a label in a table row."""
    return Locator(
        description=f"value for '{label}'",
        candidates=[
            LocatorCandidate(
                strategy=LocatorStrategy.XPATH,
                value=f"//td[normalize-space()='{label}']/following-sibling::td[1]",
                confidence=0.8,
                rationale="Anchored on the human-readable row label, not on position "
                          "or generated markup.",
            ),
        ],
    )


class Recorder:
    def __init__(self, goal: str, capability_name: str, description: str,
                 target: TargetSpec, provided_params: dict):
        self.goal = goal
        self.capability_name = capability_name
        self.description = description
        self.target = target
        self.provided_params = provided_params
        self.steps: list[Step] = []
        self.params: dict[str, ParamSpec] = {}
        self.outputs: list[OutputSpec] = []
        self._i = 0

    # -- value generalization ---------------------------------------------- #
    def _generalize(self, text: str) -> str:
        if text.startswith("{{") and text.endswith("}}"):
            self._register_param(text[2:-2].strip())
            return text
        for pname, pval in self.provided_params.items():
            if str(pval) == text:
                name = pname.lstrip("_")
                self._register_param(name)
                return f"{{{{{name}}}}}"
        return text  # literal (non-parameterized) value

    def _register_param(self, name: str) -> None:
        if name in self.params:
            return
        self.params[name] = ParamSpec(
            name=name,
            type=ParamType.STRING,
            required=True,
            description=f"Input parameter '{name}'.",
            example=self.provided_params.get(name)
            if name.lower() not in _SENSITIVE else None,
            sensitive=name.lower() in _SENSITIVE,
        )

    # -- recording API ------------------------------------------------------ #
    def _next(self) -> int:
        i = self._i
        self._i += 1
        return i

    def record_navigate(self, url: str) -> None:
        self.steps.append(Step(index=self._next(), action=ActionType.NAVIGATE,
                               route=url, description=f"navigate to {url}"))

    def record_type(self, elem: ElementDigest, text: str) -> None:
        value = self._generalize(text)
        self.steps.append(Step(
            index=self._next(), action=ActionType.TYPE,
            locator=synthesize_locator(elem), value=value,
            risk=RiskClass.REVERSIBLE,
            description=f"type into '{elem.name or elem.role}'",
        ))

    def record_select(self, elem: ElementDigest, value: str) -> None:
        self.steps.append(Step(
            index=self._next(), action=ActionType.SELECT,
            locator=synthesize_locator(elem), value=self._generalize(value),
            risk=RiskClass.REVERSIBLE,
            description=f"select '{value}' in '{elem.name or elem.role}'",
        ))

    def record_click(self, elem: ElementDigest, risk: RiskClass) -> None:
        self.steps.append(Step(
            index=self._next(), action=ActionType.CLICK,
            locator=synthesize_locator(elem), risk=risk,
            description=f"click '{elem.name or elem.role}'",
        ))

    def record_press(self, key: str) -> None:
        self.steps.append(Step(index=self._next(), action=ActionType.PRESS,
                               key=key, description=f"press {key}"))

    def record_read(self, output_name: str, label: str) -> None:
        # Dedupe: a model may re-issue a read for the same output several times; record
        # a declared output (and its READ step) only once.
        if any(o.name == output_name for o in self.outputs):
            return
        idx = self._next()
        self.steps.append(Step(
            index=idx, action=ActionType.READ,
            locator=read_locator_for_label(label),
            output_name=output_name,
            description=f"read '{output_name}' from '{label}' row",
        ))
        self.outputs.append(OutputSpec(
            name=output_name, type=ParamType.STRING,
            description=f"Extracted value for {output_name}.", source_step=idx,
        ))

    # -- finalize ----------------------------------------------------------- #
    def finalize(self, success_checkpoint: Checkpoint,
                 discovery_evidence: str | None = None,
                 planner: str | None = None,
                 model: str | None = None) -> Artifact:
        steps = self.steps
        # Content hash of the recorded flow -> a stable integrity anchor in provenance.
        flow_json = json.dumps([s.model_dump(mode="json") for s in steps],
                               sort_keys=True)
        flow_sha = hashlib.sha256(flow_json.encode("utf-8")).hexdigest()
        return Artifact(
            id=f"{self.capability_name}-{uuid.uuid4().hex[:8]}",
            name=self.capability_name,
            description=self.description,
            goal=self.goal,
            target=self.target,
            params=list(self.params.values()),
            outputs=self.outputs,
            steps=steps,
            success_checkpoint=success_checkpoint,
            discovery_evidence=discovery_evidence,
            provenance=Provenance(
                run_id=discovery_evidence, planner=planner, model=model,
                flow_sha256=flow_sha, curated=False),
        )
