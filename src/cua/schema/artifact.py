"""The capability artifact schema.

This is the contract between three parties:
  1. the discovery agent that *writes* it,
  2. the deterministic replay engine that *executes* it, and
  3. an AI agent (and human reviewer) that *invokes/reads* it.

Design principles
-----------------
* Typed + versioned + serializable (JSON). Decoupled from the raw model transcript.
* Locators are *ordered candidate lists*, not single selectors -> resilient targeting
  that degrades gracefully instead of breaking on the first missing attribute.
* Explicit param/output contract so an agent can call it like a function.
* Canonicalized routes (/member/12345 -> /member/:id) + per-variant overrides so one
  artifact can be reused across tenants running the same vendor product.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Locators: ordered, self-describing, robustness-ranked                       #
# --------------------------------------------------------------------------- #
class LocatorStrategy(str, Enum):
    """How a control is identified. Ordered roughly most→least robust.

    Bias toward strategies that survive when the surface has no clean DOM.
    `role` and `label` map onto the accessibility tree and work on desktop too;
    `coordinates` is the last-resort screenshot fallback.
    """

    ROLE = "role"            # accessibility role + accessible name (a11y tree)
    LABEL = "label"          # associated <label> / aria-label text
    TEST_ID = "test_id"      # data-testid etc. (rare in legacy apps)
    TEXT = "text"            # visible text content
    PLACEHOLDER = "placeholder"
    ALT = "alt"
    CSS = "css"
    XPATH = "xpath"          # structural; brittle but sometimes the only option
    STRUCTURAL = "structural"  # nth-of-role within a labelled region
    COORDINATES = "coordinates"  # screenshot x/y fraction; desktop / opaque UIs


class LocatorCandidate(BaseModel):
    strategy: LocatorStrategy
    value: str = Field(description="Selector value / accessible name / coordinate pair.")
    # For ROLE strategy, the ARIA role (e.g. 'textbox') lives here.
    role: str | None = None
    rationale: str = Field(
        default="",
        description="Why this candidate was chosen and how robust it is expected to be.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Locator(BaseModel):
    """An ordered set of candidates tried in priority order at replay time."""

    description: str = Field(description="Human label for the target control.")
    candidates: list[LocatorCandidate] = Field(default_factory=list)

    def primary(self) -> LocatorCandidate | None:
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------- #
# Actions & steps                                                             #
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    PRESS = "press"          # keyboard key, e.g. Enter
    WAIT_FOR = "wait_for"    # wait for a locator / checkpoint
    READ = "read"           # extract data into a declared output
    ASSERT = "assert"       # verify a checkpoint mid-flow


class RiskClass(str, Enum):
    """Reversibility classification, drives the safety policy at replay."""

    SAFE = "safe"                # read-only / navigation
    REVERSIBLE = "reversible"    # form fill that can be abandoned
    IRREVERSIBLE = "irreversible"  # submit/confirm/delete/transfer


class Step(BaseModel):
    index: int
    action: ActionType
    description: str = ""
    # Target control (omit for navigate / press / pure waits).
    locator: Locator | None = None
    # Literal value OR a "{{param}}" template referencing an input parameter.
    value: str | None = None
    # For NAVIGATE: a canonicalized route pattern, e.g. "/member/:member_id".
    route: str | None = None
    # For READ: which output this step populates.
    output_name: str | None = None
    # For PRESS: the key.
    key: str | None = None
    risk: RiskClass = RiskClass.SAFE
    # Optional per-step checkpoint asserted after the action.
    checkpoint: "Checkpoint | None" = None


# --------------------------------------------------------------------------- #
# Params, outputs, checkpoints                                                #
# --------------------------------------------------------------------------- #
class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ParamSpec(BaseModel):
    name: str
    type: ParamType = ParamType.STRING
    required: bool = True
    description: str = ""
    example: Any | None = None
    # If true, the value is redacted from logs/evidence and never persisted raw.
    sensitive: bool = False


class OutputSpec(BaseModel):
    name: str
    type: ParamType = ParamType.STRING
    description: str = ""
    # The step index that reads this value (traceability).
    source_step: int | None = None


class CheckpointType(str, Enum):
    URL_MATCHES = "url_matches"        # regex against current URL/route
    ELEMENT_VISIBLE = "element_visible"  # a locator resolves & is visible
    TEXT_PRESENT = "text_present"      # substring/regex present on page
    TEXT_ABSENT = "text_absent"


class Checkpoint(BaseModel):
    type: CheckpointType
    expression: str = Field(description="Regex / text / route depending on type.")
    locator: Locator | None = None
    description: str = ""


# --------------------------------------------------------------------------- #
# Target + tenant/variant reuse                                               #
# --------------------------------------------------------------------------- #
class TargetSpec(BaseModel):
    """Identifies the *product* this capability drives, not one concrete URL.

    `app_signature` is the vendor-product identity shared across tenants; two
    tenants running the same product share a signature and can share artifacts.
    `base_url` is resolved per-tenant at invocation time.
    """

    app_signature: str = Field(description="Vendor/product identity, e.g. 'acme-corebanking'.")
    base_url: str = Field(description="Default/base tenant base URL used at record time.")
    allowed_routes: list[str] = Field(
        default_factory=list,
        description="Canonicalized route patterns the capability is permitted to touch.",
    )


class Provenance(BaseModel):
    """Where this capability came from -- traceable to the discovery run that made
    it, without embedding the raw model transcript (kept only as redacted evidence).
    """

    run_id: str | None = None            # evidence dir of the discovery run
    planner: str | None = None           # e.g. "llm:openai"
    model: str | None = None             # concrete model id, e.g. "gpt-4o"
    flow_sha256: str | None = None       # content hash of the recorded step flow
    curated: bool = False                # whether a human reviewed/edited it


class VariantOverride(BaseModel):
    """Per-tenant/version specialization of an otherwise-shared artifact.

    A variant carries ONLY what differs from the base artifact -- a per-tenant
    `base_url`, the handful of `locator_overrides` (keyed by step index) whose target
    control is labelled differently, and optionally a `success_checkpoint` override
    when the tenant relabels the very text the checkpoint asserts. Everything else is
    inherited, so one artifact serves many tenants of the same vendor product.
    """

    variant_id: str
    base_url: str | None = None
    # Locator overrides keyed by step index (only where the variant differs).
    locator_overrides: dict[int, Locator] = Field(default_factory=dict)
    # Optional checkpoint override when the tenant relabels the asserted text.
    success_checkpoint: Checkpoint | None = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# The artifact                                                                #
# --------------------------------------------------------------------------- #
class Artifact(BaseModel):
    schema_version: str = SCHEMA_VERSION
    # Stable identity + human-visible versioning.
    id: str
    name: str = Field(description="Callable capability name, e.g. 'lookup_member_balance'.")
    version: int = 1
    description: str = Field(description="What the capability does, for humans and agents.")
    goal: str = Field(description="Original natural-language goal used at discovery.")

    target: TargetSpec
    params: list[ParamSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    success_checkpoint: Checkpoint

    # Multi-tenant reuse.
    variants: list[VariantOverride] = Field(default_factory=list)

    # Provenance / review metadata (decoupled from the raw transcript).
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "discovery-agent"
    approval_state: str = Field(default="draft", description="draft | approved")
    discovery_evidence: str | None = None  # path to discovery run log
    provenance: Provenance | None = None    # traceable origin of this capability
    stability_score: float | None = None   # fraction of successful multi-run replays

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)

    def validate_contract(self) -> list[str]:
        """Structural invariants of a *callable capability*, checked on load.

        These catch the ways a recorded artifact can be internally inconsistent
        before it is ever replayed: a READ that doesn't feed a declared output, a
        templated value that references a parameter the artifact doesn't declare, an
        output traced to a non-existent step, or a checkpoint that would leak a
        sensitive parameter. Returns a list of human-readable issues ([] == valid).
        """
        import re as _re

        issues: list[str] = []
        param_names = {p.name for p in self.params}
        output_names = {o.name for o in self.outputs}
        step_indexes = {s.index for s in self.steps}
        tmpl = _re.compile(r"\{\{\s*(\w+)\s*\}\}")

        for s in self.steps:
            if s.action == ActionType.READ and (
                    not s.output_name or s.output_name not in output_names):
                issues.append(
                    f"step {s.index}: READ does not name a declared output")
            for ref in tmpl.findall(s.value or ""):
                if ref not in param_names:
                    issues.append(
                        f"step {s.index}: references undeclared parameter '{ref}'")
        for o in self.outputs:
            if o.source_step is not None and o.source_step not in step_indexes:
                issues.append(
                    f"output '{o.name}': source_step {o.source_step} is not a step")
        for p in self.params:
            if p.sensitive and p.name in (self.success_checkpoint.expression or ""):
                issues.append(
                    f"checkpoint must not reference sensitive parameter '{p.name}'")
        return issues

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


Step.model_rebuild()
