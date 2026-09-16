"""Surface abstraction: the seam between perception/action and the recorded flow.

Everything above this interface (the agent loop, the recorder, the replay engine)
is written against `Surface` + `Locator`, NOT against Playwright. That is what lets
the same artifact schema and replay engine extend to:

  * a legacy web app (a WebSurface variant with coordinate/OCR fallbacks),
  * a native desktop app (an AccessibilitySurface driving the OS a11y tree),

without changing the artifact format. Perception is normalized into an
`Observation` (a compact, model-friendly digest of interactable controls plus the
accessibility snapshot), and action is expressed against a `Locator` (the ordered
candidate list from the schema).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schema.artifact import Locator


@dataclass
class ElementDigest:
    """One interactable control, normalized across surfaces."""

    ref: str                 # stable-within-observation handle the agent can cite
    role: str                # accessibility role (button, textbox, link, ...)
    name: str                # accessible name / visible label
    value: str | None = None
    editable: bool = False
    # Raw hints the recorder uses to synthesize robust locator candidates.
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Observation:
    """A normalized snapshot of the current surface state."""

    url: str
    title: str
    elements: list[ElementDigest]
    text: str = ""           # visible text (trimmed) for text checkpoints
    screenshot_path: str | None = None

    def digest(self, limit: int = 60) -> str:
        """Compact, token-frugal rendering for the LLM planner."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "CONTROLS:"]
        for e in self.elements[:limit]:
            v = f' value="{e.value}"' if e.value else ""
            ed = " [editable]" if e.editable else ""
            lines.append(f'  #{e.ref} {e.role} "{e.name}"{v}{ed}')
        return "\n".join(lines)


@dataclass
class ActResult:
    ok: bool
    detail: str = ""
    locator_used: str | None = None
    # Which rung of the ordered locator ladder resolved (0 = primary/most robust),
    # and how many candidates existed -- the raw signal for drift detection.
    rung: int | None = None
    candidates: int | None = None


class Surface(ABC):
    """Abstract computer-use surface. Concrete impls: web, desktop, screenshot."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def observe(self, screenshot: bool = False) -> Observation: ...

    @abstractmethod
    def navigate(self, url: str) -> ActResult: ...

    @abstractmethod
    def click(self, locator: Locator) -> ActResult: ...

    @abstractmethod
    def type_text(self, locator: Locator, text: str) -> ActResult: ...

    @abstractmethod
    def select(self, locator: Locator, value: str) -> ActResult: ...

    @abstractmethod
    def press(self, key: str) -> ActResult: ...

    @abstractmethod
    def read(self, locator: Locator) -> str | None: ...

    @abstractmethod
    def current_url(self) -> str: ...

    @abstractmethod
    def page_text(self) -> str: ...

    @abstractmethod
    def screenshot(self, path: str) -> str | None: ...

    # Escalation seam: hand the *same* live session to a human and take it back.
    @abstractmethod
    def to_headed_for_handoff(self) -> str:
        """Surface the live session for manual control; return a human-usable ref."""

    @abstractmethod
    def resume_after_handoff(self) -> None: ...
