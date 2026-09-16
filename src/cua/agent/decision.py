"""The structured decision the planner returns each step."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecisionAction = Literal[
    "navigate", "click", "type", "select", "press", "read", "done", "give_up"
]


class Decision(BaseModel):
    action: DecisionAction
    ref: str | None = Field(default=None, description="Element ref from the observation.")
    text: str | None = None       # for type
    value: str | None = None      # for select
    key: str | None = None        # for press
    url: str | None = None        # for navigate
    output_name: str | None = None  # for read
    risk: Literal["safe", "reversible", "irreversible"] = "safe"
    reason: str = ""
