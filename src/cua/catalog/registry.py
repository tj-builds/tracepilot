"""A catalog of saved capabilities an AI agent can discover and invoke by name.

Each artifact is exposed as a callable tool with a typed input schema (derived from
its params) and a typed output schema -- the same shape an agent function-calling
layer expects. This is the "agent-facing capability interface" seam.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ARTIFACTS_DIR
from ..schema.artifact import Artifact


class CapabilityRegistry:
    def __init__(self, directory: Path | None = None):
        self.dir = directory or ARTIFACTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: Artifact) -> Path:
        path = self.dir / f"{artifact.name}.json"
        path.write_text(artifact.to_json(), encoding="utf-8")
        return path

    def load(self, name: str) -> Artifact:
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no capability named '{name}'")
        return self._validated(Artifact.model_validate_json(
            path.read_text(encoding="utf-8")))

    def load_path(self, path: str | Path) -> Artifact:
        p = Path(path)
        return self._validated(Artifact.model_validate_json(
            p.read_text(encoding="utf-8")))

    @staticmethod
    def _validated(art: Artifact) -> Artifact:
        issues = art.validate_contract()
        if issues:
            raise ValueError(
                f"artifact '{art.name}' fails contract validation: " + "; ".join(issues))
        return art

    def list(self) -> list[Artifact]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(Artifact.model_validate_json(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def tool_specs(self) -> list[dict]:
        """Render capabilities as JSON-schema tool specs for agent function-calling."""
        specs = []
        for a in self.list():
            props, required = {}, []
            for p in a.params:
                props[p.name] = {"type": _json_type(p.type.value),
                                 "description": p.description}
                if p.required:
                    required.append(p.name)
            specs.append({
                "name": a.name,
                "description": a.description,
                "approval_state": a.approval_state,
                "parameters": {"type": "object", "properties": props,
                               "required": required},
                "returns": {o.name: o.type.value for o in a.outputs},
            })
        return specs


def _json_type(t: str) -> str:
    return {"string": "string", "number": "number", "boolean": "boolean"}.get(t, "string")
