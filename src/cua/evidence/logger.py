"""Structured, redacted evidence logging.

Every run (discovery or replay) gets its own timestamped directory under
/evidence/ containing a JSONL event log plus any richer signals (screenshots).
All string payloads pass through the redactor so secrets / regulated data never
land on disk.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import EVIDENCE_DIR, REPO_ROOT
from ..safety.policy import redact


class EvidenceLog:
    def __init__(self, run_kind: str, run_id: str | None = None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"{run_kind}-{ts}"
        self.dir = EVIDENCE_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self.rel_dir = self.dir.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            self.rel_dir = str(self.dir)
        self.log_path = self.dir / "run.jsonl"
        self.run_kind = run_kind
        self._seq = 0
        self.event("run_started", run_kind=run_kind)

    def event(self, kind: str, **fields: Any) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": kind,
        }
        for k, v in fields.items():
            record[k] = redact(v) if isinstance(v, str) else v
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_screenshot(self, surface, label: str) -> str | None:
        """Write a screenshot INTO this run's directory and return a repo-relative
        reference for logs/results.

        The surface is handed an *absolute* path so the file always lands in the run
        directory regardless of the process's current working directory (a relative
        path would resolve against cwd and scatter screenshots outside the repo). The
        returned value stays repo-relative so nothing leaks the absolute OS path into
        committed evidence.
        """
        name = f"{self._seq:03d}-{label}.png"
        written = surface.screenshot(str(self.dir / name))
        if not written:
            return None
        return f"{self.rel_dir}/{name}"

    def write_json(self, name: str, obj: Any) -> str:
        path = self.dir / name
        text = obj if isinstance(obj, str) else json.dumps(obj, indent=2)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def finish(self, outcome: str, **fields: Any) -> None:
        self.event("run_finished", outcome=outcome, **fields)
