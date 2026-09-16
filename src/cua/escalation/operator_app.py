"""A deliberately minimal (mock) operator console.

This is the *bare* human-facing surface the brief allows us to mock. It lists pending
intervention requests written by the escalation handler, shows their context, and
lets an operator (a) see how to take control of the same live session and (b) signal
resume. The real, well-reasoned part -- the same-session control-transfer mechanism
-- lives in handoff.py; this is just a thin window onto it.

Run:  python -m cua.escalation.operator_app
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .handoff import ESCALATION_DIR

app = FastAPI(title="CUA Operator Console (mock)")


def _pending() -> list[dict]:
    ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(ESCALATION_DIR.glob("intv-*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_resolved"] = (ESCALATION_DIR / f"{data['request_id']}.resume").exists()
            out.append(data)
        except Exception:
            continue
    return out


@app.get("/api/interventions")
def interventions():
    return JSONResponse(_pending())


@app.post("/api/interventions/{request_id}/resume")
def resume(request_id: str, actions: list[str] | None = None):
    """Signal that the human finished; automation may take the session back."""
    (ESCALATION_DIR / f"{request_id}.resume").write_text(
        json.dumps({"actions": actions or ["manual operator actions"]}),
        encoding="utf-8")
    return {"ok": True, "request_id": request_id}


@app.get("/", response_class=HTMLResponse)
def index():
    rows = ""
    for r in _pending():
        status = "resolved" if r.get("_resolved") else "PENDING"
        cdp = r.get("cdp_endpoint") or "(none — in-process handoff)"
        rows += f"""<tr>
          <td>{r['request_id']}</td><td>{r.get('reason','')}</td>
          <td>step {r.get('step_index')}</td><td>{r.get('current_url','')}</td>
          <td><code>{cdp}</code></td>
          <td>{status}</td>
          <td>
            <form method="post" action="/ui/{r['request_id']}/resume">
              <button type="submit">Take control &amp; resume</button>
            </form>
          </td></tr>"""
    return f"""<!doctype html><html><body style="font-family:sans-serif">
      <h3>Operator console (mock) &mdash; pending interventions</h3>
      <p>To take control of the <b>same live session</b>: open Chrome at
      <code>chrome://inspect</code>, add the request's <b>CDP endpoint</b>
      (<code>localhost:&lt;port&gt;</code>) under "Discover network targets", inspect
      the page, perform the manual steps, then click resume.</p>
      <table border="1" cellpadding="6" cellspacing="0">
      <tr><th>id</th><th>reason</th><th>step</th><th>url</th><th>cdp endpoint</th>
      <th>status</th><th></th></tr>
      {rows or '<tr><td colspan=7>none</td></tr>'}
      </table></body></html>"""


@app.post("/ui/{request_id}/resume", response_class=HTMLResponse)
def ui_resume(request_id: str):
    resume(request_id)
    return HTMLResponse(f"<p>Resumed {request_id}. <a href='/'>back</a></p>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5002)
