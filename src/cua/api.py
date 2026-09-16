"""Optional HTTP surface -- the agent-facing capability interface (a stretch goal).

The brief scopes the *agent-facing product* (the thing that decides what to do) out
of this system; what it does invite is "a small tool/function-calling surface, or an
API endpoint" that an agent could discover and invoke by name. That is what this file
is: a thin HTTP wrapper over the exact same `service` functions the CLI uses. A
minimal chat page is served at `/` as a convenience for a human to exercise it, but
the real deliverable is the JSON API, not the page.

Endpoints
---------
  GET  /capabilities     -> the callable catalog + the operation allowlist
  POST /run              -> route a natural-language request (reuse / human / refuse)
  POST /invoke/{name}    -> invoke a saved capability by name with typed args
  GET  /                 -> a minimal chat page that POSTs to /run

Nothing here changes the safety model: /run goes through the same operation allowlist
(non-allowed intents are routed to a human) and /invoke through the same input
completeness gate and risk policy. There is no LLM fallback -- without a key the
model-driven endpoints return 503.

Run:  python -m cua.api        # uvicorn on http://127.0.0.1:5003
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .catalog.registry import CapabilityRegistry
from .config import CONFIG
from .safety.operations import ALLOWED_OPERATIONS, HUMAN_ONLY_INTENTS
from .service import RunOutcome, invoke_capability, run_request

app = FastAPI(title="CUA Agent-Facing API")


class RunBody(BaseModel):
    request: str = Field(description="Natural-language request.")
    allow_irreversible: bool = False
    escalate: bool = False


class InvokeBody(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    allow_irreversible: bool = False
    escalate: bool = False


def _serialize_result(result) -> dict | None:
    if result is None:
        return None
    return {
        "outcome": result.outcome.value,
        "outputs": result.outputs,
        "business_code": result.business_code,
        "missing_inputs": result.missing_inputs,
        "message": result.message,
        "failed_step": result.failed_step,
    }


def _serialize_outcome(o: RunOutcome) -> dict:
    d = o.decision
    return {
        "phase": o.phase,
        "router": {"mode": d.mode,
                   "capability": d.capability or d.proposed_name,
                   "args": d.args, "reason": d.reason},
        "missing_inputs": o.missing_inputs,
        "error": o.error,
        "evidence": o.evidence,
        "result": _serialize_result(o.result),
    }


@app.get("/capabilities")
def capabilities():
    """The callable catalog (typed tool specs) plus the operation allowlist."""
    return JSONResponse({
        "capabilities": CapabilityRegistry().tool_specs(),
        "allowed_operations": sorted(ALLOWED_OPERATIONS),
        "human_only_intents": sorted(HUMAN_ONLY_INTENTS),
    })


@app.post("/run")
def run(body: RunBody):
    if not CONFIG.has_llm:
        return JSONResponse(
            {"error": "OPENAI_API_KEY is not set; the router is model-driven and has "
                      "no offline fallback."}, status_code=503)
    outcome = run_request(body.request, allow_irreversible=body.allow_irreversible,
                          escalate=body.escalate)
    return JSONResponse(_serialize_outcome(outcome))


@app.post("/invoke/{name}")
def invoke(name: str, body: InvokeBody):
    """Deterministic replay by capability name -- no LLM needed for this path."""
    result, ev = invoke_capability(name, body.args,
                                   allow_irreversible=body.allow_irreversible,
                                   escalate=body.escalate)
    payload = _serialize_result(result) or {}
    payload["evidence"] = str(ev.dir)
    return JSONResponse(payload)


_CHAT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>CUA — request console</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:760px;margin:32px auto;color:#222}
 #log{border:1px solid #ddd;border-radius:8px;padding:12px;min-height:240px;
      background:#fafafa;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:13px}
 .row{display:flex;gap:8px;margin-top:12px}
 input{flex:1;padding:10px;border:1px solid #ccc;border-radius:8px}
 button{padding:10px 16px;border:0;border-radius:8px;background:#003366;color:#fff;cursor:pointer}
 .me{color:#003366}.sys{color:#444}.warn{color:#a06a00}.err{color:#b00020}.ok{color:#0a7a3a}
 small{color:#888}
</style></head><body>
<h3>Computer-Use Automation — request console</h3>
<small>Thin UI over <code>POST /run</code>. Non-allowed operations are routed to a
human; incomplete requests are asked back. This is a demo surface, not the product.</small>
<div id="log"></div>
<div class="row">
  <input id="q" placeholder="e.g. look up the savings balance for member 100001"
         autofocus onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()">Send</button>
</div>
<script>
const log=document.getElementById('log');
function line(cls,txt){const d=document.createElement('div');d.className=cls;
  d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function send(){
  const q=document.getElementById('q');const req=q.value.trim();if(!req)return;
  line('me','you › '+req);q.value='';
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({request:req})});
    const j=await r.json();
    if(j.error){line('err','system › '+j.error);return;}
    const cap=j.router&&j.router.capability;
    line('sys','router › '+j.router.mode+(cap?(' ('+cap+')'):'')+' — '+j.router.reason);
    if(j.phase==='escalated'){line('warn','↪ routed to a human operator — '+j.error);return;}
    if(j.phase==='needs_input'){line('warn','↪ more info needed — missing: '+
      (j.missing_inputs||[]).join(', '));return;}
    const res=j.result;
    if(res){
      const cls=(res.outcome==='success'||res.outcome==='business_outcome')?'ok':'warn';
      line(cls,'result › '+res.outcome+(res.business_code?(' ['+res.business_code+']'):'')
        +' '+(res.message||''));
      if(res.outputs&&Object.keys(res.outputs).length)
        line('ok','outputs › '+JSON.stringify(res.outputs));
    } else {line('sys','phase › '+j.phase+(j.error?(' — '+j.error):''));}
  }catch(e){line('err','system › '+e);}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_CHAT_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5003)
