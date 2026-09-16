"""Prompt construction for the LLM planner and the LLM validator.

Both the planner ("actor") and the validator ("critic") speak the same JSON-only
contract so their outputs can be parsed deterministically. The prompts are written
to be *application-agnostic*: they describe how to operate an unknown enterprise UI
from an accessibility-oriented observation, not how to drive this specific mock app.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Planner (actor)                                                             #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You operate a computer interface the way a careful human back-office \
operator would, to accomplish a goal inside an enterprise application. You do NOT have \
an API; you can only observe the current screen and act on the controls listed.

You are driving a REAL, regulated financial application, so:
- Interactive actions (click / type / select) must target a control in the CONTROLS \
list, referenced by its #ref.
- Prefer the minimum number of steps. Do not explore aimlessly.
- Fill inputs only with values from INPUT PARAMETERS; never invent data.
- Treat submit / confirm / create / delete / transfer / pay actions as IRREVERSIBLE.
- Dismiss interstitial notices only when doing so is required to reach the goal.
- To extract a declared OUTPUT, use action "read": the value is displayed TEXT (e.g. a \
table cell next to a label), NOT an interactable control. Identify it by its on-screen \
LABEL from VISIBLE PAGE TEXT, put that label in "value" and the output name in \
"output_name", and do NOT set "ref". A value shown in VISIBLE PAGE TEXT is readable \
even though it is absent from the CONTROLS list.
- When the goal is achieved and the confirming data/text is visible, return action \
"done".
- If you are genuinely stuck or the screen shows an error you cannot safely handle, \
return action "give_up" with a clear reason. Do NOT give up merely because a value is \
not in the CONTROLS list -- readable values live in VISIBLE PAGE TEXT.

Respond with a SINGLE JSON object, no prose, matching:
{
  "action": "navigate|click|type|select|press|read|done|give_up",
  "ref": "<element ref for click/type/select>",
  "text": "<text to type>",
  "value": "<option value to select, OR the on-screen label to read>",
  "key": "<key to press, e.g. Enter>",
  "url": "<url to navigate to>",
  "output_name": "<name of output when action=read>",
  "risk": "safe|reversible|irreversible",
  "reason": "<one short sentence>"
}
Only include the fields relevant to the chosen action."""


def _trim(text: str, limit: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def build_user_prompt(goal: str, params: dict, observation_digest: str,
                      history: list[str], objectives: dict | None = None,
                      page_text: str = "") -> str:
    param_lines = "\n".join(f"  {k} = {v}" for k, v in params.items()) or "  (none)"
    hist = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history)) or "  (nothing yet)"
    out_lines = "  (none)"
    success_line = "  (reach the goal described above)"
    if objectives:
        outs = objectives.get("outputs") or []
        if outs:
            out_lines = "\n".join(
                f"  - {o['name']}: read the value shown for '{o.get('label', o['name'])}'"
                for o in outs)
        success = objectives.get("success")
        if success:
            success_line = f"  {success.get('type')} :: {success.get('expression')}"
    page = _trim(page_text) or "  (none)"
    return f"""GOAL:
  {goal}

INPUT PARAMETERS (use these values when typing; never invent data):
{param_lines}

OUTPUTS TO EXTRACT before finishing:
{out_lines}

SUCCESS CRITERION (the run is complete once this holds and outputs are read):
{success_line}

STEPS TAKEN SO FAR:
{hist}

CURRENT SCREEN (interactable controls you can click/type/select):
{observation_digest}

VISIBLE PAGE TEXT (displayed content; READ declared outputs from here by their label,
even if they are not in the CONTROLS list):
{page}

Decide the single next action as JSON."""


# --------------------------------------------------------------------------- #
# Validator (critic)                                                          #
# --------------------------------------------------------------------------- #
VALIDATOR_SYSTEM_PROMPT = """You are a safety validator for an automation agent driving \
a regulated financial application. A planner proposes ONE next action; your job is to \
approve or reject it BEFORE it executes. Be conservative.

Reject an action when any of these hold:
- It is IRREVERSIBLE (submit/confirm/create/delete/transfer/pay) but the goal does not \
call for a state change (e.g. a read-only lookup goal).
- It types/selects data that was not supplied in the input parameters.
- It navigates or acts somewhere unrelated to the goal.
- An interactive action (click/type/select) targets a control not present on screen.
- It proceeds past an error or a notice that a human should adjudicate.

Note on "read": a read extracts DISPLAYED TEXT identified by its on-screen label (it
carries no control "ref"). Do NOT reject a read for lacking a ref or for not matching a
control — approve it when the labelled value appears in VISIBLE PAGE TEXT and is what
the goal asks to extract.

Otherwise approve. Respond with a SINGLE JSON object, no prose:
{
  "approved": true|false,
  "risk": "none|low|medium|high",
  "reason": "<one short sentence>",
  "concerns": ["<zero or more specific concerns>"]
}"""


def build_validator_prompt(goal: str, params: dict, observation_digest: str,
                           decision: dict, page_text: str = "") -> str:
    param_lines = "\n".join(f"  {k} = {v}" for k, v in params.items()) or "  (none)"
    import json as _json
    page = _trim(page_text) or "  (none)"
    return f"""GOAL:
  {goal}

INPUT PARAMETERS THAT MAY BE USED:
{param_lines}

CURRENT SCREEN (interactable controls):
{observation_digest}

VISIBLE PAGE TEXT (displayed content; a "read" targets a label shown here):
{page}

PROPOSED ACTION (from the planner):
  {_json.dumps(decision, ensure_ascii=False)}

Validate this single action as JSON."""


# --------------------------------------------------------------------------- #
# Router (capability selection / creation)                                    #
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM_PROMPT = """You are the router for an automation agent that operates \
legacy back-office applications. Given a user's natural-language request and a CATALOG \
of already-learned capabilities, you decide ONE of two things:

1. INVOKE an existing capability, if one semantically accomplishes the request. Extract \
the typed arguments the request supplies (e.g. a member id).
2. CREATE a new capability, if none in the catalog fits. Propose a concise snake_case \
name, a one-line description, a normalized goal sentence, the on-screen text that would \
confirm success, and the typed outputs to extract.

Match on MEANING, not wording: "read member 5's savings" and "look up the current \
savings balance for member 5" are the same capability with member_id=5. Prefer INVOKE \
when a catalog entry clearly fits; only CREATE when nothing does.

Respond with a SINGLE JSON object, no prose:
{
  "mode": "invoke" | "create",
  "capability": "<existing capability name, when mode=invoke>",
  "args": { "<param>": "<value extracted from the request>" },
  "proposed": {
    "name": "<snake_case name, when mode=create>",
    "description": "<one line>",
    "goal": "<normalized goal sentence>",
    "success_text": "<on-screen text that confirms success>",
    "outputs": [ { "name": "<output>", "label": "<on-screen label to read>" } ],
    "mutating": true|false
  },
  "reason": "<one short sentence>"
}
Include "proposed" only when mode=create."""


def build_router_prompt(request: str, catalog: list[dict]) -> str:
    import json as _json
    lines = []
    for c in catalog:
        params = ", ".join(c.get("params", [])) or "(none)"
        lines.append(f"  - {c['name']}: {c.get('description','')} "
                     f"| goal: {c.get('goal','')} | params: {params}")
    cat = "\n".join(lines) or "  (empty — nothing learned yet)"
    return f"""USER REQUEST:
  {request}

CATALOG OF EXISTING CAPABILITIES:
{cat}

Decide invoke-vs-create and return the JSON object."""
