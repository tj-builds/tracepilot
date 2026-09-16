# Computer-Use Automation System

An AI agent discovers how to complete a goal inside a legacy, API-less back-office
application by driving its UI. The successful run is captured as a **typed, versioned
capability artifact**, which is then **replayed deterministically** — no LLM in the
loop — with typed inputs/outputs, a runtime-error taxonomy, layered safety guardrails,
and a human-in-the-loop escalation that takes over the *same* live browser session.

> The model discovers. The artifact becomes a reusable capability.
> Deterministic replay is how the agent invokes it in production.

The full design rationale and trade-offs are in **[REPORT.md](REPORT.md)**.

## Table of contents

- [How it works](#how-it-works)
- [Target application](#target-application)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Safety model](#safety-model)
- [Human-in-the-loop handoff](#human-in-the-loop-handoff)
- [Agent-facing API](#agent-facing-api)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Data & persistence](#data--persistence)
- [Evidence & artifacts](#evidence--artifacts)
- [Security](#security)

## How it works

```
goal ─▶ LLM-driven discovery (observe → decide → validate → act on a live UI)
     ─▶ saved capability artifact (artifacts/*.json)
     ─▶ deterministic replay (typed params → typed outputs, error taxonomy)
     ─▶ human escalation + same-session handoff when stuck
     ─▶ redacted evidence for every run (evidence/*)
```

During discovery, every action the planner ("actor") proposes is checked by a
**validator** ("critic") before it runs — a two-layer guardrail (deterministic
allowlist + an LLM critic) that can veto an off-goal, out-of-scope, or unexpectedly
irreversible step. Replay is intentionally LLM-free; its guardrail is the deterministic
policy alone.

## Target application

The proxy target is a local, deliberately **legacy-styled credit-union servicing
console** (`mock_app/app.py`): table-based non-semantic markup, no test IDs — but real
`<label>`/button text, so accessibility-tree targeting works. It injects the runtime
conditions the system must handle:

| Member ID | Behaviour | Replay classification |
|-----------|-----------|-----------------------|
| `100001`, `100002` | normal member | `success` |
| `000000` / unknown | "No such member" | business outcome (`member_not_found`) |
| `999999` | "Permission denied" | business outcome (`permission_denied`) |
| `222222` | interstitial "compliance review" notice | recoverable **or** escalated |

Faults are reproducible on demand via `--inject` (backed by `/_simulate/inject/<cond>`):

| `--inject` | Simulates | Replay classification |
|------------|-----------|-----------------------|
| `session_timeout` | session expires at the detail screen | session timeout → escalate/re-auth |
| `app_error` | 500 "internal error" page | `hard_failure` |
| `slow_load` | transient slowness | absorbed by bounded retry |
| `persistent_notice` | an interstitial that never clears | `recovery_exhausted` |

A second tenant (`?tenant=pioneer`) runs the *same* product with a relabelled balance
row ("Savings balance" → "Ledger balance") plus a decoy — the multi-tenant reuse case.

## Setup

Requires **Python 3.11+** and a one-time Playwright browser download.

```bash
pip install -r requirements.txt
python -m playwright install chromium
pip install -e .            # makes `python -m cua.cli` importable
cp .env.example .env        # required — then set OPENAI_API_KEY inside
```

Configuration is read **strictly** from `.env` with no hardcoded defaults, so copying
`.env.example` to `.env` is required before running anything (including the tests); a
missing variable fails fast with a clear message.

## Configuration

- **OpenAI is the only provider and is required** for discovery, routing, and
  validation. Set `OPENAI_API_KEY` (and `OPENAI_MODEL`). There is **no offline
  fallback**: the model-driven paths refuse to run without a key rather than silently
  faking a model. Verify connectivity with `python -m cua.cli ping-llm`.
- **Replay never uses an LLM or any key** — that is the point. Its guardrail is the
  deterministic allowlist/risk policy.
- **Tests** inject their own `LLMClient` double (`tests/fakes.py`) and need no key.

All variables are documented in [`.env.example`](.env.example).

## Usage

Start the target app in one terminal:

```bash
python mock_app/app.py          # serves http://127.0.0.1:5001
```

**Discover a capability** (LLM-driven; add `--no-validator` to roughly halve the
model calls under a tight rate limit):

```bash
python -m cua.cli discover \
  --goal "look up member 100001 and read their current savings balance" \
  --capability lookup_member_balance --member-id 100001
```

**Replay deterministically** (no LLM):

```bash
python -m cua.cli replay --capability lookup_member_balance --member-id 100001
#   → success   outputs: {"savings_balance": "$1234.56"}

python -m cua.cli replay --capability lookup_member_balance --member-id 000000
#   → business_outcome   business_code: member_not_found   (a valid answer, not a crash)

python -m cua.cli replay --capability lookup_member_balance --member-id 100001 --inject app_error
#   → hard_failure   (fault surfaced with step/expected/observed + screenshot)
```

**Agent-facing catalog and invoke-by-name** (typed args, the production path):

```bash
python -m cua.cli catalog
python -m cua.cli invoke --name lookup_member_balance --json "{\"member_id\": \"100002\"}"
```

**Natural-language routing** — reuse an existing capability or refuse an incomplete
request rather than guessing:

```bash
python -m cua.cli run "look up the savings balance for member 100002"
#   → router invokes lookup_member_balance, replays it, returns the balance

python -m cua.cli run "open a savings account for customer alicia with 100 dollars"
#   → needs_input   missing: member_id   ("alicia" is a name; inputs are never fabricated)
```

**A state-changing capability** records an irreversible step, so replay is gated by
default:

```bash
python -m cua.cli discover \
  --goal "open a new sub-account for member 100001 and reach the confirmation screen" \
  --capability open_subaccount --member-id 100001 --subaccount \
  --account-type savings --deposit 100.00

python -m cua.cli replay --capability open_subaccount --member-id 100001 \
  --account-type savings --deposit 100.00
#   → blocked_by_policy   (irreversible action blocked by default)

python -m cua.cli replay --capability open_subaccount --member-id 100001 \
  --account-type savings --deposit 100.00 --allow-irreversible
#   → success   outputs: {"confirmation_ref": "SA-100001-..."}
```

**Multi-tenant reuse** — specialise one artifact for a second tenant without
re-recording:

```bash
python -m cua.cli add-variant --capability lookup_member_balance --variant pioneer \
  --base-url "http://127.0.0.1:5001/?tenant=pioneer" --relabel "Savings balance=Ledger balance"

python -m cua.cli replay --capability lookup_member_balance --member-id 100001 --tenant pioneer
#   → success   (reads the relabelled "Ledger balance" cell)
python -m cua.cli replay --capability lookup_member_balance --member-id 100001 --tenant pioneer --ignore-overrides
#   → hard_failure   (proves the override is load-bearing, not decoration)
```

**Stability scorecard** — replay N times, record a flakiness signal, and gate
`draft → approved`:

```bash
python -m cua.cli stability --capability lookup_member_balance --member-id 100001 --runs 5 --approve
```

## Safety model

Guardrails are layered; the full model and its limits are in
[REPORT.md §6](REPORT.md). Highlights:

- **Operation allowlist** (`python -m cua.cli operations`) — a semantic layer that
  decides, *before any execution*, whether an operation may run autonomously or must go
  to a human:

  | Operation / intent | Disposition |
  |--------------------|-------------|
  | `lookup_member_balance`, `open_subaccount` | allow — autonomous (still risk-gated at replay) |
  | `transfer` / `wire` / `withdraw` / `close` / `delete` / `freeze` … | escalate — human only, always |
  | any unknown / new operation | escalate — human authorization |

- **Route/host allowlist** — everything not explicitly permitted is denied.
- **Action-risk gating** — irreversible actions are blocked on unattended replay unless
  `--allow-irreversible` is passed.
- **Input completeness** — required inputs are never fabricated from recorded examples
  or defaults; a missing one returns `needs_input`.
- **Redaction & no-persist** — credentials are sensitive params, redacted from logs and
  never written into artifacts.

## Human-in-the-loop handoff

When escalation is enabled (`--escalate`), the browser is launched with a **CDP
remote-debugging port** (`HANDOFF_CDP_PORT`, default 9222). On a handoff the automation
pauses and exposes that live session; the queued intervention request
(`evidence/_escalations/`) carries a **CDP endpoint** an operator attaches to:

1. Chrome → `chrome://inspect` → add `localhost:9222` under "Discover network targets".
2. Drive the **same** session (same cookies/auth/state — not a fresh one).
3. Signal resume; automation takes control back and continues.

```bash
python -m cua.cli replay --capability lookup_member_balance --member-id 222222 --escalate
python -m cua.escalation.operator_app     # optional mock console at http://127.0.0.1:5002
```

The control-transfer mechanism (CDP attach + policed `HandoffControl` with one-shot
grants) is real; only the polished operator console UI is mocked. The unattended demo
uses a scripted stand-in operator so the full loop runs without a person.

## Agent-facing API

A thin HTTP surface over the same service functions the CLI uses (the stretch-goal
capability interface):

```bash
python -m cua.api                  # http://127.0.0.1:5003
#   GET  /capabilities   → callable catalog + operation allowlist
#   POST /run            → route a natural-language request
#   POST /invoke/{name}  → deterministic replay by name with typed args
#   GET  /               → a minimal chat page over /run
```

It enforces the same allowlist, completeness gate, and risk policy as the CLI.

## Tests

```bash
python -m pytest -q
```

Covers the load-bearing, browser-free logic: artifact round-trip and contract
validation, locator-candidate ordering, the allowlist and irreversible gating,
redaction, the error taxonomy, checkpoint verification, template resolution, the
operation allowlist, the input-completeness gate, one-shot authorization grants,
multi-tenant variant resolution, and bounded recovery. Browser-driven behaviour is
exercised by the runs in `evidence/`.

## Project structure

```
mock_app/app.py                 legacy-styled target application (the proxy)
mock_app/data/                  CSV-backed member records + change log
src/cua/
  schema/artifact.py            capability artifact schema (the contract)
  schema/results.py             replay result + outcome taxonomy
  surface/base.py               Surface abstraction (perceive/act seam)
  surface/web.py                Playwright web surface + same-session handoff
  agent/loop.py                 observe → decide → validate → act discovery loop
  agent/llm.py                  LLMClient interface + OpenAI provider
  agent/planner.py              the LLM planner ("actor")
  agent/validator.py            the action validator ("critic")
  agent/recorder.py             synthesises robust locators + the artifact
  replay/engine.py              deterministic replay (production path)
  replay/errors.py              runtime error / exceptional-state taxonomy
  safety/policy.py              route allowlist, action-risk gating, redaction
  safety/operations.py          operation allowlist (autonomous vs human)
  escalation/handoff.py         stuck detection + same-session control transfer
  escalation/operator_app.py    mock operator console
  catalog/registry.py           agent-invocable capability catalog
  cli.py                        command-line entry points
  api.py                        optional HTTP surface + chat page
artifacts/                      saved capability artifacts (deliverable)
evidence/                       discovery + replay logs & screenshots (deliverable)
tests/                          unit tests
```

## Data & persistence

The mock app is data-driven; the seed data is data, not code:

```
mock_app/data/members.csv      member records (member_id, name, savings, status, sub_accounts)
mock_app/data/changes.csv      append-only audit log of every mutation
```

- **Reads change nothing** — a balance lookup writes to neither file.
- **A successful sub-account open persists to both files** — an audit row is appended
  to `changes.csv`, and the member's `sub_accounts` count is incremented in
  `members.csv`. `savings` is left unchanged so the read-only balance lookup stays
  deterministic.

## Evidence & artifacts

- **`artifacts/`** holds saved capabilities (`<name>.json`) produced by real
  LLM-driven discovery — typed params/outputs, ordered locator candidates, a success
  checkpoint, provenance, and per-tenant variants.
- **`evidence/`** holds per-run, redacted logs — each run directory has a `run.jsonl`
  event log, a `replay_result.json` (for replays), and failure/escalation screenshots.
  The committed set spans the full slice: discovery, success, both business outcomes,
  escalation handoff, an injected hard failure, exhausted recovery, and a stability
  scorecard.

Discovery is the only step that spends model calls; every replay is deterministic and
free. Committed provenance shows the served model id (`gpt-5.6-luna`, via an
OpenAI-compatible gateway); the code path is the standard OpenAI client.

## Security

The mock app accepts any non-empty credentials and contains no real data. Login
credentials are supplied at invocation as sensitive parameters — redacted from all logs
and never persisted into artifacts. Never point this at a real system or use real
credentials or PII.
