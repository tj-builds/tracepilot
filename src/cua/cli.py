"""Command-line entry points -- the demo path.

    python -m cua.cli discover  --goal "..." --capability lookup_member_balance --member-id 100001
    python -m cua.cli replay    --capability lookup_member_balance --member-id 100001
    python -m cua.cli replay    --capability lookup_member_balance --member-id 000000   # business outcome
    python -m cua.cli replay    --capability lookup_member_balance --member-id 222222 --escalate
    python -m cua.cli catalog
    python -m cua.cli invoke    --name lookup_member_balance --json "{\"member_id\": \"100001\"}"
"""
from __future__ import annotations

import json

import typer
from rich import print as rprint
from rich.table import Table

from .agent.llm import build_client
from .catalog.registry import CapabilityRegistry
from .config import CONFIG
from .escalation.handoff import AutoOperatorHandler
from .evidence.logger import EvidenceLog
from .replay.engine import ReplayEngine
from .safety.policy import Policy
from .safety.operations import ALLOWED_OPERATIONS, HUMAN_ONLY_INTENTS
from .schema.artifact import Checkpoint, CheckpointType, Locator, VariantOverride
from .service import (
    creds,
    discover_capability,
    invoke_capability,
    missing_inputs,
    run_request,
)
from .surface.web import WebSurface

app = typer.Typer(add_completion=False, help="Computer-Use Automation System")


@app.command()
def discover(
    goal: str = typer.Option(..., help="Natural-language goal."),
    target: str = typer.Option(
        None, help="Target app base URL / entry point (defaults to TARGET_BASE_URL)."),
    capability: str = typer.Option(..., help="Name to save the capability under."),
    member_id: str = typer.Option(..., help="Member ID to use during discovery."),
    subaccount: bool = typer.Option(False, help="Also open a new sub-account."),
    account_type: str = typer.Option(None, help="Sub-account type (required with --subaccount)."),
    deposit: str = typer.Option(None, help="Initial deposit (required with --subaccount)."),
    no_validator: bool = typer.Option(
        False, help="Skip the LLM validator critic (~halves LLM calls; for tight quotas)."),
    run_id: str = typer.Option(None, help="Deterministic evidence dir name."),
):
    """Run an LLM-driven discovery run (planner + validator) and save a capability."""
    if not CONFIG.has_llm:
        rprint("[red]OPENAI_API_KEY is not set.[/red] Discovery is model-driven and "
               "has no offline fallback — set the key in .env and retry.")
        raise typer.Exit(code=1)
    if subaccount:
        if not account_type or not deposit:
            rprint("[red]--subaccount requires --account-type and --deposit[/red] "
                   "(no default values are assumed for a state-changing operation).")
            raise typer.Exit(code=1)
        cp = Checkpoint(type=CheckpointType.TEXT_PRESENT, expression="Sub-account created",
                        description="confirmation screen reached")
        description = "Open a new sub-account for a member and reach the confirmation screen."
        extra = {"account_type": account_type, "deposit": deposit}
        outputs_spec = [{"name": "confirmation_ref", "label": "Reference"}]
    else:
        cp = Checkpoint(type=CheckpointType.TEXT_PRESENT, expression="Savings balance",
                        description="member detail with savings balance visible")
        description = "Look up a member and read their current savings balance."
        extra = {}
        outputs_spec = [{"name": "savings_balance", "label": "Savings balance"}]

    if no_validator:
        rprint("[yellow]Validator disabled (--no-validator): ~half the LLM calls, "
               "but no critic layer on discovery actions.[/yellow]")

    try:
        res, ev = discover_capability(
            goal, capability, description, cp, outputs_spec,
            member_id=member_id, extra_params=extra, no_validator=no_validator,
            base_url=target, run_id=run_id)
    except Exception as exc:  # e.g. provider rate limit / auth error
        rprint(f"[red]Discovery aborted:[/red] {exc}")
        if "rate limit" in str(exc).lower() or "429" in str(exc):
            rprint("[yellow]OpenAI rejected the run on rate limits. Wait for the "
                   "quota to reset or raise the limit.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"[bold]Discovery status:[/bold] {res.status}")
    if res.outputs:
        rprint(f"[bold]Extracted (live):[/bold] {res.outputs}")
    rprint(f"[bold]Evidence:[/bold] {res.evidence_dir}")
    if res.status == "success" and res.artifact:
        rprint(f"[green]Saved capability -> artifacts/{res.artifact.name}.json[/green]")
        rprint(f"[dim]Steps: {len(res.artifact.steps)}, "
               f"params: {[p.name for p in res.artifact.params]}, "
               f"outputs: {[o.name for o in res.artifact.outputs]}[/dim]")
    else:
        rprint(f"[red]No artifact saved ({res.status}): {res.reason}[/red]")
        raise typer.Exit(code=1)


@app.command()
def replay(
    capability: str = typer.Option(None, help="Saved capability name."),
    artifact: str = typer.Option(None, help="Path to an artifact JSON (overrides name)."),
    member_id: str = typer.Option(..., help="Member ID to operate on (required)."),
    account_type: str = typer.Option(
        None, help="Sub-account type; required for capabilities that declare it."),
    deposit: str = typer.Option(
        None, help="Initial deposit; required for capabilities that declare it."),
    allow_irreversible: bool = typer.Option(False, help="Permit irreversible actions."),
    escalate: bool = typer.Option(False, help="Enable auto-operator escalation."),
    with_creds: bool = typer.Option(True, help="Supply login credentials at invocation."),
    tenant: str = typer.Option(None, help="Activate a per-tenant VariantOverride by id."),
    ignore_overrides: bool = typer.Option(
        False, help="With --tenant: drive the tenant but SKIP its overrides "
                    "(shows the override is load-bearing)."),
    inject: str = typer.Option(
        None, help="Arm a runtime fault before replay: session_timeout | app_error | "
                   "slow_load | persistent_notice."),
    run_id: str = typer.Option(None, help="Deterministic evidence dir name."),
):
    """Deterministically replay a saved capability (no LLM in the loop)."""
    CONFIG.ensure_dirs()
    reg = CapabilityRegistry()
    art = reg.load_path(artifact) if artifact else reg.load(capability)

    ev = EvidenceLog("replay", run_id=run_id)
    policy = Policy.from_allowlist(CONFIG.target_base_url, art.target.allowed_routes,
                                   allow_irreversible=allow_irreversible)
    # Build the caller's args explicitly from CLI flags. `member_id` has a visible
    # manual-tool default; account_type/deposit are only sent when provided.
    args = {"member_id": member_id}
    if account_type is not None:
        args["account_type"] = account_type
    if deposit is not None:
        args["deposit"] = deposit

    # Same completeness gate the agent-facing path uses: never fabricate a required
    # input from a recorded example. Missing -> stop and say what's needed.
    gaps = missing_inputs(art, args)
    if gaps:
        rprint(f"[yellow]Incomplete invocation: missing required input(s) "
               f"{', '.join(gaps)}.[/yellow] Pass them explicitly, e.g. "
               f"--account-type savings --deposit 100.00")
        raise typer.Exit(code=1)

    # Recorded examples may seed OPTIONAL params only; required inputs come from args.
    params = {p.name: p.example for p in art.params
              if p.example is not None and not p.sensitive and not p.required}
    for k, v in args.items():
        params[k] = str(v)
    if with_creds:
        params.update(creds())  # sensitive; redacted in logs, never persisted
    if tenant:
        params["_variant"] = tenant  # activate the per-tenant override

    escalation = None
    escalate_codes: set[str] = set()
    cdp_port = None
    if escalate:
        escalation = AutoOperatorHandler(CONFIG.target_base_url, headless=CONFIG.headless,
                                         **creds())
        # A compliance-review interstitial is a human judgment call -> escalate it.
        escalate_codes = {"interstitial_notice", "session_timeout"}
        # Expose CDP so a human operator can attach to the same live session.
        cdp_port = CONFIG.handoff_cdp_port

    surface = WebSurface(headless=CONFIG.headless, timeout_ms=CONFIG.step_timeout_ms,
                         cdp_port=cdp_port)
    surface.start()
    try:
        if inject:
            # Arm the fault on the live session BEFORE the policed run begins. This is
            # test setup, not part of the replay, so it does not pass the policy gate.
            surface.navigate(f"{CONFIG.target_base_url}/_simulate/inject/{inject}")
            ev.event("fault_injected", condition=inject)
        engine = ReplayEngine(surface, policy, ev, CONFIG, escalation=escalation,
                              escalate_codes=escalate_codes,
                              apply_variant_overrides=not ignore_overrides)
        result = engine.replay(art, params)
    finally:
        surface.stop()

    _print_result(result, ev)
    reg_path = ev.write_json("replay_result.json", result.model_dump())
    rprint(f"[dim]Result written -> {reg_path}[/dim]")
    if not result.is_terminal_success:
        raise typer.Exit(code=2)


@app.command("add-variant")
def add_variant(
    capability: str = typer.Option(..., help="Saved capability to specialize."),
    variant: str = typer.Option(..., help="Variant/tenant id, e.g. 'pioneer'."),
    base_url: str = typer.Option(..., help="Per-tenant base URL."),
    relabel: list[str] = typer.Option(
        None, "--relabel", help="Label remap 'Old=New' (repeatable). Rewrites only "
                                "the locators/checkpoint that reference the old label."),
    notes: str = typer.Option("", help="Free-text note on the variant."),
):
    """Attach a per-tenant override to an existing artifact WITHOUT re-recording.

    This is the multi-tenant reuse path: the same artifact serves a second tenant of
    the same vendor product; only the labels that differ are overridden.
    """
    reg = CapabilityRegistry()
    art = reg.load(capability)
    remaps = dict(r.split("=", 1) for r in (relabel or []) if "=" in r)

    locator_overrides: dict[int, Locator] = {}
    for step in art.steps:
        if step.locator is None:
            continue
        if _locator_mentions(step.locator, remaps):
            locator_overrides[step.index] = _relabel_locator(step.locator, remaps)

    cp_override = None
    expr = art.success_checkpoint.expression
    new_expr = _apply_remaps(expr, remaps)
    if new_expr != expr:
        cp_override = art.success_checkpoint.model_copy(update={"expression": new_expr})

    ov = VariantOverride(variant_id=variant, base_url=base_url,
                         locator_overrides=locator_overrides,
                         success_checkpoint=cp_override, notes=notes)
    art.variants = [v for v in art.variants if v.variant_id != variant] + [ov]
    path = reg.save(art)
    rprint(f"[green]Attached variant '{variant}' -> {path}[/green]")
    rprint(f"[dim]overrode locators on steps {sorted(locator_overrides)}; "
           f"checkpoint overridden: {cp_override is not None}[/dim]")


def _apply_remaps(text: str, remaps: dict) -> str:
    for old, new in remaps.items():
        text = text.replace(old, new)
    return text


def _locator_mentions(loc: Locator, remaps: dict) -> bool:
    hay = loc.description + " " + " ".join(c.value for c in loc.candidates)
    return any(old in hay for old in remaps)


def _relabel_locator(loc: Locator, remaps: dict) -> Locator:
    new = loc.model_copy(deep=True)
    new.description = _apply_remaps(new.description, remaps)
    for c in new.candidates:
        c.value = _apply_remaps(c.value, remaps)
    return new


@app.command("ping-llm")
def ping_llm():
    """Verify the configured LLM client is reachable (one real call, no browser).

    Handy before spending discovery calls: confirms your key/model resolve and the
    provider actually answers. Prints which client was selected.
    """
    from .agent.llm import build_client

    try:
        client = build_client(CONFIG)
    except RuntimeError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    rprint(f"[bold]client:[/bold] {client.name}   "
           f"[bold]model:[/bold] {CONFIG.model_label}")
    try:
        reply = client.complete(
            "You reply with a single JSON object and nothing else.",
            "Return the JSON object with a single key ok set to true.")
        rprint(f"[green]OK[/green] — raw reply: {reply}")
    except Exception as exc:
        rprint(f"[red]LLM call failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def stability(
    capability: str = typer.Option(..., help="Saved capability to benchmark."),
    member_id: str = typer.Option(..., help="Member ID to benchmark against (required)."),
    runs: int = typer.Option(5, help="How many deterministic replays to run."),
    tenant: str = typer.Option(None, help="Benchmark a specific tenant variant."),
    approve: bool = typer.Option(
        False, help="Promote draft->approved if the pass rate is 100%."),
    run_id: str = typer.Option(None),
):
    """Replay a capability N times and report a stability/flakiness scorecard.

    Writes the measured `stability_score` (pass fraction) back onto the artifact and,
    with --approve, gates the draft->approved lifecycle on a perfect score. This is
    the confidence signal an unattended caller should consult before trusting a
    capability.
    """
    import statistics
    import time as _time

    CONFIG.ensure_dirs()
    reg = CapabilityRegistry()
    art = reg.load(capability)
    ev = EvidenceLog("stability", run_id=run_id)
    policy = Policy.from_allowlist(CONFIG.target_base_url, art.target.allowed_routes)

    outcomes, latencies, fallback_runs = [], [], 0
    for i in range(runs):
        params = _replay_params(art, member_id)
        if tenant:
            params["_variant"] = tenant
        surface = WebSurface(headless=CONFIG.headless, timeout_ms=CONFIG.step_timeout_ms)
        surface.start()
        try:
            engine = ReplayEngine(surface, policy, ev, CONFIG)
            t0 = _time.time()
            result = engine.replay(art, params)
            latencies.append(int((_time.time() - t0) * 1000))
        finally:
            surface.stop()
        outcomes.append(result.outcome.value)
        if any(s.used_fallback for s in result.steps):
            fallback_runs += 1

    passes = sum(1 for o in outcomes if o == "success")
    score = round(passes / runs, 3) if runs else 0.0
    art.stability_score = score
    if approve and score == 1.0:
        art.approval_state = "approved"
    reg.save(art)

    scorecard = {
        "capability": capability, "runs": runs, "tenant": tenant,
        "pass_rate": score, "outcomes": outcomes,
        "latency_ms": {"min": min(latencies), "median": int(statistics.median(latencies)),
                       "max": max(latencies)} if latencies else {},
        "runs_using_fallback": fallback_runs,
        "approval_state": art.approval_state,
    }
    ev.write_json("stability.json", scorecard)
    ev.finish("stability", **scorecard)

    table = Table(title=f"Stability scorecard: {capability}"
                        + (f" [{tenant}]" if tenant else ""))
    table.add_column("metric"); table.add_column("value")
    table.add_row("runs", str(runs))
    table.add_row("pass_rate (stability_score)", f"{score:.3f}")
    table.add_row("outcomes", ", ".join(sorted(set(outcomes))))
    if latencies:
        table.add_row("latency ms (min/median/max)",
                      f"{min(latencies)}/{int(statistics.median(latencies))}/{max(latencies)}")
    table.add_row("runs using a fallback locator", str(fallback_runs))
    table.add_row("approval_state", art.approval_state)
    rprint(table)
    rprint(f"[dim]Scorecard -> {ev.dir}[/dim]")


def _replay_params(art, member_id: str) -> dict:
    # Recorded examples seed OPTIONAL params only; required inputs are never fabricated.
    params = {p.name: p.example for p in art.params
              if p.example is not None and not p.sensitive and not p.required}
    params["member_id"] = member_id
    params.update(creds())
    return params


@app.command()
def catalog():
    """List saved capabilities as agent-invocable tool specs."""
    reg = CapabilityRegistry()
    specs = reg.tool_specs()
    if not specs:
        rprint("[yellow]No capabilities saved yet. Run 'discover' first.[/yellow]")
        return
    table = Table(title="Capability catalog (agent-invocable tools)")
    table.add_column("name"); table.add_column("approval")
    table.add_column("params"); table.add_column("returns")
    for s in specs:
        table.add_row(s["name"], s["approval_state"],
                      ", ".join(s["parameters"]["properties"].keys()),
                      ", ".join(s["returns"].keys()))
    rprint(table)
    rprint("\n[dim]Tool JSON schema:[/dim]")
    rprint(json.dumps(specs, indent=2))


@app.command()
def operations():
    """Show the operation allowlist: what the system may do autonomously vs what is
    routed to a human in the loop (the semantic safety layer above route/risk gating).
    """
    table = Table(title="Operation policy (autonomous vs human-in-the-loop)")
    table.add_column("operation / intent")
    table.add_column("disposition")
    for op in sorted(ALLOWED_OPERATIONS):
        table.add_row(op, "[green]ALLOW — autonomous[/green]")
    for kw, desc in sorted(HUMAN_ONLY_INTENTS.items()):
        table.add_row(f"{kw}  ({desc})", "[yellow]ESCALATE — human only[/yellow]")
    table.add_row("[dim]* any other / unknown operation[/dim]",
                  "[yellow]ESCALATE — human authorization[/yellow]")
    rprint(table)
    rprint("[dim]Non-allowed requests are not executed; they raise a human "
           "authorization request (see the operator console).[/dim]")


@app.command()
def invoke(
    name: str = typer.Option(..., help="Capability name to invoke."),
    json_args: str = typer.Option("{}", "--json", help="JSON object of input params."),
    allow_irreversible: bool = typer.Option(False),
    escalate: bool = typer.Option(False),
):
    """Invoke a capability by name with typed args (agent-facing entry point)."""
    args = json.loads(json_args)  # ALL typed args are forwarded, not just member_id
    result, ev = invoke_capability(name, args, allow_irreversible=allow_irreversible,
                                   escalate=escalate)
    _print_result(result, ev)
    if not result.is_terminal_success:
        raise typer.Exit(code=2)


@app.command()
def run(
    request: str = typer.Argument(..., help="A natural-language request."),
    no_autocreate: bool = typer.Option(
        False, help="If no capability matches, do NOT discover one; just report."),
    allow_irreversible: bool = typer.Option(False),
    escalate: bool = typer.Option(False),
):
    """Agentic entrypoint: understand a request, then reuse or CREATE the workflow.

    The router decides — by meaning, not wording — whether an existing capability
    accomplishes the request (INVOKE it) or whether none fits (DISCOVER a new one,
    then invoke it). This is the "does the workflow exist, or must an agent build it?"
    decision the assignment's agent-facing layer would make.
    """
    if not CONFIG.has_llm:
        rprint("[red]OPENAI_API_KEY is not set.[/red] The agentic router is "
               "model-driven and has no offline fallback — set the key in .env.")
        raise typer.Exit(code=1)
    outcome = run_request(request, autocreate=not no_autocreate,
                          allow_irreversible=allow_irreversible, escalate=escalate)
    d = outcome.decision
    rprint(f"[bold]Router:[/bold] mode=[cyan]{d.mode}[/cyan] "
           f"capability={d.capability or d.proposed_name} args={d.args}")
    rprint(f"[dim]reason: {d.reason}[/dim]")
    rprint(f"[bold]Phase:[/bold] {outcome.phase}")

    # Incomplete request -> a clarification, not a failure. Say what's missing and stop
    # WITHOUT touching the surface.
    if outcome.phase == "needs_input":
        miss = ", ".join(outcome.missing_inputs) or "required input(s)"
        rprint(f"[bold yellow]More information needed[/bold yellow] — missing: {miss}")
        if outcome.error:
            rprint(f"[yellow]{outcome.error}[/yellow]")
        if outcome.result is not None:
            _print_result(outcome.result, _EvDir(outcome.evidence))
        raise typer.Exit(code=1)

    # Operation not permitted autonomously -> handed to a human in the loop.
    if outcome.phase == "escalated":
        rprint(f"[bold yellow]Routed to a human operator[/bold yellow] — "
               f"{outcome.error}")
        rprint(f"[dim]authorization request: {outcome.evidence}[/dim]")
        raise typer.Exit(code=3)

    if outcome.phase == "error":
        rprint(f"[red]{outcome.error}[/red]")
        raise typer.Exit(code=1)

    if outcome.error:
        rprint(f"[red]{outcome.error}[/red]")
    if outcome.result is not None:
        _print_result(outcome.result, _EvDir(outcome.evidence))
        if not outcome.result.is_terminal_success:
            raise typer.Exit(code=2)
    elif outcome.phase in ("would_create", "create_failed"):
        raise typer.Exit(code=1)


class _EvDir:
    """Tiny shim so _print_result can show the evidence dir from a RunOutcome."""
    def __init__(self, d: str):
        self.dir = d


def _print_result(result, ev):
    color = {"success": "green", "business_outcome": "cyan", "recovered": "cyan",
             "escalated": "yellow", "blocked_by_policy": "yellow",
             "recovery_exhausted": "yellow", "needs_input": "yellow",
             "hard_failure": "red"}.get(result.outcome.value, "white")
    rprint(f"[bold {color}]Outcome: {result.outcome.value}[/bold {color}]")
    if result.business_code:
        rprint(f"  business_code: {result.business_code}")
    if result.missing_inputs:
        rprint(f"  missing_inputs: {', '.join(result.missing_inputs)}")
    if result.message:
        rprint(f"  message: {result.message}")
    if result.outputs:
        rprint(f"  outputs: {result.outputs}")
    if result.evidence.get("drift"):
        rprint(f"  [yellow]drift: {result.evidence['drift']}[/yellow]")
    if result.failed_step is not None:
        rprint(f"  failed_step: {result.failed_step} "
               f"(expected: {result.expected}, observed: {result.observed})")
    rprint(f"  evidence: {ev.dir}")


if __name__ == "__main__":
    app()
