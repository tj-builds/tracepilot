"""A deliberately *legacy-styled* credit-union back-office console.

This stands in for the real thing. It intentionally exhibits the properties the
brief calls out:

  * Non-semantic, table-based markup, messy class names, no test IDs.
  * But real <label> associations + button text, so accessibility-tree / role /
    label targeting works -- which is exactly the robust-locator lesson.
  * Injectable runtime error & exceptional states so replay error handling can be
    demonstrated:
        - member 000000 / unknown  -> "No such member" (business outcome)
        - member 999999            -> permission denied (hard-ish / policy)
        - member 222222            -> interstitial "Notice" dialog (recoverable)
        - GET /_simulate/expire    -> clears the session (session timeout)
        - ?slow=1 on detail        -> transient slow load

Flow: login -> search -> member detail (read savings balance)
                      -> open new sub-account (multi-field form) -> confirmation.

Data is a structured dataset, not code: member records are seeded from
`data/members.csv`. A successful mutation (opening a sub-account) is persisted to BOTH
datasets -- appended to `data/changes.csv` (an append-only audit log) AND written back
to `data/members.csv` (the member's `sub_accounts` count is incremented) so the change
is reflected in queryable member state. Read-only queries (a balance lookup) correctly
change nothing.
"""
from __future__ import annotations

import csv
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = "mock-app-not-a-secret"  # local demo only; no real data

# --------------------------------------------------------------------------- #
# Structured dataset (CSV-backed)                                             #
# --------------------------------------------------------------------------- #
# Member records live in a CSV so the seed data is data, not code: edit
# data/members.csv to add/adjust members. Every mutation the automation performs
# (currently: opening a sub-account) is appended to data/changes.csv, giving a
# structured, auditable log of "existing info + all changes" in one place.
DATA_DIR = Path(__file__).resolve().parent / "data"
MEMBERS_CSV = DATA_DIR / "members.csv"
CHANGES_CSV = DATA_DIR / "changes.csv"
_CHANGES_HEADER = ["timestamp", "change_type", "member_id", "account_type",
                   "deposit", "reference"]
_MEMBERS_HEADER = ["member_id", "name", "savings", "status", "sub_accounts"]

_DEFAULT_MEMBERS = {
    "100001": {"name": "Alice Nguyen", "savings": "1234.56", "status": "active",
               "sub_accounts": "0"},
    "100002": {"name": "Bob Carter", "savings": "42.00", "status": "active",
               "sub_accounts": "0"},
    "222222": {"name": "Carol Diaz", "savings": "500.00", "status": "active",
               "sub_accounts": "0"},
    "999999": {"name": "Restricted Member", "savings": "0.00", "status": "restricted",
               "sub_accounts": "0"},
}


def load_members() -> dict:
    """Load member records from CSV; fall back to built-in defaults if absent.

    `sub_accounts` is optional in the seed file (defaults to 0); it is the field a
    successful sub-account creation increments and persists back (see `save_members`).
    """
    if not MEMBERS_CSV.exists():
        return dict(_DEFAULT_MEMBERS)
    out: dict = {}
    with MEMBERS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = (row.get("member_id") or "").strip()
            if not mid:
                continue
            out[mid] = {"name": (row.get("name") or "").strip(),
                        "savings": (row.get("savings") or "0.00").strip(),
                        "status": (row.get("status") or "active").strip(),
                        "sub_accounts": (row.get("sub_accounts") or "0").strip()}
    return out or dict(_DEFAULT_MEMBERS)


def save_members() -> None:
    """Persist the in-memory member records back to members.csv.

    Called after a mutation so a state change is reflected in the *queryable* member
    dataset (not just the changes log). We deliberately do NOT touch `savings` on a
    sub-account open, so the read-only balance lookup stays deterministic; the visible,
    persisted change is the `sub_accounts` count.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MEMBERS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_MEMBERS_HEADER)
        for mid, m in MEMBERS.items():
            w.writerow([mid, m.get("name", ""), m.get("savings", "0.00"),
                        m.get("status", "active"), m.get("sub_accounts", "0")])


def record_change(change_type: str, member_id: str, account_type: str = "",
                  deposit: str = "", reference: str = "") -> None:
    """Append one mutation to the changes dataset (auditable, append-only)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CHANGES_CSV.exists() or CHANGES_CSV.stat().st_size == 0
    with CHANGES_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(_CHANGES_HEADER)
        w.writerow([datetime.now(timezone.utc).isoformat(), change_type, member_id,
                    account_type, deposit, reference])


MEMBERS = load_members()

# Two tenants running the SAME vendor product, configured/branded differently.
# `harbor` is the base app discovery records against; `pioneer` is a second tenant
# that relabels the savings row ("Savings balance" -> "Ledger balance") and adds a
# decoy "Available balance" row -- so a locator that isn't anchored on the right
# human label would read the WRONG cell. This is the multi-tenant reuse case: one
# artifact + a per-variant override reads both, and without the override it fails
# rather than returning a wrong balance.
TENANTS = {
    "harbor": {"brand": "CoreCU", "savings_label": "Savings balance", "decoy": None},
    "pioneer": {"brand": "Pioneer CU", "savings_label": "Ledger balance",
                "decoy": ("Available balance", "0.00")},
}


def current_tenant() -> str:
    t = (request.args.get("tenant") or "").strip().lower()
    if t in TENANTS:
        session["tenant"] = t
    return session.get("tenant", "harbor")


def tenant_cfg() -> dict:
    return TENANTS.get(session.get("tenant", "harbor"), TENANTS["harbor"])

PAGE = """
<!doctype html><html><head><title>CoreCU Servicing</title></head>
<body bgcolor="#f4f4f4" style="font-family:Verdana,Arial,sans-serif;font-size:12px">
<table width="900" align="center" cellpadding="0" cellspacing="0" border="0"
       bgcolor="#ffffff" style="border:1px solid #cccccc">
 <tr><td bgcolor="#003366" style="padding:8px;color:#ffffff;font-weight:bold">
   CoreCU &raquo; Member Servicing Console
   {% if session.get('user') %}
     &nbsp;|&nbsp; <font color="#cccccc">operator: {{ session['user'] }}</font>
   {% endif %}
 </td></tr>
 <tr><td style="padding:16px">{{ body|safe }}</td></tr>
 <tr><td bgcolor="#eeeeee" style="padding:6px;color:#888888">
   Internal use only. Legacy system v3.2.
 </td></tr>
</table>
</body></html>
"""


@app.before_request
def _resolve_tenant():
    # Honor ?tenant=... on any request; the choice persists in the session so it
    # survives the app's own redirects (search -> /member/:id).
    current_tenant()


def render(body: str, **kw):
    return render_template_string(PAGE, body=render_template_string(body, **kw))


def require_login():
    if not session.get("user"):
        return redirect(url_for("login"))
    return None


@app.route("/")
def home():
    if not session.get("user"):
        return redirect(url_for("login"))
    return redirect(url_for("search"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        u = (request.form.get("username") or "").strip()
        p = (request.form.get("password") or "").strip()
        # Accept any non-empty creds. NEVER use real credentials here.
        if u and p:
            session["user"] = u
            return redirect(url_for("search"))
        error = "Username and password are required."
    body = """
    <b>Sign in</b>
    {% if error %}<div style="color:#cc0000;margin:6px 0">{{ error }}</div>{% endif %}
    <form method="post" action="/login">
     <table border="0" cellpadding="4">
      <tr><td><label for="u">Username</label></td>
          <td><input id="u" name="username" type="text"></td></tr>
      <tr><td><label for="pw">Password</label></td>
          <td><input id="pw" name="password" type="password"></td></tr>
      <tr><td></td><td><button type="submit">Sign in</button></td></tr>
     </table>
    </form>"""
    return render(body, error=error)


@app.route("/search")
def search():
    r = require_login()
    if r:
        return r
    body = """
    <b>Member lookup</b>
    <form method="get" action="/member">
     <table border="0" cellpadding="4">
      <tr><td><label for="mid">Member ID</label></td>
          <td><input id="mid" name="member_id" type="text"></td>
          <td><button type="submit">Search</button></td></tr>
     </table>
    </form>
    <div style="color:#888888;margin-top:8px">Try 100001, 100002, 222222, 999999.</div>"""
    return render(body)


@app.route("/member")
def member_lookup():
    """Search submits here with ?member_id=..., we redirect to canonical route."""
    r = require_login()
    if r:
        return r
    mid = (request.args.get("member_id") or "").strip()
    return redirect(url_for("member_detail", member_id=mid))


@app.route("/member/<member_id>")
def member_detail(member_id: str):
    r = require_login()
    if r:
        return r

    # ---- Injected runtime conditions (deterministic fault injection) ---- #
    inj = session.get("inject")
    if inj == "session_timeout":       # session expires right at the detail screen
        session.clear()
        return redirect(url_for("login"))
    if inj == "app_error":             # outright application error (hard failure)
        body = ("""<div style="color:#cc0000"><b>Internal error.</b>
                The system encountered an unexpected error (ref 500).</div>""")
        return render(body), 500
    if inj == "slow_load":             # transient slowness absorbed by retry
        time.sleep(3)

    # Interstitial "Notice" dialog -> recoverable condition. `persistent_notice`
    # makes it recur even after acknowledgement, to exercise bounded recovery ->
    # RECOVERY_EXHAUSTED.
    persistent = inj == "persistent_notice"
    if (member_id == "222222" and not session.get("ack_222222")) or persistent:
        body = """
        <b>Notice</b>
        <div style="border:1px solid #cc9900;background:#fffbe6;padding:10px;margin:8px 0">
          This member has a pending compliance review. Continue anyway?
        </div>
        <form method="post" action="/member/{{ mid }}/ack">
          <button type="submit">Continue</button>
        </form>"""
        return render(body, mid=member_id)

    if member_id == "999999":
        body = """
        <div style="color:#cc0000"><b>Permission denied.</b>
        You are not authorized to view this member.</div>
        <div style="margin-top:8px"><a href="/search">Back to search</a></div>"""
        return render(body), 403

    m = MEMBERS.get(member_id)
    if not m or member_id == "000000":
        body = """
        <div style="color:#cc0000"><b>No such member.</b>
        Member ID {{ mid }} was not found.</div>
        <div style="margin-top:8px"><a href="/search">Back to search</a></div>"""
        return render(body, mid=member_id), 404

    if request.args.get("slow"):
        time.sleep(3)  # transient slowness

    cfg = tenant_cfg()
    body = """
    <b>Member detail</b>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;margin-top:8px">
      <tr><td>Member ID</td><td>{{ mid }}</td></tr>
      <tr><td>Name</td><td>{{ m.name }}</td></tr>
      <tr><td>Status</td><td>{{ m.status }}</td></tr>
      <tr><td>Sub-accounts</td><td>{{ m.sub_accounts }}</td></tr>
      {% if decoy %}<tr><td>{{ decoy_label }}</td>
          <td>$<span>{{ decoy_value }}</span></td></tr>{% endif %}
      <tr><td>{{ savings_label }}</td>
          <td>$<span>{{ m.savings }}</span></td></tr>
    </table>
    <div style="margin-top:10px">
      <a href="/member/{{ mid }}/sub-account/new">Open new sub-account</a>
    </div>"""
    decoy = cfg.get("decoy")
    return render(body, mid=member_id, m=m,
                  savings_label=cfg["savings_label"],
                  decoy=bool(decoy),
                  decoy_label=(decoy[0] if decoy else ""),
                  decoy_value=(decoy[1] if decoy else ""))


@app.route("/member/<member_id>/ack", methods=["POST"])
def member_ack(member_id: str):
    r = require_login()
    if r:
        return r
    session["ack_222222"] = True
    return redirect(url_for("member_detail", member_id=member_id))


@app.route("/member/<member_id>/sub-account/new", methods=["GET"])
def subaccount_new(member_id: str):
    r = require_login()
    if r:
        return r
    if member_id not in MEMBERS:
        return redirect(url_for("member_detail", member_id=member_id))
    body = """
    <b>Open new sub-account for {{ mid }}</b>
    <form method="post" action="/member/{{ mid }}/sub-account/confirm">
     <table border="0" cellpadding="4">
      <tr><td><label for="atype">Account type</label></td>
          <td><select id="atype" name="account_type">
                <option value="">-- select --</option>
                <option value="savings">Savings</option>
                <option value="checking">Checking</option>
                <option value="cd">Certificate</option>
          </select></td></tr>
      <tr><td><label for="dep">Initial deposit</label></td>
          <td><input id="dep" name="deposit" type="text"></td></tr>
      <tr><td></td><td><button type="submit">Create sub-account</button></td></tr>
     </table>
    </form>"""
    return render(body, mid=member_id)


@app.route("/member/<member_id>/sub-account/confirm", methods=["POST"])
def subaccount_confirm(member_id: str):
    r = require_login()
    if r:
        return r
    atype = (request.form.get("account_type") or "").strip()
    deposit = (request.form.get("deposit") or "").strip()
    if not atype:
        body = """<div style="color:#cc0000">Account type is required.</div>
        <div><a href="/member/{{ mid }}/sub-account/new">Back</a></div>"""
        return render(body, mid=member_id), 400
    digest = hashlib.md5(f"{member_id}|{atype}|{deposit}".encode()).hexdigest()
    ref = f"SA-{member_id}-{int(digest, 16) % 100000:05d}"
    # Persist the mutation to BOTH datasets: an append-only audit row in changes.csv,
    # and the member's own record in members.csv (increment their sub-account count),
    # so the change is reflected in queryable member state, not only the audit log.
    m = MEMBERS.get(member_id)
    if m is not None:
        m["sub_accounts"] = str(int(m.get("sub_accounts", "0") or "0") + 1)
        save_members()
    record_change("open_subaccount", member_id, atype, deposit, ref)
    body = """
    <b>Sub-account created</b>
    <div style="border:1px solid #339900;background:#eaffea;padding:10px;margin:8px 0">
      Confirmation: a new <u>{{ atype }}</u> sub-account was created for member
      {{ mid }} with initial deposit ${{ deposit or '0.00' }}.
    </div>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
      <tr><td>Reference</td><td><span>{{ ref }}</span></td></tr>
    </table>"""
    return render(body, mid=member_id, atype=atype, deposit=deposit, ref=ref)


@app.route("/_simulate/expire")
def simulate_expire():
    """Testing hook: clear the session to simulate a timeout on next request."""
    session.clear()
    return "session cleared", 200


@app.route("/_simulate/inject/<cond>")
def simulate_inject(cond: str):
    """Arm (or clear) a deterministic runtime fault for this session.

    Conditions: session_timeout, app_error, slow_load, persistent_notice, clear.
    Used by `cua replay --inject <cond>` to make each runtime-condition branch
    reproducible on demand for evidence.
    """
    if cond == "clear":
        session.pop("inject", None)
        return "inject cleared", 200
    session["inject"] = cond
    return f"inject set: {cond}", 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
