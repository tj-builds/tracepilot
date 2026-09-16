"""Playwright-backed web surface.

Perception is biased toward what survives on a legacy surface: we build the element
digest from accessibility role + accessible name (label/aria/placeholder/text),
not from framework-specific attributes. Locator resolution walks the artifact's
ordered candidate list and uses the first that resolves to a visible control.

Same-session handoff: the browser runs against a persistent `user_data_dir`, and when
escalation is enabled it is launched with a **CDP remote-debugging port**. On handoff,
automation does NOT close the context -- closing would drop in-memory session cookies
and it would no longer be the *same* session. Instead the context stays open and a
human attaches to the exposed CDP endpoint (`chrome://inspect` -> `localhost:<port>`,
or any co-browse tool) to drive the exact same live page; on resume, automation takes
control back. State is genuinely shared -- it is the same session, not a fresh one.
The scripted `AutoOperatorHandler` drives the same page in-process for the unattended
demo. (`resume_after_handoff` keeps a defensive re-open path only for a handler that
deliberately detaches the profile.)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from ..schema.artifact import Locator, LocatorCandidate, LocatorStrategy
from .base import ActResult, ElementDigest, Observation, Surface

_DIGEST_JS = r"""
() => {
  const roleFor = (el) => {
    const r = el.getAttribute('role');
    if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link';
    if (t === 'button') return 'button';
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textbox';
    if (t === 'input') {
      const ty = (el.getAttribute('type') || 'text').toLowerCase();
      if (['submit','button','reset'].includes(ty)) return 'button';
      if (ty === 'checkbox') return 'checkbox';
      if (ty === 'radio') return 'radio';
      return 'textbox';
    }
    return t;
  };
  const nameFor = (el) => {
    const al = el.getAttribute('aria-label');
    if (al) return al.trim();
    if (el.id) {
      const lab = document.querySelector(`label[for="${el.id}"]`);
      if (lab) return lab.textContent.trim();
    }
    const ph = el.getAttribute('placeholder');
    if (ph) return ph.trim();
    const t = (el.textContent || '').trim();
    if (t) return t.slice(0, 80);
    return (el.getAttribute('name') || el.getAttribute('value') || '').trim();
  };
  const editable = (el) => {
    const t = el.tagName.toLowerCase();
    if (t === 'textarea' || t === 'select') return true;
    if (t === 'input') {
      const ty = (el.getAttribute('type') || 'text').toLowerCase();
      return !['submit','button','reset','hidden'].includes(ty);
    }
    return false;
  };
  const nodes = Array.from(document.querySelectorAll(
    'a,button,input,select,textarea,[role=button],[role=link]'));
  const out = [];
  let i = 0;
  for (const el of nodes) {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    if ((el.getAttribute('type') || '').toLowerCase() === 'hidden') continue;
    out.push({
      ref: 'e' + (i++),
      role: roleFor(el),
      name: nameFor(el),
      value: (el.value !== undefined ? el.value : null),
      editable: editable(el),
      attrs: {
        tag: el.tagName.toLowerCase(),
        id: el.id || '',
        name: el.getAttribute('name') || '',
        type: el.getAttribute('type') || '',
        placeholder: el.getAttribute('placeholder') || '',
        text: (el.textContent || '').trim().slice(0, 80),
      }
    });
  }
  return out;
}
"""


class WebSurface(Surface):
    def __init__(self, headless: bool = True, timeout_ms: int = 8000,
                 user_data_dir: str | None = None, cdp_port: int | None = None):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.user_data_dir = user_data_dir or tempfile.mkdtemp(prefix="cua-profile-")
        # When set, Chromium is launched with a remote-debugging (CDP) port so a human
        # operator can attach to the SAME live context during handoff (see
        # `to_headed_for_handoff`). Enabled when escalation is active.
        self.cdp_port = cdp_port
        self._pw = None
        self._ctx = None
        self.page = None
        self._control = "automation"

    # ---- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        self._pw = sync_playwright().start()
        launch_kwargs = {"headless": self.headless}
        if self.cdp_port:
            # Expose CDP so an operator's browser/tooling can attach to this exact
            # context at handoff time (chrome://inspect -> localhost:<port>).
            launch_kwargs["args"] = [f"--remote-debugging-port={self.cdp_port}"]
        self._ctx = self._pw.chromium.launch_persistent_context(
            self.user_data_dir, **launch_kwargs
        )
        self._ctx.set_default_timeout(self.timeout_ms)
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()

    def cdp_endpoint(self) -> str | None:
        """The DevTools/CDP endpoint an operator attaches to during handoff, if the
        session was launched with a remote-debugging port; otherwise None."""
        return f"http://127.0.0.1:{self.cdp_port}" if self.cdp_port else None

    def stop(self) -> None:
        try:
            if self._ctx:
                self._ctx.close()
        finally:
            if self._pw:
                self._pw.stop()
            self._ctx = None
            self._pw = None
            self.page = None

    # ---- perception ------------------------------------------------------- #
    def observe(self, screenshot: bool = False) -> Observation:
        elements_raw = self.page.evaluate(_DIGEST_JS)
        elements = [
            ElementDigest(
                ref=e["ref"], role=e["role"], name=e["name"],
                value=e.get("value"), editable=e.get("editable", False),
                attrs=e.get("attrs", {}),
            )
            for e in elements_raw
        ]
        text = self.page_text()
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            elements=elements,
            text=text,
        )

    def page_text(self) -> str:
        try:
            return (self.page.inner_text("body") or "").strip()
        except Exception:
            return ""

    def current_url(self) -> str:
        return self.page.url if self.page else ""

    def screenshot(self, path: str) -> str | None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=path, full_page=True)
            return path
        except Exception:
            return None

    # ---- locator resolution ---------------------------------------------- #
    def _resolve(self, locator: Locator):
        """Return (playwright_locator, candidate) for the first resolving candidate."""
        for cand in locator.candidates:
            try:
                pl = self._candidate_to_locator(cand)
                if pl is None:
                    continue
                if pl.count() > 0:
                    return pl.first, cand
            except Exception:
                continue
        return None, None

    def _candidate_to_locator(self, c: LocatorCandidate):
        p = self.page
        s = c.strategy
        if s == LocatorStrategy.ROLE:
            role = c.role or "button"
            return p.get_by_role(role, name=c.value) if c.value else p.get_by_role(role)
        if s == LocatorStrategy.LABEL:
            return p.get_by_label(c.value)
        if s == LocatorStrategy.TEST_ID:
            return p.get_by_test_id(c.value)
        if s == LocatorStrategy.TEXT:
            return p.get_by_text(c.value)
        if s == LocatorStrategy.PLACEHOLDER:
            return p.get_by_placeholder(c.value)
        if s == LocatorStrategy.ALT:
            return p.get_by_alt_text(c.value)
        if s in (LocatorStrategy.CSS, LocatorStrategy.STRUCTURAL):
            return p.locator(c.value)
        if s == LocatorStrategy.XPATH:
            return p.locator(f"xpath={c.value}")
        return None

    # ---- actions ---------------------------------------------------------- #
    def navigate(self, url: str) -> ActResult:
        try:
            self.page.goto(url, wait_until="domcontentloaded")
            return ActResult(ok=True, detail=f"navigated to {url}")
        except Exception as exc:
            return ActResult(ok=False, detail=f"navigation failed: {exc}")

    def click(self, locator: Locator) -> ActResult:
        # Coordinate fallback (opaque UIs) is handled distinctly.
        pl, cand = self._resolve(locator)
        if pl is None:
            coord = next((c for c in locator.candidates
                          if c.strategy == LocatorStrategy.COORDINATES), None)
            if coord:
                return self._click_coord(coord.value)
            return ActResult(ok=False, detail=f"no locator resolved for '{locator.description}'")
        try:
            pl.click()
            return self._ok("clicked", locator, cand)
        except Exception as exc:
            return ActResult(ok=False, detail=f"click failed: {exc}")

    def _click_coord(self, value: str) -> ActResult:
        try:
            fx, fy = [float(x) for x in value.split(",")]
            box = self.page.viewport_size
            self.page.mouse.click(fx * box["width"], fy * box["height"])
            return ActResult(ok=True, detail="clicked (coordinates)",
                             locator_used=f"coordinates:{value}")
        except Exception as exc:
            return ActResult(ok=False, detail=f"coordinate click failed: {exc}")

    def type_text(self, locator: Locator, text: str) -> ActResult:
        pl, cand = self._resolve(locator)
        if pl is None:
            return ActResult(ok=False, detail=f"no locator resolved for '{locator.description}'")
        try:
            pl.fill(text)
            return self._ok("typed", locator, cand)
        except Exception as exc:
            return ActResult(ok=False, detail=f"type failed: {exc}")

    def select(self, locator: Locator, value: str) -> ActResult:
        pl, cand = self._resolve(locator)
        if pl is None:
            return ActResult(ok=False, detail=f"no locator resolved for '{locator.description}'")
        try:
            pl.select_option(value)
            return self._ok("selected", locator, cand)
        except Exception as exc:
            return ActResult(ok=False, detail=f"select failed: {exc}")

    def _ok(self, detail: str, locator: Locator, cand: LocatorCandidate) -> ActResult:
        """Build a success ActResult tagged with which locator-ladder rung fired."""
        try:
            rung = locator.candidates.index(cand)
        except ValueError:
            rung = 0
        return ActResult(ok=True, detail=detail,
                         locator_used=f"[{rung + 1}/{len(locator.candidates)}:"
                                      f"{cand.strategy.value}] {cand.value}",
                         rung=rung, candidates=len(locator.candidates))

    def press(self, key: str) -> ActResult:
        try:
            self.page.keyboard.press(key)
            return ActResult(ok=True, detail=f"pressed {key}")
        except Exception as exc:
            return ActResult(ok=False, detail=f"press failed: {exc}")

    def read(self, locator: Locator) -> str | None:
        pl, _ = self._resolve(locator)
        if pl is None:
            return None
        try:
            return (pl.inner_text() or "").strip()
        except Exception:
            return None

    # ---- handoff ---------------------------------------------------------- #
    def to_headed_for_handoff(self) -> str:
        """Yield control to a human who operates the SAME live context/page.

        We deliberately do NOT close the context: closing would drop in-memory
        session cookies, so it would no longer be the *same* session. When the surface
        was launched with a CDP port (the case whenever escalation is enabled), this
        returns a real **DevTools/CDP endpoint** (`http://127.0.0.1:<port>`) that a
        human operator attaches to — via `chrome://inspect` (configure
        `localhost:<port>`) or any CDP co-browse tool — to drive the exact same
        context/page. That is a genuine control transfer to a person on the live
        session, not a fresh one. The scripted `AutoOperatorHandler` used in the
        unattended demo drives the same page in-process instead. Falls back to the
        profile dir when no CDP port is configured.
        """
        self._control = "human"
        return self.cdp_endpoint() or self.user_data_dir

    def control_page(self):
        """The live page a human operator acts on during handoff (same session)."""
        return self.page

    def resume_after_handoff(self) -> None:
        """Take control back. The context stayed open, so state is fully preserved."""
        self._control = "automation"
        if self._ctx is None:  # only if a handler chose to fully detach
            self._ctx = self._pw.chromium.launch_persistent_context(
                self.user_data_dir, headless=self.headless)
            self._ctx.set_default_timeout(self.timeout_ms)
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
