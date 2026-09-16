"""Runtime configuration — loaded strictly from the environment (`.env`).

Every value comes directly from the environment; there are **no hardcoded fallback
defaults** in the code. The canonical values (target URL, model, runtime knobs, mock
credentials) live in `.env.example` — copy it to `.env` and fill in your
`OPENAI_API_KEY`. A missing required variable fails fast with a clear message rather
than silently substituting a default.

LLM provider is **OpenAI only** and is **required** for routing/discovery/validation
(`agent/llm.build_client` raises without a key). There is no multi-provider selector
and no offline fallback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
EVIDENCE_DIR = REPO_ROOT / "evidence"

# Load the project's .env explicitly (by absolute path) so configuration resolves the
# same way regardless of the current working directory. Real environment variables
# already set in the process take precedence (load_dotenv does not override them).
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv missing is a setup error surfaced below
    pass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or malformed."""


_SETUP_HINT = "Copy .env.example to .env and set it (see README > Setup)."


def _require(name: str) -> str:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        raise ConfigError(f"Required environment variable '{name}' is not set. {_SETUP_HINT}")
    return val.strip()


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable '{name}' must be an integer, got '{raw}'.") from exc


def _require_bool(name: str) -> bool:
    raw = _require(name).lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(
        f"Environment variable '{name}' must be a boolean (true/false), got '{raw}'.")


@dataclass(frozen=True)
class Config:
    # Secret: read directly from the environment. Absent -> None (enforced where the
    # model is actually needed, in build_client). Never carries a hardcoded default.
    openai_api_key: str | None
    openai_model: str
    target_base_url: str
    operator_base_url: str
    # Mock-app login. Supplied at invocation as sensitive params (redacted from logs,
    # never persisted). The mock accepts any non-empty pair; values live in .env.
    app_username: str
    app_password: str
    headless: bool
    max_steps: int
    step_timeout_ms: int
    # Remote-debugging (CDP) port exposed during human handoff so an operator can
    # attach to the SAME live browser session and drive it.
    handoff_cdp_port: int

    @classmethod
    def from_env(cls) -> "Config":
        """Build the config strictly from the environment. Every field except the
        secret key is required; a missing one raises `ConfigError`."""
        return cls(
            openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
            openai_model=_require("OPENAI_MODEL"),
            target_base_url=_require("TARGET_BASE_URL"),
            operator_base_url=_require("OPERATOR_BASE_URL"),
            app_username=_require("APP_USERNAME"),
            app_password=_require("APP_PASSWORD"),
            headless=_require_bool("HEADLESS"),
            max_steps=_require_int("MAX_STEPS"),
            step_timeout_ms=_require_int("STEP_TIMEOUT_MS"),
            handoff_cdp_port=_require_int("HANDOFF_CDP_PORT"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def model_label(self) -> str:
        return self.openai_model

    def ensure_dirs(self) -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = Config.from_env()
