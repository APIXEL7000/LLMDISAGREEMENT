"""API-key handling.

Rules
-----
* Keys come from (1) the key typed in the web page for this run only, or (2) environment
  variables, optionally loaded from a local `.env` file (git-ignored).
* Keys are never written to disk, to the response cache, to run exports, or to logs, and
  are never sent back to the browser (only a yes/no "key available" flag is).
* Every error message passes through `redact()` before it is shown.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .config import ROOT

# provider -> (display name, environment variables that may hold the key)
KEY_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama-cloud": ("OLLAMA_API_KEY",),
}


def load_dotenv(path: Path | None = None) -> int:
    """Load KEY=VALUE lines from `.env` into os.environ (existing variables win)."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def get_key(provider: str, supplied: str | None = None) -> str | None:
    """Key typed by the user for this run, else the environment, else None."""
    if supplied and supplied.strip():
        return supplied.strip()
    for var in KEY_ENV.get(provider, ()):
        if os.environ.get(var, "").strip():
            return os.environ[var].strip()
    return None


def key_status() -> dict[str, bool]:
    """Which providers have a key configured on the server (booleans only)."""
    return {p: get_key(p) is not None for p in KEY_ENV}


def redact(text: object, *keys: str | None) -> str:
    """Remove any known key (and anything shaped like one) from a message."""
    out = str(text)
    for k in keys:
        if k and len(k) >= 6:
            out = out.replace(k, "***")
    out = re.sub(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{8,})", "***", out)
    return out
