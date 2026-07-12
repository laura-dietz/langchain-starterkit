"""Checks that the installed autojudge-base is new enough.

Two layers:
1. against this repo's own pyproject pin (always runs), and
2. against the pin in the upstream template — read from the `starterkit`
   (or `upstream`) git remote that the setup guide has you keep — so a clone
   notices when the template has moved to a newer framework requirement.
   Skipped when no such remote/ref is available (e.g. offline, or inside the
   template itself).
"""

import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.version import Version

REPO = Path(__file__).parent.parent
PIN_RE = re.compile(r'"autojudge-base\s*>=\s*([0-9][0-9a-zA-Z.]*)"')


def _installed() -> Version:
    try:
        return Version(version("autojudge-base"))
    except PackageNotFoundError:
        pytest.fail("autojudge-base is not installed — run: uv pip install -e '.[all]'")


def _pin_from(text: str) -> Version:
    m = PIN_RE.search(text)
    assert m, "no autojudge-base>=X pin found in pyproject.toml"
    return Version(m.group(1))


def test_installed_meets_local_pin():
    """Installed autojudge-base satisfies this repo's own minimum pin."""
    pin = _pin_from((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    installed = _installed()
    assert installed >= pin, (
        f"installed autojudge-base {installed} < pinned {pin} — "
        "run: uv pip install -e '.[all]' --refresh"
    )


def _upstream_pyproject() -> str:
    """pyproject.toml from the upstream template's remote ref, or '' if unavailable."""
    for remote in ("starterkit", "upstream"):
        try:
            subprocess.run(
                ["git", "fetch", "--quiet", remote, "main"],
                cwd=REPO, capture_output=True, timeout=10, check=False,
            )  # best effort: offline is fine if a previously fetched ref exists
            out = subprocess.run(
                ["git", "show", f"{remote}/main:pyproject.toml"],
                cwd=REPO, capture_output=True, text=True, check=True,
            ).stdout
            if out:
                return out
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            continue
    return ""


def test_installed_meets_upstream_template_pin():
    """Installed autojudge-base is the same or later than the upstream template requires."""
    upstream = _upstream_pyproject()
    if not upstream:
        pytest.skip("no starterkit/upstream remote ref available (offline or template repo)")
    pin = _pin_from(upstream)
    installed = _installed()
    assert installed >= pin, (
        f"installed autojudge-base {installed} < {pin} required by the upstream "
        "template — the framework moved on; run: uv pip install --upgrade autojudge-base "
        "(and consider pulling template changes: git fetch starterkit && git merge starterkit/main)"
    )
