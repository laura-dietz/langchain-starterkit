"""Checks that the template was made your own (mandatory before submitting).

These tests stay silent inside the bare-bones template itself (where origin
points at trec-auto-judge/auto-judge-starter-kit, or no origin exists yet).
Once you add your own `origin` remote — the last command of setup Step 1 —
they fail until you complete setup Step 3: rename the project in
pyproject.toml and replace README.md with a description of YOUR judge.
See: https://github.com/trec-auto-judge/.github/blob/main/profile/howto/01-setup-environment.md
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
TEMPLATE_NAME = "auto-judge-starterkit"
TEMPLATE_REMOTE = "trec-auto-judge/auto-judge-starter-kit"


def _skip_reason() -> str:
    """Empty string = tests apply; otherwise why they are skipped."""
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "no origin remote yet (finish setup Step 1 first)"
    if TEMPLATE_REMOTE in url:
        return "running inside the bare-bones template itself"
    return ""


pytestmark = pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or "n/a")


def test_project_name_customized():
    """pyproject.toml must not still carry the template's project name."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^name\s*=\s*"([^"]+)"', text)
    assert m, "pyproject.toml declares no project name"
    assert m.group(1) != TEMPLATE_NAME, (
        f'pyproject.toml still says name = "{TEMPLATE_NAME}" — rename the project '
        "to make the template your own (setup Step 3, mandatory before submitting)"
    )


def test_readme_customized():
    """README.md must describe YOUR judge, not the starter-kit template."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert not readme.lstrip().startswith("# Auto-Judge Starterkit"), (
        "README.md still carries the template's title — replace it with a "
        "description of your judge (setup Step 3, mandatory before submitting)"
    )
    assert "A forkable template repository with example Auto-Judge implementations" not in readme, (
        "README.md still contains the template's tagline — replace the overview "
        "with a description of your judge (setup Step 3)"
    )
