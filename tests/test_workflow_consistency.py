"""Consistency between a judge's workflow.yml lifecycle flags and its classes.

If a judge declares it *produces* an artifact — `create_nuggets`, `create_qrels`,
or `judge` — the class that produces it must be wired in the same `workflow.yml`.
This is the cheap, LLM-free half of "verify your outputs": it catches a judge
that claims to emit nuggets or qrels but has no class to build them, before a run
ever starts, instead of failing mid-run.

The complementary runtime guarantee — that a produced artifact is actually
complete (non-empty nugget banks, every expected topic scored) — is enforced by
the workflow runner's own verification during the run, and encouraged as an
explicit `verify(...)` call in judge code (see the develop-an-autojudge howto).
Judges are discovered dynamically, so this never goes stale as judges change.
"""

import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parent.parent


def _tracked_workflows():
    """Judges defined in this repo = git-tracked judges/*/workflow.yml
    (a filesystem glob would also pick up local untracked leftovers)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "judges/*/workflow.yml"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split()
        if out:
            return [REPO / p for p in out]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return sorted(REPO.glob("judges/*/workflow.yml"))


WORKFLOWS = _tracked_workflows()

# (lifecycle flag that declares production, class that must produce it, flag default).
# Defaults match the runner: judge runs unless disabled; nuggets/qrels are opt-in.
PRODUCER_RULES = [
    ("create_nuggets", "nugget_class", False),
    ("create_qrels", "qrels_class", False),
    ("judge", "judge_class", True),
]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.parent.name)
def test_declared_outputs_have_a_producer_class(workflow):
    cfg = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
    for flag, class_key, default in PRODUCER_RULES:
        if cfg.get(flag, default):
            assert cfg.get(class_key), (
                f"{workflow.parent.name}: workflow.yml has {flag}: "
                f"{cfg.get(flag, default)} but declares no {class_key} to produce "
                f"that artifact — the run will fail when it reaches that phase. "
                f"Wire the class, or set {flag}: false."
            )
