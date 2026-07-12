"""Smoke tests: every judge defined in this repo parses and imports."""

import importlib
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

def test_judges_discovered():
    """At least one judge with a workflow.yml is defined in this repo."""
    assert WORKFLOWS, "no tracked judges/*/workflow.yml found"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.parent.name)
def test_workflow_parses_and_classes_import(workflow):
    """Every judge defined in this repo has a parseable workflow.yml whose
    declared classes import. Judges are discovered dynamically (git-tracked
    workflow.yml files), so this test never goes stale when judges are
    added or removed."""
    cfg = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    refs = [cfg[k] for k in ("judge_class", "nugget_class", "qrels_class") if cfg.get(k)]
    assert refs, f"{workflow} declares no judge/nugget/qrels class"
    for ref in refs:
        module_name, _, attr = ref.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), f"{ref}: {attr} not found in {module_name}"