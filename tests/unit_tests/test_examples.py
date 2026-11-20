"""The example script runs, prints what the README says, and every README code block is checked."""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(b) for b in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(QUICKSTART)], capture_output=True, text=True, check=True, cwd=ROOT, timeout=120
    )
    [output] = [b for b in _blocks("text") if b.startswith("send_email:")]
    assert completed.stdout == output


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_deny_and_audit_runs_offline_without_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in list(os.environ):
        if key.endswith("API_KEY"):
            monkeypatch.delenv(key)
    # Register values first so monkeypatch restores the environment after the example sets them.
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(key, "false")
    # pytest-socket disables network access for this in-process execution.
    runpy.run_path(str(ROOT / "examples" / "deny_and_audit.py"), run_name="__main__")
    output = capsys.readouterr().out
    execution, audit = output.split("Audit trail (application decisions):\n")
    assert execution == "Executed record_note: The meeting starts at noon.\n"
    decisions = [json.loads(line) for line in audit.splitlines()]
    assert [decision["verdict"] for decision in decisions] == ["allow", "deny"]
    assert [decision["tool_name"] for decision in decisions] == ["record_note", "record_note"]
    assert decisions[0]["rule_ids"] == []
    assert decisions[1]["rule_ids"] == ["ACGS-006"]
    assert "Blocked by" in decisions[1]["reason"]
    assert all(decision["error"] is None for decision in decisions)
    assert "demo-secret" not in output


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 5
    for block in blocks:
        exec(compile(block, "<README>", "exec", dont_inherit=True), {"__name__": "readme"})  # noqa: S102
    assert "normalized: update config | dry run" in capsys.readouterr().out
