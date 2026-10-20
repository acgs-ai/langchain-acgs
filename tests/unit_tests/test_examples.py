"""The example script runs, prints what the README says, and every README code block is checked."""

from __future__ import annotations

import re
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


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 5
    for block in blocks:
        exec(compile(block, "<README>", "exec", dont_inherit=True), {"__name__": "readme"})  # noqa: S102
    assert "normalized: update config | dry run" in capsys.readouterr().out
