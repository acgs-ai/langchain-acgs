"""Check that the version in pyproject.toml matches ``langchain_acgs._version.__version__``."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Return 0 when both versions agree."""
    project = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    module = re.search(r'^__version__ = "([^"]+)"$', (ROOT / "langchain_acgs" / "_version.py").read_text(), re.M)
    project_version = project.group(1) if project else None
    module_version = module.group(1) if module else None
    if project_version is None or project_version != module_version:
        print(f"Version mismatch: pyproject.toml={project_version} _version.py={module_version}")
        return 1
    print(f"Version check passed: {project_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
