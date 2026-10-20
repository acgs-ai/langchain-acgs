"""Import every file given on the command line and report the ones that fail."""

import importlib.util
import sys
import traceback


def main(paths: list[str]) -> int:
    """Return 1 if any file fails to import."""
    failed = False
    for index, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location(f"_check_{index}", path)
        if spec is None or spec.loader is None:
            print(f"{path}: cannot be loaded")
            failed = True
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001  # report every failure, then exit non-zero
            print(path)
            traceback.print_exc()
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
