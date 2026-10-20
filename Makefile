.PHONY: all format lint type test tests integration_tests check_imports check_version help

all: help

TEST_FILE ?= tests/unit_tests/ langchain_acgs/
PYTEST_EXTRA ?=
PYTHON_FILES = .

test tests:
	uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket --cov --cov-report=term-missing $(TEST_FILE)

integration_test integration_tests:
	uv run --group test pytest $(PYTEST_EXTRA) tests/integration_tests/

lint:
	./scripts/lint_imports.sh
	uv run --all-groups ruff check $(PYTHON_FILES)
	uv run --all-groups ruff format $(PYTHON_FILES) --diff
	uv run --all-groups mypy

type:
	uv run --all-groups mypy

format:
	uv run --all-groups ruff format $(PYTHON_FILES)
	uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_acgs -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

check_version:
	uv run python ./scripts/check_version.py

help:
	@echo 'format          - run code formatters'
	@echo 'lint            - run import checks, ruff and mypy --strict'
	@echo 'type            - run mypy --strict'
	@echo 'test            - run unit tests with the coverage gate (network disabled)'
	@echo 'check_imports   - check that every module imports'
	@echo 'check_version   - check that pyproject.toml and _version.py agree'
