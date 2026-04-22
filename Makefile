.PHONY: help install test lint fmt fmt-check typecheck docs docs-serve clean precommit lockfile

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@echo "Targets:"
	@echo "  install       - create .venv and pip install -e .[notebook,dev,docs]"
	@echo "  test          - run pytest"
	@echo "  lint          - ruff check"
	@echo "  fmt           - ruff check --fix + ruff format"
	@echo "  fmt-check     - ruff format --check (no write)"
	@echo "  typecheck     - mypy src/"
	@echo "  docs          - mkdocs build --strict"
	@echo "  docs-serve    - mkdocs serve (http://127.0.0.1:8000)"
	@echo "  precommit     - run pre-commit on all files"
	@echo "  lockfile      - regenerate requirements-lock.txt from pyproject.toml (strips torch)"
	@echo "  clean         - remove caches, build artifacts, site/"

install:
	bash setup.sh

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src/ tests/

fmt:
	$(PY) -m ruff check --fix src/ tests/
	$(PY) -m ruff format src/ tests/

fmt-check:
	$(PY) -m ruff format --check src/ tests/

typecheck:
	$(PY) -m mypy src/

docs:
	$(PY) -m mkdocs build --strict

docs-serve:
	$(PY) -m mkdocs serve

precommit:
	$(PY) -m pre_commit run --all-files

lockfile:
	$(PY) -m piptools compile \
	    --extra=dev --extra=docs --extra=notebook \
	    --strip-extras --no-emit-index-url --quiet \
	    --output-file=requirements-lock.txt pyproject.toml
	$(PY) scripts/strip_torch_from_lock.py requirements-lock.txt

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache site dist build
	rm -rf src/quant_research.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
