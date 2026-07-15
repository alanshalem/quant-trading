# Contributing

## Setup

```bash
git clone <repo-url> quant-trading
cd quant-trading
bash setup.sh          # Mac/Linux
setup.bat              # Windows
```

The setup script installs all extras (`[notebook,dev,docs,ml]`). If you
only need the event-driven path (no PyTorch), install manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"       # skip [ml] → no torch (~500 MB saved)
```

Activate the venv afterward:

```bash
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate.bat    # Windows
```

Install pre-commit hooks once:

```bash
pre-commit install
```

## Daily workflow

Common commands live in the [Makefile](Makefile):

| Command        | What it does |
|----------------|--------------|
| `make test`    | Run pytest (expect 28+ passing) |
| `make lint`    | Ruff check — no write |
| `make fmt`     | Ruff auto-fix + format |
| `make typecheck` | Mypy on `src/` |
| `make docs`    | Build mkdocs site under `site/` |
| `make docs-serve` | Live-reload docs at <http://127.0.0.1:8000> |
| `make precommit` | Run all pre-commit hooks on every file |
| `make clean`   | Nuke caches, build artifacts, egg-info |

## Code style

- Ruff with `line-length = 120` (see `[tool.ruff]` in `pyproject.toml`).
- Selected lint rules: `E`, `F`, `W`, `I`. `E501` ignored.
- Ruff format (Black-compatible) runs on every commit via pre-commit.
- Mypy on `src/` — gradual typing, prefer explicit annotations on public functions.
- Prefer Polars over Pandas. If you need Pandas at a boundary, convert at the edge.
- No emojis in code or commits unless explicitly requested by the user.

## Commit hooks

`.pre-commit-config.yaml` installs:

- `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending` (LF).
- `check-yaml`, `check-toml`, `check-added-large-files` (<=2MB).
- `ruff` + `ruff-format`.
- `nbstripout` — kills cell output in `.ipynb` so diffs stay reviewable.

Skip hooks only as a last resort (`git commit --no-verify`).

## Tests

Place new tests in `tests/` matching `test_*.py`. Reuse the
`synthetic_ohlcv` fixture from `tests/conftest.py` for strategy /
backtest tests — keeps the suite deterministic.

```python
def test_something(synthetic_ohlcv):
    strat = EnvelopeStrategy(params, synthetic_ohlcv)
    ...
```

## Docs

Update docstrings in the module, not a separate `.md`. The mkdocstrings
plugin renders them automatically. Check with `make docs` before pushing.

## PRs

- One logical change per PR.
- Tests must pass (`make test`) and lint must be clean (`make lint`).
- CI runs Python 3.10 / 3.11 / 3.12 matrix — green across all before merge.
- Mention which paradigm (vectorized / event-driven) the change touches
  in the PR description.

## Reporting bugs

Open an issue with:

- Steps to reproduce.
- Expected vs actual behavior.
- Version of Python / polars / torch (`pip list | grep -E 'polars|torch|ccxt'`).
