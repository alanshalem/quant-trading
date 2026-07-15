# 02 · Classical Strategies (no ML)

Two complete, self-contained trading strategies built with **pandas only** — no
machine learning, no `quant_research` library. They are the bridge between the
fundamentals in [`01_notebooks/`](../01_notebooks) (where module 04 introduces
*mean reversion vs momentum* and module 08 introduces *signals and costs*) and
the ML track in [`03_ml_strategy/`](../03_ml_strategy).

Each notebook derives an edge from a single statistical pattern — the sign of the
previous bar's log return — then backtests it end to end, including round-trip
maker/taker fees.

| # | Notebook | Signal | Data | Bar |
|---|----------|--------|------|-----|
| 01 | `01-momentum.ipynb` | `signal = sign(return_lag_1)` — follow the trend | TAO | weekly (1w) |
| 02 | `02-mean_reversion.ipynb` | `signal = -sign(return_lag_1)` — fade the move | BCH | daily (1d) |

Both notebooks share the same 8-step skeleton, differing only in the **sign** of
the signal:

```
load OHLC → log returns → lag(1) → sign() → group by direction
→ in/out-of-sample split (75/25) → build signal → trade log return
→ cumulative equity → round-trip fees → net equity
```

## Data

The notebooks read committed sample CSVs from [`../../data/samples/`](../../data/samples)
(`momentum_ohlc.csv`, `mean_reversion_ohlc.csv`). No network access required — they
run offline and inside Docker. The path resolves whether the notebook is launched
from the repo root (VS Code default) or from this folder.

## Annualized Sharpe

The annualization factor matches each dataset's bar size, so the two differ on
purpose:

- Momentum (weekly): `sqrt(365 / 7) ≈ sqrt(52)`
- Mean reversion (daily): `sqrt(365)` — crypto trades every day

## Run

Open either notebook in VS Code (the repo `.venv` kernel is auto-detected) and
click **Run All**, or launch JupyterLab via `docker compose up`.
