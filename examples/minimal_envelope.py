"""
minimal_envelope.py — Smallest runnable envelope backtest.

Usage
-----
    # one-time: download + cache candles
    python examples/minimal_envelope.py --download

    # subsequent runs: read from cache
    python examples/minimal_envelope.py

Assumes the project is installed editable (`pip install -e .`) and runs
from a virtualenv where dependencies are available.
"""

from __future__ import annotations

import argparse
import sys

from quant_research.backtest import BacktestAnalysis
from quant_research.connectors import CCXTLoader
from quant_research.strategies import EnvelopeStrategy


def main(download: bool, symbol: str, timeframe: str, start_date: str) -> int:
    loader = CCXTLoader(exchange="binanceusdm")

    if download:
        print(f"Downloading {symbol} {timeframe} from {start_date} ...")
        loader.download(symbol, timeframe=timeframe, start_date=start_date)

    df = loader.load(symbol, timeframe=timeframe, start_date=start_date)
    print(f"Loaded {df.height} bars")

    strategy = EnvelopeStrategy(
        params={
            "average_type": "SMA",
            "average_period": 6,
            "envelopes": [0.07, 0.11, 0.14],
            "stop_loss_pct": 0.3,
            "position_size_percentage": 100,
        },
        ohlcv=df,
    )
    strategy.run_backtest(
        initial_balance=1000,
        leverage=1,
        open_fee_rate=0.0002,
        close_fee_rate=0.0006,
    )

    results = BacktestAnalysis(strategy)
    results.print_metrics()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="download + cache candles before loading")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start-date", default="2023-01-01 00:00:00")
    args = parser.parse_args()
    sys.exit(main(args.download, args.symbol, args.timeframe, args.start_date))
