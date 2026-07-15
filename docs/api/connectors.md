# Connectors

Exchange data connectors. Two classes of loaders:

- **Raw trade loaders** (Binance, Bybit, OKX, Kraken, Coinbase) — download tick-level trades from public data dumps; used by the ML pipeline.
- **CCXT candle loader** — unified OHLCV candle downloader via the CCXT library; used by the event-driven strategies.

## Base

::: quant_research.connectors.base

## Binance

::: quant_research.connectors.binance

## Bybit

::: quant_research.connectors.bybit

## OKX

::: quant_research.connectors.okx

## Kraken

::: quant_research.connectors.kraken

## Coinbase

::: quant_research.connectors.coinbase

## CCXT OHLCV Loader

::: quant_research.connectors.ccxt_loader
