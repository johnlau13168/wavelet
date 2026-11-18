# Fuzzified Wavelet Alpha Engine (HKEX Leveraged ETFs)

A causal, walk-forward backtesting engine trading Hong Kong leveraged and inverse ETFs (**7226.HK** CSOP Hang Seng Tech 2x Bull and **7552.HK** CSOP Hang Seng Tech -2x Bear) using discrete wavelet decomposition and fuzzy game-theoretic position sizing.

## Strategy Overview
- **Timeframe Resampling:** Converts raw 1-minute HKEX session data (`09:30–12:00`, `13:00–16:00`) into uniform 75-minute bars (4 bars per trading day).
- **Wavelet Signal Extraction:** Applies rolling `db4` level-4 discrete wavelet transform (`pywt.wavedec`) to separate macro trend approximation (`cA4` slope) from cyclical detail (`cD3`).
- **Fuzzy Conviction & Sizing:** Uses smooth sigmoid membership functions and drawdown-aware ceiling scaling to size positions between `41%` and `74%` of available cash.
- **HKEX Execution Realism:** Enforces 100-share board lots, configurable tick-spread slippage (`spread_ticks`), and exact Futu HK regulatory fee calculations.

## Project Structure
- `wavelet_backtest.py` — Main backtesting engine and trade ledger generator.
- `config.yaml` — Externalized strategy, execution, and risk parameters.
- `requirements.txt` — Python package dependencies.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place market data:**
   Put your 1-minute CSV files (`7226_1m.csv` and `7552_1m.csv` with `datetime,open,high,low,close,volume` columns) inside `test_csv/`.

3. **Run backtest:**
   ```bash
   python wavelet_backtest.py
   ```
