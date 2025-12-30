# SPY-Vs.-NVDA

Compares NVIDIA (NVDA) to SPY as a benchmark using historical adjusted close prices. Computes daily and cumulative returns plus key metrics (CAGR, volatility, beta, correlation, max drawdown) and saves simple plots and a metrics table.

## How to run
1. Install dependencies:
   - `pip install yfinance pandas numpy matplotlib`

2. Run:
   - `python main.py`

## Output
The script creates an `outputs/` folder containing:
- `metrics.csv`
- `prices.png`
- `cumulative_returns.png`

## Notes
- Edit tickers and dates at the top of `main.py` if needed.

