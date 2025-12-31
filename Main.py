"""
NVDA vs SPY Benchmark Comparison (Simple Version)

What this program does:
1) Downloads Adjusted Close prices for NVDA and SPY
2) Computes daily returns and cumulative returns (growth of $1)
3) Computes basic stats: CAGR, annual volatility, correlation, beta, max drawdown
4) Saves two plots + a metrics CSV into the outputs/ folder

How to run/execute:
A) Install packages (once):
   pip install yfinance pandas numpy matplotlib

B) Run the script:
   python main.py

Notes:
- Keep it simple and make sure it works (project guideline).
- This script is intentionally not "too advanced".
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------
# SETTINGS (edit if needed)
# -----------------------
ASSET_TICKER = "NVDA"
BENCHMARK_TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = ""  # leave empty to use today's date
OUTPUT_DIR = "outputs"
TRADING_DAYS_PER_YEAR = 252


# -----------------------
# Helper functions
# -----------------------
def ensure_output_dir(folder_name: str) -> None:
    """Create output folder if it doesn't exist."""
    os.makedirs(folder_name, exist_ok=True)


def download_adjusted_close(tickers, start_date, end_date):
    """
    Download Adjusted Close prices for the given tickers.
    Basic validation included to avoid silent failures.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date if end_date else None,
        progress=False,
        auto-adjust=False
    )

    if data.empty:
        raise ValueError("No data returned. Check tickers and date range.")

    adj_close = data["Adj Close"]

    # If only one ticker, yfinance may return a Series instead of DataFrame
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])

    # Forward fill missing values (simple handling)
    adj_close = adj_close.dropna(how="all").ffill()

    return adj_close


def compute_daily_returns(prices_df):
    """Compute daily simple returns from price levels."""
    returns = prices_df.pct_change().dropna()
    if returns.empty:
        raise ValueError("Returns are empty after pct_change().")
    return returns


def cumulative_growth(returns_series):
    """Convert daily returns into a growth-of-1 series."""
    return (1 + returns_series).cumprod()


def cagr(daily_returns):
    """Compute CAGR (annualized geometric return)."""
    total_growth = (1 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    return float(total_growth ** (1 / years) - 1)


def annual_volatility(daily_returns):
    """Annualized volatility from daily returns."""
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(growth_series):
    """Maximum drawdown (negative number)."""
    peak = growth_series.cummax()
    drawdown = (growth_series / peak) - 1.0
    return float(drawdown.min())


def beta(asset_returns, benchmark_returns):
    """Beta of asset vs benchmark."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")

    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1]
    var = np.var(aligned.iloc[:, 1], ddof=1)

    if var == 0:
        return float("nan")

    return float(cov / var)


def save_line_plot(df, title, y_label, filepath):
    """Save a simple line plot for each column of df."""
    fig = plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# -----------------------
# Main program
# -----------------------
def main():
    ensure_output_dir(OUTPUT_DIR)

    tickers = [ASSET_TICKER, BENCHMARK_TICKER]

    prices = download_adjusted_close(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE
    )

    returns = compute_daily_returns(prices)

    asset_ret = returns[ASSET_TICKER]
    bench_ret = returns[BENCHMARK_TICKER]

    asset_growth = cumulative_growth(asset_ret)
    bench_growth = cumulative_growth(bench_ret)

    # Metrics table (simple and readable)
    metrics = pd.DataFrame(
        {
            ASSET_TICKER: {
                "CAGR": cagr(asset_ret),
                "Annual Volatility": annual_volatility(asset_ret),
                "Max Drawdown": max_drawdown(asset_growth),
                "Correlation with SPY": float(asset_ret.corr(bench_ret)),
                "Beta vs SPY": beta(asset_ret, bench_ret),
            },
            BENCHMARK_TICKER: {
                "CAGR": cagr(bench_ret),
                "Annual Volatility": annual_volatility(bench_ret),
                "Max Drawdown": max_drawdown(bench_growth),
                "Correlation with SPY": 1.0,
                "Beta vs SPY": 1.0,
            },
        }
    )

    # Save outputs
    metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))

    save_line_plot(
        df=prices,
        title="Adjusted Close Prices",
        y_label="Price",
        filepath=os.path.join(OUTPUT_DIR, "prices.png"),
    )

    cumulative_df = pd.DataFrame(
        {ASSET_TICKER: asset_growth, BENCHMARK_TICKER: bench_growth}
    )

    save_line_plot(
        df=cumulative_df,
        title="Cumulative Returns (Growth of $1)",
        y_label="Growth",
        filepath=os.path.join(OUTPUT_DIR, "cumulative_returns.png"),
    )

    # Print summary (so the program clearly “does something” when run)
    print("\n=== NVDA vs SPY Summary ===")
    print(metrics.round(4))
    print("\nSaved outputs to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Simple, clear error message (helps debugging / grading)
        print("\nERROR:", error)
        print("Tip: check your internet connection and that the packages are installed.")
