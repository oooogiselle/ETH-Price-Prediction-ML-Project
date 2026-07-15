"""
data_loader.py

Downloads historical Ethereum (ETH-USD) OHLCV data using yfinance
(the original BTC project pulled data from the CoinMarketCap web API;
here we swap in yfinance since it's free, key-less, and stable long-term).

Usage:
    python data_loader.py
"""

import yfinance as yf
import pandas as pd
import os

TICKER = "ETH-USD"
START_DATE = "2017-11-09"  # earliest date yfinance has reliable ETH-USD data
END_DATE = None            # None -> up to today
OUT_PATH = os.path.join("data", "eth_ohlcv.csv")


def download_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check your connection or ticker symbol.")

    # Flatten possible MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = download_data()
    df.to_csv(OUT_PATH)
    print(f"Saved {len(df)} rows of {TICKER} data to {OUT_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()
