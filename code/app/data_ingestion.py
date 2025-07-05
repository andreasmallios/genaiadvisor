import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def fetch_ticker_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch ticker data cleanly without metadata headers, cache to CSV with Date index.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Loading cached data for {ticker} from {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        print(f"[INFO] Fetching data for {ticker} using Ticker.history() to avoid metadata pollution")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data fetched for ticker: {ticker}")

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # drop 'Dividends' and 'Stock Splits'
        df.index.name = "Date"
        df.to_csv(csv_path)
        print(f"[INFO] Saved clean data for {ticker} to {csv_path}")

    return df

if __name__ == "__main__":
    ticker = "MSFT"
    df = fetch_ticker_data(ticker)
    print(df.tail())
