import pandas as pd
from datetime import datetime
import os
from app.backtesting import backtest_ticker

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_batch_backtest(tickers, dates, lookahead_days=30):
    results = []

    for ticker in tickers:
        for date in dates:
            try:
                result = backtest_ticker(ticker, date, lookahead_days=lookahead_days)
                results.append(result)
                print(f"[INFO] Completed: {ticker} on {date}")
            except Exception as e:
                print(f"[WARN] Skipping {ticker} on {date} due to: {e}")

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"batch_backtest_{timestamp}.csv")
    df_results.to_csv(output_path, index=False)
    print(f"[INFO] Batch backtest results saved to {output_path}")

if __name__ == "__main__":
    # Load tickers from CSV file
    tickers_df = pd.read_csv("./data/tickers.csv")
    tickers = tickers_df["Symbol"].dropna().unique().tolist()
    dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]

    run_batch_backtest(tickers, dates, lookahead_days=30)
