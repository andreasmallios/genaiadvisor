import pandas as pd
from datetime import timedelta
from app.data_ingestion import fetch_ticker_data
# from app.strategy_engine import generate_recommendation
from app.strategy_engine.engine import generate_combined_recommendation

def backtest_ticker(ticker: str, as_of_date: str, lookahead_days: int = 30) -> dict:
    """
    Run a backtest for the given ticker as of a historical date.
    Returns recommendation and price movement over lookahead_days.
    """
    cutoff_date = pd.to_datetime(as_of_date).tz_localize('America/New_York')

    # Get data up to the cutoff date for generating the recommendation
    df = fetch_ticker_data(ticker)
    df = df[df.index <= cutoff_date]

    if df.empty:
        raise ValueError(f"No data available for {ticker} up to {as_of_date}.")

    # recommendation = generate_recommendation(df, ticker)
    recommendation = generate_combined_recommendation(df, ticker)

    # Evaluate actual price movement over lookahead_days
    df_future = fetch_ticker_data(ticker)
    df_future = df_future[(df_future.index > cutoff_date) &
                          (df_future.index <= (cutoff_date + timedelta(days=lookahead_days)))]

    if df_future.empty:
        price_change = None
    else:
        start_price = df_future.iloc[0]["Close"]
        end_price = df_future.iloc[-1]["Close"]
        price_change_pct = ((end_price - start_price) / start_price) * 100
        price_change = round(price_change_pct, 2)

    result = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "recommendation": recommendation["recommendation"],
        "reason": recommendation["reason"],
        "price_change_%": price_change
    }

    return result

if __name__ == "__main__":
    test_result = backtest_ticker("MSFT", "2025-06-01", lookahead_days=30)
    print(test_result)
