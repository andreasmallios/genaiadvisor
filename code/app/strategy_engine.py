# import pandas as pd

# def generate_recommendation(df: pd.DataFrame, ticker: str) -> dict:
#     """
#     Given a DataFrame of OHLCV data, compute 50-day and 200-day SMAs,
#     and return a structured recommendation.
#     """
#     df = df.copy()
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()
#     df['SMA_200'] = df['Close'].rolling(window=200).mean()

#     latest = df.iloc[-1]
#     recommendation = "HOLD"
#     reason = "Insufficient signal to buy."

#     if latest['SMA_50'] > latest['SMA_200']:
#         recommendation = "BUY"
#         reason = "50-day SMA is above the 200-day SMA, indicating bullish momentum."

#     return {
#         "ticker": ticker,
#         "recommendation": recommendation,
#         "reason": reason,
#         "date": str(latest.name.date())
#     }

# if __name__ == "__main__":
#     from data_ingestion import fetch_ticker_data

#     ticker = "MSFT"
#     df = fetch_ticker_data(ticker)
#     rec = generate_recommendation(df, ticker)
#     print(rec)
