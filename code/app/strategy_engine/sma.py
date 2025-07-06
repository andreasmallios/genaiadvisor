import pandas as pd

def sma_crossover_signal(df: pd.DataFrame, short_window=50, long_window=200):
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    latest = df.iloc[-1]

    if latest['SMA_short'] > latest['SMA_long']:
        recommendation = "BUY"
        reason = f"{short_window}-day SMA is above {long_window}-day SMA, indicating bullish momentum."
    else:
        recommendation = "HOLD"
        reason = f"{short_window}-day SMA is below {long_window}-day SMA, insufficient signal to buy."

    return {
        "signal": "SMA Crossover",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
