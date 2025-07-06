import pandas as pd

def macd_signal(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9):
    df = df.copy()
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    latest = df.iloc[-1]

    if latest['MACD'] > latest['Signal_Line']:
        recommendation = "BUY"
        reason = "MACD is above the Signal Line, indicating potential bullish momentum."
    else:
        recommendation = "HOLD"
        reason = "MACD is below the Signal Line, insufficient signal to buy."

    return {
        "signal": "MACD",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
