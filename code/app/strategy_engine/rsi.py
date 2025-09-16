import pandas as pd

def compute_rsi(df: pd.DataFrame, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def rsi_signal(df: pd.DataFrame, window=14):
    df = df.copy()
    df['RSI'] = compute_rsi(df, window)
    latest = df.iloc[-1]

    if latest['RSI'] < 30:
        recommendation = "BUY"
        reason = f"RSI ({latest['RSI']:.2f}) indicates oversold conditions."
    elif latest['RSI'] > 70:
        recommendation = "SELL"
        reason = f"RSI ({latest['RSI']:.2f}) indicates overbought conditions."
    else:
        recommendation = "HOLD"
        reason = f"RSI ({latest['RSI']:.2f}) indicates neutral conditions."

    return {
        "signal": "RSI",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
