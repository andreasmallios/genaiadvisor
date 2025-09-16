import pandas as pd

def stochastic_signal(df: pd.DataFrame, k_window=14, d_window=3):
    df = df.copy()
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_window).mean()
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # BUY: %K crosses above %D below 20 (oversold)
    if (prev['%K'] < prev['%D']) and (latest['%K'] > latest['%D']) and (latest['%K'] < 20):
        recommendation = "BUY"
        reason = f"Stochastic %K crossed above %D below 20, indicating potential bullish reversal."
    # SELL: %K crosses below %D above 80 (overbought)
    elif (prev['%K'] > prev['%D']) and (latest['%K'] < latest['%D']) and (latest['%K'] > 80):
        recommendation = "SELL"
        reason = f"Stochastic %K crossed below %D above 80, indicating potential bearish reversal."
    else:
        recommendation = "HOLD"
        reason = "Stochastic conditions do not indicate a clear reversal."

    return {
        "signal": "Stochastic Oscillator",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
