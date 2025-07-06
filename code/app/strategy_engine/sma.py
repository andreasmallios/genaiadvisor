import pandas as pd

def sma_crossover_signal(df: pd.DataFrame, short_window=50, long_window=200):
    """
    Generates BUY, HOLD, or SELL based on SMA crossover:
    - BUY: short SMA crosses above long SMA (bullish momentum)
    - SELL: short SMA crosses below long SMA (bearish momentum)
    - HOLD: otherwise
    """
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df['SMA_prev_short'] = df['SMA_short'].shift(1)
    df['SMA_prev_long'] = df['SMA_long'].shift(1)
    latest = df.iloc[-1]

    if (
        latest['SMA_short'] > latest['SMA_long'] and
        latest['SMA_prev_short'] <= latest['SMA_prev_long']
    ):
        recommendation = "BUY"
        reason = (
            f"The {short_window}-day SMA has just crossed above the {long_window}-day SMA, "
            "indicating a new bullish momentum signal."
        )
    elif (
        latest['SMA_short'] < latest['SMA_long'] and
        latest['SMA_prev_short'] >= latest['SMA_prev_long']
    ):
        recommendation = "SELL"
        reason = (
            f"The {short_window}-day SMA has just crossed below the {long_window}-day SMA, "
            "indicating a new bearish momentum signal."
        )
    else:
        recommendation = "HOLD"
        reason = (
            f"No new crossover detected between the {short_window}-day and {long_window}-day SMAs, "
            "indicating a neutral hold position."
        )

    return {
        "signal": "SMA Crossover",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
