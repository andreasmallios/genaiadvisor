import pandas as pd

def bollinger_band_signal(df: pd.DataFrame, window=20, num_std=2):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['SMA'] + (num_std * df['STD'])
    df['Lower'] = df['SMA'] - (num_std * df['STD'])
    latest = df.iloc[-1]

    if latest['Close'] < latest['Lower']:
        recommendation = "BUY"
        reason = (
            f"Price {latest['Close']:.2f} crossed below lower Bollinger Band "
            f"{latest['Lower']:.2f}, indicating oversold conditions and a potential rebound."
        )
    elif latest['Close'] > latest['Upper']:
        recommendation = "SELL"
        reason = (
            f"Price {latest['Close']:.2f} crossed above upper Bollinger Band "
            f"{latest['Upper']:.2f}, indicating overbought conditions and a potential pullback."
        )
    else:
        recommendation = "HOLD"
        reason = (
            f"Price {latest['Close']:.2f} is within Bollinger Bands "
            f"{latest['Lower']:.2f} - {latest['Upper']:.2f}, indicating no clear signal."
        )

    return {
        "signal": "Bollinger Bands",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(latest.name.date())
    }
