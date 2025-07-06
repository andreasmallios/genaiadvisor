from app.strategy_engine.sma import sma_crossover_signal
from app.strategy_engine.rsi import rsi_signal
from app.strategy_engine.macd import macd_signal
from app.strategy_engine.bollinger import bollinger_band_signal
from app.strategy_engine.stochastic import stochastic_signal
from app.strategy_engine.ml_classifier import predict_signal

def generate_combined_recommendation(df, ticker):
    signals = [
        sma_crossover_signal(df),
        rsi_signal(df),
        macd_signal(df),
        bollinger_band_signal(df),
        stochastic_signal(df),
        predict_signal(df)  # ML classifier signal
    ]

    buy_count = sum(1 for s in signals if s['recommendation'] == "BUY")
    if buy_count >= 1:
        final_recommendation = "BUY"
        reason = f"{buy_count} out of {len(signals)} signals indicate BUY."
    else:
        final_recommendation = "HOLD"
        reason = "No signals strongly indicate a buy condition."

    return {
        "ticker": ticker,
        "recommendation": final_recommendation,
        "reason": reason,
        "date": signals[0]["date"],
        "details": signals
    }
