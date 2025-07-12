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
        predict_signal(df)
    ]

    weights = {
        "SMA Crossover": 0.2,
        "RSI": 0.15,
        "MACD": 0.2,
        "Bollinger Bands": 0.15,
        "Stochastic Oscillator": 0.15,
        "ML Classifier": 0.15
    }

    score = 0
    total_weight = 0

    for s in signals:
        w = weights.get(s["signal"], 0.1)
        if s["recommendation"] == "BUY":
            score += 1 * w
        elif s["recommendation"] == "SELL":
            score -= 1 * w
        # HOLD = 0, no score change
        total_weight += w

    normalized_score = score / total_weight if total_weight > 0 else 0

    if normalized_score >= 0.3:
        final_recommendation = "BUY"
        reason = f"Weighted signals support a BUY with a confidence score of {normalized_score:.2f}."
    elif normalized_score <= -0.3:
        final_recommendation = "SELL"
        reason = f"Weighted signals support a SELL with a confidence score of {normalized_score:.2f}."
    else:
        final_recommendation = "HOLD"
        reason = f"Signals are mixed; suggesting HOLD with a confidence score of {normalized_score:.2f}."

    return {
        "ticker": ticker,
        "recommendation": final_recommendation,
        "reason": reason,
        "date": signals[0]["date"],
        "details": signals,
        "score": normalized_score
    }
