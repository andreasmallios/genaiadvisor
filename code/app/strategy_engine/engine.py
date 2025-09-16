from app.strategy_engine.sma import sma_crossover_signal
from app.strategy_engine.rsi import rsi_signal
from app.strategy_engine.macd import macd_signal
from app.strategy_engine.bollinger import bollinger_band_signal
from app.strategy_engine.stochastic import stochastic_signal
from app.strategy_engine.ml_classifier import compute_features

# Force TensorFlow to run on CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model
from pathlib import Path
import joblib
import numpy as np

from tensorflow.keras import backend as K
import gc

# Load TensorFlow model and scaler once
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
tf_model = load_model(MODEL_DIR / "tf_model.h5")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


def tensorflow_classifier_signal(df):
    features = compute_features(df).iloc[[-1]][[
        'SMA_10', 'SMA_50', 'SMA_diff', 'RSI', 'MACD',
        'Return_1d', 'Return_5d', 'Return_10d', 'Volume'
    ]]
    X_scaled = scaler.transform(features)
    prediction = tf_model.predict(X_scaled)
    label = int(np.argmax(prediction)) - 1  # Convert [0,1,2] -> [-1,0,1]

    # Clean up TF memory after prediction
    K.clear_session()
    gc.collect()

    if label == 1:
        recommendation = "BUY"
        reason = "TensorFlow model predicts positive price movement."
    elif label == -1:
        recommendation = "SELL"
        reason = "TensorFlow model predicts negative price movement."
    else:
        recommendation = "HOLD"
        reason = "TensorFlow model predicts neutral price movement."

    return {
        "signal": "ML Classifier",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(df.index[-1].date())
    }



def generate_combined_recommendation(df, ticker):
    signals = [
        sma_crossover_signal(df),
        rsi_signal(df),
        macd_signal(df),
        bollinger_band_signal(df),
        stochastic_signal(df),
        tensorflow_classifier_signal(df)
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
        total_weight += w

    normalised_score = score / total_weight if total_weight > 0 else 0

    if normalised_score >= 0.3:
        final_recommendation = "BUY"
        reason = f"Weighted signals support a BUY with a confidence score of {normalised_score:.2f}."
    elif normalised_score <= -0.3:
        final_recommendation = "SELL"
        reason = f"Weighted signals support a SELL with a confidence score of {normalised_score:.2f}."
    else:
        final_recommendation = "HOLD"
        reason = f"Signals are mixed; suggesting HOLD with a confidence score of {normalised_score:.2f}."

    return {
        "ticker": ticker,
        "recommendation": final_recommendation,
        "reason": reason,
        "date": signals[0]["date"],
        "details": signals,
        "score": normalised_score
    }

# ===
# STRATEGY ENGINE
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
# ===