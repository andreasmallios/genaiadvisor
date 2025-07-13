# test_ml_classifier.py

import pytest
import pandas as pd
from pathlib import Path
from genaiadvisor.code.app import ml_classifier

# Use a small ticker for speed
TICKER = "AAPL"

@pytest.fixture(scope="module")
def fetched_data():
    df = ml_classifier.fetch_ticker_data(TICKER, period="1y")
    assert not df.empty, "Fetched data is empty"
    return df

def test_compute_features(fetched_data):
    df_feat = ml_classifier.compute_features(fetched_data)
    assert all(col in df_feat.columns for col in ['SMA_10', 'RSI', 'MACD']), "Missing computed features"
    assert not df_feat.isna().any().any(), "NaNs present in feature dataframe"

def test_create_labels(fetched_data):
    df_feat = ml_classifier.compute_features(fetched_data)
    labels = ml_classifier.create_labels(df_feat)
    assert set(labels.dropna().unique()).issubset({0, 1, 2}), "Labels not in expected classes"

def test_model_prediction(fetched_data):
    signal = ml_classifier.predict_signal(fetched_data)
    assert signal["recommendation"] in {"BUY", "HOLD", "SELL"}, "Invalid prediction label"
    assert "TensorFlow model prediction" in signal["reason"], "Missing explanation"

