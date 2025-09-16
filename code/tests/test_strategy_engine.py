import pandas as pd
import pytest
from app.data_ingestion import fetch_ticker_data
from app.strategy_engine.sma import sma_crossover_signal
from app.strategy_engine.rsi import rsi_signal
from app.strategy_engine.macd import macd_signal
from app.strategy_engine.engine import generate_combined_recommendation

@pytest.fixture
def msft_df():
    return fetch_ticker_data("MSFT")

def test_sma_signal(msft_df):
    result = sma_crossover_signal(msft_df)
    assert isinstance(result, dict)
    assert "recommendation" in result
    assert result["recommendation"] in ["BUY", "HOLD"]
    print(result)

def test_rsi_signal(msft_df):
    result = rsi_signal(msft_df)
    assert isinstance(result, dict)
    assert "recommendation" in result
    assert result["recommendation"] in ["BUY", "HOLD"]
    print(result)

def test_macd_signal(msft_df):
    result = macd_signal(msft_df)
    assert isinstance(result, dict)
    assert "recommendation" in result
    assert result["recommendation"] in ["BUY", "HOLD"]
    print(result)

def test_combined_recommendation(msft_df):
    result = generate_combined_recommendation(msft_df, "MSFT")
    assert isinstance(result, dict)
    assert "recommendation" in result
    assert result["recommendation"] in ["BUY", "HOLD"]
    assert "details" in result
    assert isinstance(result["details"], list)
    print(result)
