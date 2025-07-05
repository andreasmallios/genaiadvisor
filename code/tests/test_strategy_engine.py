from app.strategy_engine import generate_recommendation
from app.data_ingestion import fetch_ticker_data

def test_generate_recommendation_output():
    df = fetch_ticker_data("AAPL")
    rec = generate_recommendation(df, "AAPL")
    assert isinstance(rec, dict)
    assert "recommendation" in rec
    assert rec["recommendation"] in ["BUY", "HOLD"]
