import os
from app.data_ingestion import fetch_ticker_data

def test_fetch_ticker_data_creates_csv():
    ticker = "AAPL"
    df = fetch_ticker_data(ticker)
    assert not df.empty
    assert "Close" in df.columns
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"{ticker}.csv")
    assert os.path.exists(csv_path)
