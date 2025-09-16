from app.strategy_engine.engine import generate_combined_recommendation
from app.strategy_engine.ml_classifier import fetch_ticker_data

df = fetch_ticker_data("MSFT")
result = generate_combined_recommendation(df, "MSFT")
print(result)
