from app.strategy_engine.bollinger import bollinger_band_signal
from app.strategy_engine.stochastic import stochastic_signal
from app.strategy_engine.ml_classifier import fetch_ticker_data

def test_bollinger_stochastic_signals():
    df = fetch_ticker_data("MSFT")
    bollinger_result = bollinger_band_signal(df)
    stochastic_result = stochastic_signal(df)

    assert 'recommendation' in bollinger_result
    assert 'recommendation' in stochastic_result
    assert bollinger_result['recommendation'] in ['BUY', 'HOLD']
    assert stochastic_result['recommendation'] in ['BUY', 'HOLD']
    print("[TEST PASSED] Bollinger and Stochastic signals produce valid outputs.")

if __name__ == "__main__":
    test_bollinger_stochastic_signals()
