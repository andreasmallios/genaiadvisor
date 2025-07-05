from app.explanation_generator import generate_explanation

def test_generate_explanation_returns_string():
    sample_rec = {
        "ticker": "AAPL",
        "recommendation": "BUY",
        "reason": "Test reason",
        "date": "2025-07-05"
    }
    explanation = generate_explanation(sample_rec)
    assert isinstance(explanation, str)
    assert len(explanation) > 10
