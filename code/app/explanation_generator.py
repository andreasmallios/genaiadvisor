import subprocess

def generate_explanation(recommendation: dict) -> str:
    """
    Generate an explanation using local 'ollama run mistral' CLI call.
    """
    prompt = (
        f"Generate a clear, concise investment explanation for a retail investor:\n\n"
        f"Ticker: {recommendation.get('ticker', '')}\n"
        f"Recommendation: {recommendation.get('recommendation', '')}\n"
        f"Reason: {recommendation.get('reason', '')}\n"
        f"Date: {recommendation.get('date', '')}\n\n"
        f"Keep it under 80 words, in plain British English."
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        output = result.stdout.decode("utf-8").strip()
        return output
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8").strip()
        return f"Error generating explanation: {error_msg}"

if __name__ == "__main__":
    sample_rec = {
        "ticker": "MSFT",
        "recommendation": "BUY",
        "reason": "50-day SMA is above the 200-day SMA, indicating bullish momentum.",
        "date": "2025-07-03"
    }
    explanation = generate_explanation(sample_rec)
    print(explanation)
