import subprocess

def generate_explanation(recommendation: dict) -> str:
    """
    Generate an explanation using local 'ollama run mistral' CLI call.
    """

    # prompt = f"""
    #     You are GenAI Advisor, an educational investment explanation generator.

    #     Generate a clear, structured explanation in under 200 words using this format:
    #     1. **Summary:** One sentence on the recommendation (BUY/HOLD) and {recommendation.get('ticker', '')}.
    #     2. **Reason:** One sentence summarising the key signal(s): {recommendation.get('reason', '')}.
    #     3. **Action:** One sentence clarifying that this is for educational purposes and not financial advice.

    #     Use British English.
    #     """
    
    prompt = f"""
        You are GenAI Advisor, an educational investment explanation generator.

        Generate a clear, friendly explanation in **under 300 words**, targeted at a layperson with no technical background, using this structured format:

        1. **Summary:** State the recommendation ({recommendation.get('recommendation', '')}) for {recommendation.get('ticker', '')} in one clear sentence.

        2. **Reason:** This section should Explain the specific signals that led to the recommendation in a straightforward manner, referencing which signals were positive, which were neutral, and why they matter. Moreover, it should add another paragraph that simplifies the technical explanation into a more relatable summary, without treating the user in a patronising way. 

        The signals you may reference include:
        - SMA Crossover
        - RSI (Relative Strength Index)
        - MACD
        - Bollinger Bands
        - Stochastic Oscillator
        - ML Classifier output

        3. **Disclaimer:** One clear sentence stating that this analysis is for educational purposes only and does not constitute financial advice.

        **Output instructions:**
        - Return the **'Summary'** on its own line.
        - Return the **'Reason'** on its own line.
        - Return the **'Action'** on its own line.
        - Do **not** add anything else.
        - Use clear **British English** throughout.
        - Ensure output remains latin-1 compatible.

        Ensure the total output does not exceed **300 words**.
        """





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
