# GenAI Advisor

> **Educational AI-powered investment assistant for retail investors (offline capable).**

## Disclaimer

This tool is for educational purposes only and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions. 

## Overview

GenAI Advisor fetches equity data from Yahoo Finance, analyses using a hybrid **rule-based and ML-enhanced strategy engine**, generates **clear LLM explanations (Ollama Mistral)**, and presents results in a **Streamlit dashboard** for an educational, privacy-preserving investment analysis pipeline.

## Features

✅ **Data Ingestion:**  
- Yahoo Finance fetching with CSV caching for reproducibility and offline capabilities.  
- 10 years historical lookback for robust analysis.

✅ **Strategy Engine:**  
- Enhanced signals: SMA Crossover, RSI, MACD, Bollinger Bands, Stochastic Oscillator.  
- ML classifiers (Random Forest, Logistic Regression) with feature engineering, trained offline.  
- Signal outputs can now return BUY, HOLD, or SELL with refined thresholds.

✅ **Explanation Generator:**  
- Offline generation via **local Ollama Mistral**.  
- Uses structured prompts for clear, layperson-friendly, under 200-word outputs.

✅ **User Interface:**  
- **Streamlit app** for interactive exploration.  
- Portfolio-level overview with per-ticker drill-down.  
- Downloadable PDF analysis reports (charts, recommendations, explanations).  
- Educational signal explanations embedded in the interface.

✅ **Testing:**  
- Pytest tests covering ingestion, strategy signals, ML predictions, and explanation generation.

✅ **Ready for evaluation, submission, and further extension** (e.g., MariaDB storage, advanced interpretability).

---

## Usage

```bash
# Activate environment
conda activate genaiadvisor

# Launch the Streamlit app
streamlit run streamlit_app/main.py
