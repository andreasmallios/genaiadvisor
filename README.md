
# GenAI Advisor

> **Educational AI-powered investment assistant for retail investors (offline capable).**

## Disclaimer

This tool is for educational purposes only and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

## Overview

GenAI Advisor is a powerful educational tool designed to assist retail investors by providing an AI-powered investment analysis pipeline. It fetches equity data from Yahoo Finance, analyzes it using a hybrid **rule-based and ML-enhanced strategy engine**, generates **clear LLM explanations (Ollama Mistral)**, and presents results in an intuitive **Streamlit dashboard** for privacy-preserving, offline investment analysis.

## Features

### ✅ **Data Ingestion:**
- Yahoo Finance fetching with **CSV caching** for reproducibility and **offline capabilities**.
- 10 years of historical data for robust analysis and trend prediction.

### ✅ **Strategy Engine:**
- Enhanced signals: **SMA Crossover**, **RSI**, **MACD**, **Bollinger Bands**, **Stochastic Oscillator**.
- **ML classifiers** (Random Forest, Logistic Regression) with feature engineering, trained offline.
- **Refined thresholds** for recommendations (BUY, HOLD, SELL) to improve decision accuracy.

### ✅ **Explanation Generator:**
- **Offline explanation generation** via **local Ollama Mistral**.
- Uses structured prompts for clear, layperson-friendly, **under 200-word outputs** to provide concise reasons behind investment recommendations.

### ✅ **User Interface:**
- **Streamlit app** for interactive exploration of portfolio and individual ticker analysis.
- **Portfolio-level overview** with per-ticker drill-down.
- **Downloadable PDF analysis reports** containing charts, recommendations, and detailed explanations.
- **Educational signal explanations** embedded within the interface to help users understand the analysis.

### ✅ **Testing:**
- **Pytest tests** covering data ingestion, strategy signals, machine learning predictions, and explanation generation.
- Automated testing to ensure the robustness of data handling and analysis results.

### ✅ **Evaluation and Extendability:**
- **Ready for evaluation** and **submission**.
- Future extensions may include **MariaDB storage**, **advanced interpretability features**, and **further fine-tuning** of machine learning models.

---

## Usage

### Running the App

1. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app/main.py
   ```

2. The app will open in your browser, where you can interact with the portfolio overview, detailed ticker analysis, and explanations.

### Generate Analysis Report

- Select a company from the **Ticker Analysis** section.
- Click **Run Detailed Analysis** to generate recommendations, visualizations, and downloadable reports.

---

## Example Output

The **GenAI Advisor** app will provide you with:
- **Portfolio Overview** with general recommendations for each ticker.
- **Detailed Ticker Analysis** including historical data, signal details, and an explanation of each recommendation.
- **PDF Report** which can be downloaded for further reference.

---

## Future Work

- **Advanced Interpretability**: Exploring deeper model interpretability using SHAP and other techniques for clearer insights into the machine learning predictions.
- **Data Storage Integration**: Adding support for **MariaDB** or similar storage solutions to persist data and analyses.
- **User Customization**: Allow users to adjust model parameters, thresholds, and strategy signals for personalized recommendations.
