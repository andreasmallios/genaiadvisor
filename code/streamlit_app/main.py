import streamlit as st
import pandas as pd
import sys
import os
import time

# Ensure local imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.data_ingestion import fetch_ticker_data
from app.strategy_engine import generate_recommendation
from app.explanation_generator import generate_explanation
from app.logger import log_interaction

st.set_page_config(page_title="GenAI Advisor", layout="centered")
st.title("GenAI Advisor")

st.write("Enter a ticker symbol to receive an AI-powered investment insight with a clear explanation.")

ticker = st.text_input("Enter ticker symbol (e.g., MSFT):", value="MSFT")
run_analysis = st.button("Run Analysis")

if run_analysis:
    try:
        start_time = time.time()
        with st.spinner(f"Fetching data for {ticker}..."):
            df = fetch_ticker_data(ticker.upper())

        st.success("Data loaded successfully.")

        st.subheader("Recent Prices")
        st.line_chart(df["Close"])

        with st.spinner("Generating recommendation..."):
            recommendation = generate_recommendation(df, ticker.upper())

        st.info(f"**Recommendation:** {recommendation['recommendation']}")
        st.write(f"**Reason:** {recommendation['reason']}")
        st.write(f"**Date:** {recommendation['date']}")

        with st.spinner("Generating explanation with Mistral..."):
            explanation = generate_explanation(recommendation)

        latency_ms = int((time.time() - start_time) * 1000)

        st.success("Explanation generated:")
        st.write(explanation)

        # Log the interaction for evaluation tracking
        log_interaction(
            ticker.upper(),
            recommendation["recommendation"],
            explanation[:100] + "...",  # preview
            latency_ms
        )
        st.info(f"Analysis completed in {latency_ms} ms.")

    except Exception as e:
        st.error(f"Error: {e}")
