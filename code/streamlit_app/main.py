import streamlit as st
import pandas as pd
import sys
import os

# Ensure local imports work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.data_ingestion import fetch_ticker_data
from app.strategy_engine import generate_recommendation
from app.explanation_generator import generate_explanation

st.set_page_config(page_title="GenAI Advisor", layout="centered")
st.title("GenAI Advisor")

ticker = st.text_input("Enter ticker symbol (e.g., MSFT):", value="MSFT")
run_analysis = st.button("Run Analysis")

if run_analysis:
    try:
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

        st.success("Explanation generated:")
        st.write(explanation)

    except Exception as e:
        st.error(f"Error: {e}")
