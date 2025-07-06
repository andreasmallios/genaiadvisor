import streamlit as st
import pandas as pd
import sys
import os
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
from io import BytesIO

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.data_ingestion import fetch_ticker_data
from app.strategy_engine.engine import generate_combined_recommendation
from app.explanation_generator import generate_explanation
from app.logger import log_interaction

st.set_page_config(page_title="GenAI Advisor", layout="centered")

# LEGAL BANNER
st.markdown(
    """
    <div style="background-color:#f8d7da;padding:10px;border-radius:5px;border:1px solid #f5c6cb;">
    <strong>Disclaimer:</strong> This tool is for <strong>educational purposes only</strong> and does not constitute financial advice.
    Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.
    </div>
    """,
    unsafe_allow_html=True
)

st.title("GenAI Advisor")

# Load tickers
tickers_df = pd.read_csv("./data/tickers.csv")
names = tickers_df["Name"].tolist()
ticker_name_map = dict(zip(tickers_df["Name"], tickers_df["Symbol"]))

selected_name = st.selectbox("Select Company:", names)
ticker = ticker_name_map[selected_name]
genai_info = tickers_df.loc[tickers_df["Name"] == selected_name, "GenAI specific info"].values[0]
st.info(genai_info)

# EDUCATIONAL SIGNAL EXPLANATIONS
with st.expander("What do the signals mean?"):
    st.markdown("""
    - **SMA Crossover:** Compares short-term vs long-term average prices to identify momentum shifts.
    - **RSI (Relative Strength Index):** Measures if a stock is overbought or oversold.
    - **MACD (Moving Average Convergence Divergence):** Captures momentum by comparing two moving averages.
    - **Bollinger Bands:** Show volatility and potential price breakouts.
    - **Stochastic Oscillator:** Indicates potential reversals by comparing closing prices to price ranges.
    - **ML Classifier:** Uses machine learning to predict BUY/HOLD signals based on historical patterns.
    """)

run_analysis = st.button("Run Analysis")

def generate_pdf_with_chart(ticker, recommendation, explanation, latency_ms, df, signals_detail):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"GenAI Advisor Analysis: {ticker}", ln=True, align="C")
    pdf.ln(10)

    # Chart generation
    plt.figure(figsize=(6, 3))
    df["Close"].plot(title=f"{ticker} Closing Prices (10y)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    chart_path = f"{ticker}_chart_temp.png"
    with open(chart_path, "wb") as f:
        f.write(buf.read())
    pdf.image(chart_path, x=10, y=None, w=180)
    os.remove(chart_path)

    # Details
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=(
        f"Recommendation: {recommendation['recommendation']}\n\n"
        f"Reason: {recommendation['reason']}\n\n"
        f"Explanation: {explanation}\n\n"
        f"Latency: {latency_ms} ms\n\n"
        "Signal Details:\n" +
        "\n".join([f"{s['signal']}: {s['recommendation']} - {s['reason']}" for s in signals_detail])
    ))

    return pdf.output(dest='S').encode('latin-1')

if run_analysis:
    try:
        start_time = time.time()
        with st.spinner(f"Fetching 10y data for {ticker}..."):
            df = fetch_ticker_data(ticker, period="10y")
        st.success("Data loaded successfully.")
        st.subheader("Price Chart")
        st.line_chart(df["Close"])

        with st.spinner("Generating recommendation..."):
            recommendation = generate_combined_recommendation(df, ticker)

        st.info(f"**Recommendation:** {recommendation['recommendation']}")
        st.write(f"**Reason:** {recommendation['reason']}")
        st.write(f"**Date:** {recommendation['date']}")

        st.subheader("Signal Details")
        for signal in recommendation["details"]:
            st.write(f"- **{signal['signal']}**: {signal['recommendation']} – {signal['reason']}")

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(recommendation)

        latency_ms = int((time.time() - start_time) * 1000)
        st.success("Explanation generated.")
        st.write(explanation)
        st.info(f"Analysis completed in {latency_ms} ms.")

        log_interaction(
            ticker.upper(),
            recommendation["recommendation"],
            explanation[:100] + "...",
            latency_ms
        )

        pdf_bytes = generate_pdf_with_chart(ticker, recommendation, explanation, latency_ms, df, recommendation["details"])
        st.download_button(
            label="Download Analysis as PDF",
            data=pdf_bytes,
            file_name=f"{ticker}_GenAI_Analysis.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error: {e}")
