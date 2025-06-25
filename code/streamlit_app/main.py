import streamlit as st

st.title("GenAI Advisor")

symbol = st.text_input("Enter stock symbol:")

if st.button("Run Demo"):
    st.write("Fetching data for:", symbol)
    # Simulate response for now
    st.write({
        "recommendation": "BUY",
        "confidence": 0.82,
        "explanation": "The P/E ratio and ML signals suggest strong performance."
    })
