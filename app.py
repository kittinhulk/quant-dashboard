import yfinance as yf
import pandas as pd
import streamlit as st

import datetime

st.title("📈 XAU/USD Real-Time Dashboard")

# ช่วงเวลา: 30 วันย้อนหลัง
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=30)

# ดึงข้อมูลจาก Yahoo Finance
try:
    df = yf.download("GC=F", start=start_date, end=end_date, interval="1h")

    # เช็กว่ามีข้อมูลหรือไม่
    if df.empty:
        st.warning("⚠️ No data available. Try adjusting the date range or interval.")
    else:
        st.line_chart(df["Close"])
        st.metric("Latest Price", f'{df["Close"].iloc[-1]:,.2f} USD')
except Exception as e:
    st.error(f"Error loading data: {e}")
