import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("📈 XAU/USD Real-Time Dashboard")

# ตั้งช่วงเวลาที่จะดึงข้อมูล
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=30)

# ดึงราคาทองคำ (XAU/USD)
# ชื่อย่อใน Yahoo Finance คือ "XAUUSD=X"
df = yf.download("XAUUSD=X", start=start_date, end=end_date, interval="1h")

# แสดงข้อมูล
st.line_chart(df["Close"])
if not df.empty and "Close" in df.columns:
    st.write("Latest Price:", df["Close"].iloc[-1])
else:
    st.warning("No data available to display the latest price.")

