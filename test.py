import yfinance as yf
import streamlit as st


df = yf.download("XAUUSD=X", period="7d", interval="1h")
print(df.tail())
