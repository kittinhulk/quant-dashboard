import yfinance as yf
import streamlit as st


df = yf.download("GC=F", period="7d", interval="1h")
print(df.tail())
