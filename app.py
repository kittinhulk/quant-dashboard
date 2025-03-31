import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("📈 Real-Time Quant Dashboard (Mocked Jim Simons Style)")

# Simulated data
price = 1800 + np.random.randn(100).cumsum()
df = pd.DataFrame({'XAU/USD': price})

st.line_chart(df)

# Real-time update simulation
if st.button("📡 Refresh Data"):
    st.rerun()

