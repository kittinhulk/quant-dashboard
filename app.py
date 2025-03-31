
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- GPU Setup ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("🔮 LSTM Forecast Dashboard: Gold Futures (GC=F)")

days_to_predict = st.slider("Select number of days to forecast:", min_value=1, max_value=60, value=30)
future_steps = days_to_predict * 24

with st.spinner("Loading and processing data..."):
    df = yf.download("GC=F", period="max", interval="1h")
    df.dropna(inplace=True)
    df['Date'] = df.index
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, window_size=24):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, window_size=24)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

with st.spinner("Training LSTM model..."):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

# ---------------- Evaluate on Test Set ----------------
y_pred_test_scaled = model.predict(X_test)
y_pred_test = scaler.inverse_transform(y_pred_test_scaled)
y_actual_test = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_actual_test, y_pred_test))
mae = mean_absolute_error(y_actual_test, y_pred_test)

st.subheader("📊 Test Set Evaluation")
st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**MAE:** {mae:,.2f}")

# ---------------- Forecast Future ----------------
st.subheader(f"📈 Forecast for next {days_to_predict} days ({future_steps} hours)")
last_input = scaled_data[-24:]
future_predictions = []
timestamps = [df['Date'].iloc[-1] + pd.Timedelta(hours=i+1) for i in range(future_steps)]

for i in range(future_steps):
    pred_scaled = model.predict(last_input.reshape(1, 24, 1), verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    future_predictions.append(pred)
    last_input = np.vstack([last_input[1:], pred_scaled])

forecast_df = pd.DataFrame({'datetime': timestamps, 'forecast_price': future_predictions})
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast as CSV", data=csv, file_name='gold_forecast.csv', mime='text/csv')

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(timestamps, future_predictions, label='Forecasted Price', color='orange')
ax.set_title(f'Gold Futures (GC=F) Forecast - Next {days_to_predict} Days')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.grid(True)
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)
