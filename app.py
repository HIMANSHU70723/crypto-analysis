import streamlit as st
import numpy as np
import joblib

model = joblib.load("liquidity_model.pkl")

st.title("📊 Crypto Liquidity Predictor")

price = st.number_input("Price", value=100.0)
change_1h = st.number_input("1h % Change", value=0.5)
change_24h = st.number_input("24h % Change", value=2.0)
change_7d = st.number_input("7d % Change", value=10.0)
market_cap = st.number_input("Market Cap", value=1e9)
volume_24h = st.number_input("24h Volume", value=5e8)

price_change_24h = price * change_24h
price_volatility = price * (change_7d / 7)
volume_ratio = volume_24h / (market_cap + 1)
log_volume = np.log1p(volume_24h)
log_mkt_cap = np.log1p(market_cap)
price_to_mcap = price / (market_cap + 1)
daily_volatility_pct = (change_7d / 7) * 100

input_data = np.array([[price, change_1h, change_24h, change_7d,
                        price_change_24h, price_volatility,
                        volume_ratio, log_volume, log_mkt_cap,
                        price_to_mcap, daily_volatility_pct]])

if st.button("Predict Liquidity"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔮 Predicted Liquidity (volume_ratio): {prediction:.4f}")