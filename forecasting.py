# forecasting.py

import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_stock_data_daily(ticker, period="1y"):
    """Fetch daily stock data for the given ticker and period."""
    data = yf.download(ticker, period=period, interval="1d")
    if data.empty:
        st.error(f"No data fetched for ticker: {ticker}")
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        new_columns = {col: col.split(',')[0] for col in data.columns if isinstance(col, str)}
        data.rename(columns=new_columns, inplace=True)
    data = data.dropna(subset=['Close'])
    data['Pct_Change'] = data['Close'].pct_change() * 100
    data.dropna(subset=['Pct_Change'], inplace=True)
    data.reset_index(inplace=True)
    return data

def forecast_pct_change(ticker, forecast_days=3, period="1y"):
    """Use Prophet to forecast future stock percentage change."""
    data = fetch_stock_data_daily(ticker, period)
    if data.empty:
        return None
    # Prepare data for Prophet
    df = data[['Date', 'Pct_Change']].rename(columns={'Date': 'ds', 'Pct_Change': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast_future = forecast.tail(forecast_days)
    return forecast_future
