import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Title
st.title("Container Yard Analysis App")

# Read the data directly from the repository file
data = pd.read_csv('container_data.csv', sep=';')
data['Gate in'] = pd.to_datetime(data['Gate in'])
data['Day'] = data['Gate in'].dt.day_name()

# Filter REC and DEL
rec_data = data[data['MOVEMENT'] == 'REC']
del_data = data[data['MOVEMENT'] == 'DEL']

st.header("1. Average Movement per Day")
option = st.selectbox("Choose Movement Type", ["REC", "DEL"])
if option == "REC":
    daily_avg = rec_data.groupby('Day').size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
else:
    daily_avg = del_data.groupby('Day').size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

st.bar_chart(daily_avg)
st.write("Daily Average:", daily_avg)

st.header("2. Average per Service (DAY 1 to DAY 7)")
service_option = st.selectbox("Choose Movement Type for Service", ["REC", "DEL"])
if service_option == "REC":
    grouped_data = rec_data.groupby(['SERVICE', 'VESSEL ID']).size()
else:
    grouped_data = del_data.groupby(['SERVICE', 'VESSEL ID']).size()

service_summary = grouped_data.groupby(level=0).mean().reset_index(name="Average Containers")
st.write(service_summary)

st.header("3. Forecasting REC and DEL")
forecast_option = st.selectbox("Choose Movement Type for Forecasting", ["REC", "DEL"])
forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=30, value=7)

if forecast_option == "REC":
    forecast_data = rec_data.groupby('Gate in').size().resample('D').sum()
else:
    forecast_data = del_data.groupby('Gate in').size().resample('D').sum()

# Fill missing dates with 0
forecast_data = forecast_data.fillna(0)

# Model fitting and forecasting
model = ExponentialSmoothing(forecast_data, seasonal="add", seasonal_periods=7)
model_fit = model.fit()
forecast = model_fit.forecast(forecast_days)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_data.index, forecast_data, label="Actual")
plt.plot(forecast.index, forecast, label="Forecast", linestyle="--")
plt.title(f"Forecasting {forecast_option} for {forecast_days} days")
plt.xlabel("Date")
plt.ylabel("Number of Containers")
plt.legend()
st.pyplot(plt)
