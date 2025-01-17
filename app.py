import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Title
st.title("Container Yard Analysis App")

# Read the data directly from the repository file
data = pd.read_csv('container_data.csv', sep=';')
data['Gate in'] = pd.to_datetime(data['Gate in'], format='%d/%m/%Y %H:%M', errors='coerce')
data['Day'] = data['Gate in'].dt.day_name()
data['Day Number'] = (data['Gate in'] - data['Gate in'].min()).dt.days + 1

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

st.header("2. Total and Average per Service (DAY 1 to DAY 7)")
service_option = st.selectbox("Choose Movement Type for Service", ["REC", "DEL"])
if service_option == "REC":
    # Group by SERVICE and Voyage
    grouped = rec_data.groupby(['SERVICE', 'VOYAGE']).size()

    # Sum containers per Service
    total_containers_per_service = grouped.groupby('SERVICE').sum()

    # Count unique Voyage combinations per Service
    unique_voyages_per_service = grouped.groupby('SERVICE').size()

    st.write("Total Containers Per Service:")
    st.write(total_containers_per_service)

    st.write("Unique Voyage Combinations Per Service:")
    st.write(unique_voyages_per_service)

    # Apply outlier filtering based on standard deviation
    filtered_data = grouped.groupby('SERVICE').apply(lambda x: x[(x >= x.mean() - 2 * x.std()) & (x <= x.mean() + 2 * x.std())])

    # Group by SERVICE, Day Number, and Voyage
    filtered_grouped = rec_data[rec_data['VOYAGE'].isin(filtered_data.index.get_level_values('VOYAGE'))]
    filtered_grouped = filtered_grouped.groupby(['SERVICE', 'Day Number', 'VOYAGE']).size()

    # Sum containers per Day Number per Service
    total_per_day = filtered_grouped.groupby(['SERVICE', 'Day Number']).sum()

    # Count Voyage combinations per Day Number per Service
    voyage_count_per_day = filtered_grouped.groupby(['SERVICE', 'Day Number']).size()

    # Calculate average per day per service
    average_per_day_per_service = (total_per_day / voyage_count_per_day).unstack(fill_value=0)
else:
    # Group by SERVICE and Voyage
    grouped = del_data.groupby(['SERVICE', 'VOYAGE']).size()

    # Sum containers per Service
    total_containers_per_service = grouped.groupby('SERVICE').sum()

    # Count unique Voyage combinations per Service
    unique_voyages_per_service = grouped.groupby('SERVICE').size()

    st.write("Total Containers Per Service:")
    st.write(total_containers_per_service)

    st.write("Unique Voyage Combinations Per Service:")
    st.write(unique_voyages_per_service)

    # Apply outlier filtering based on standard deviation
    filtered_data = grouped.groupby('SERVICE').apply(lambda x: x[(x >= x.mean() - 2 * x.std()) & (x <= x.mean() + 2 * x.std())])

    # Group by SERVICE, Day Number, and Voyage
    filtered_grouped = del_data[del_data['VOYAGE'].isin(filtered_data.index.get_level_values('VOYAGE'))]
    filtered_grouped = filtered_grouped.groupby(['SERVICE', 'Day Number', 'VOYAGE']).size()

    # Sum containers per Day Number per Service
    total_per_day = filtered_grouped.groupby(['SERVICE', 'Day Number']).sum()

    # Count Voyage combinations per Day Number per Service
    voyage_count_per_day = filtered_grouped.groupby(['SERVICE', 'Day Number']).size()

    # Calculate average per day per service
    average_per_day_per_service = (total_per_day / voyage_count_per_day).unstack(fill_value=0)

# Keep only Day 1 to Day 7
average_per_day_per_service = average_per_day_per_service.iloc[:, :7]

st.write("Average Containers Per Service (Day 1 to Day 7):")
st.write(average_per_day_per_service)

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

# Generate future dates for the forecast
last_date = forecast_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_data.index, forecast_data, label="Actual")
plt.plot(future_dates, forecast, label="Forecast", linestyle="--")
plt.title(f"Forecasting {forecast_option} for {forecast_days} days")
plt.xlabel("Date")
plt.ylabel("Number of Containers")
plt.legend()
st.pyplot(plt)
