import streamlit as st
import pandas as pd
from model import arima_forecast  # Import fungsi dari model.py

# Judul aplikasi
st.title('Forecasting Pergerakan Kontainer')

# URL file CSV di Github (ganti dengan URL Anda)
url = "https://raw.githubusercontent.com/irhassha/trends/refs/heads/main/data/container_data.csv"

# Baca data dari URL
df = pd.read_csv(url)

# Baca data dari URL
df = pd.read_csv(url)

# Preprocessing data
df['Gate in'] = pd.to_datetime(df['Gate in'])  # Note the correction: 'Gate in' instead of 'Gate in'
df_agg = df.groupby([pd.Grouper(key='Gate in', freq='D'), 'Movement']).size().reset_index(name='Count')
df_rec = df_agg[df_agg['Movement'] == 'REC']
df_del = df_agg[df_agg['Movement'] == 'DEL']

# Input jumlah hari untuk forecasting
forecast_days = st.number_input("Jumlah hari untuk forecasting:", min_value=1, value=7)

# Tombol untuk menjalankan forecasting
if st.button('Jalankan Forecasting'):
    # Jalankan forecasting (ganti order ARIMA dengan nilai yang sesuai)
    order_rec = (5, 1, 0)
    order_del = (5, 1, 0)

    # Buat model ARIMA untuk REC
    model_rec = ARIMA(df_rec['Count'], order=order_rec)
    model_fit_rec = model_rec.fit()

    # Buat model ARIMA untuk DEL
    model_del = ARIMA(df_del['Count'], order=order_del)
    model_fit_del = model_del.fit()

    # Forecasting
    forecast_rec = model_fit_rec.forecast(steps=forecast_days)
    forecast_del = model_fit_del.forecast(steps=forecast_days)

    # Tampilkan hasil forecasting
    st.write("### Hasil Forecasting:")
    st.write("**REC:**")
    st.write(forecast_rec)
    st.write("**DEL:**")
    st.write(forecast_del)


        # Visualisasi (opsional)
        # ...
