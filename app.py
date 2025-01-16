import streamlit as st
import pandas as pd
from model import arima_forecast  # Import fungsi dari model.py

# Judul aplikasi
st.title('Forecasting Pergerakan Kontainer')

# Upload file CSV
uploaded_file = st.file_uploader("Upload file data kontainer (CSV)", type="csv")

if uploaded_file is not None:
    # Baca data
    df = pd.read_csv(uploaded_file)

    # Preprocessing data (sama seperti di contoh sebelumnya)
    df['Gate In'] = pd.to_datetime(df['Gate In'])
    df_agg = df.groupby([pd.Grouper(key='Gate In', freq='D'), 'Movement']).size().reset_index(name='Count')
    df_rec = df_agg[df_agg['Movement'] == 'REC']
    df_del = df_agg[df_agg['Movement'] == 'DEL']

    # Input jumlah hari untuk forecasting
    forecast_days = st.number_input("Jumlah hari untuk forecasting:", min_value=1, value=7)

    # Tombol untuk menjalankan forecasting
    if st.button('Jalankan Forecasting'):
        # Jalankan forecasting
        forecast_rec, forecast_del = arima_forecast(df_rec, df_del, forecast_days)

        # Tampilkan hasil forecasting
        st.write("### Hasil Forecasting:")
        st.write("**REC:**")
        st.write(forecast_rec)
        st.write("**DEL:**")
        st.write(forecast_del)

        # Visualisasi (opsional)
        # ...
