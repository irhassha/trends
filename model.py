import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(df_rec, df_del, forecast_days):
    """
    Melakukan forecasting jumlah kontainer REC dan DEL menggunakan ARIMA.

    Args:
        df_rec: DataFrame data REC.
        df_del: DataFrame data DEL.
        forecast_days: Jumlah hari untuk forecasting.

    Returns:
        Tuple: (forecast_rec, forecast_del) - hasil forecasting REC dan DEL.
    """

    # Tentukan order (p, d, q) model ARIMA (ganti dengan nilai yang sesuai)
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

    return forecast_rec, forecast_del
