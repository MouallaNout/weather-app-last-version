import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# إعداد اللغة
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
title = "توقع الطقس باستخدام الذكاء الاصطناعي" if is_ar else "AI-Based Weather Forecast"
st.title(title)

# اختيار الدولة والمدينة
city_coords = {
    "USA": {
        "New York": (40.71, -74.01),
        "Los Angeles": (34.05, -118.24)
    },
    "Saudi Arabia": {
        "Riyadh": (24.7136, 46.6753),
        "Jeddah": (21.4858, 39.1925)
    },
    "Germany": {
        "Berlin": (52.52, 13.4050),
        "Munich": (48.1351, 11.5820)
    }
}

st.sidebar.markdown("### 🌍 " + ("اختر الدولة والمدينة" if is_ar else "Select Country and City"))
country = st.sidebar.selectbox("الدولة" if is_ar else "Country", list(city_coords.keys()))
city = st.sidebar.selectbox("المدينة" if is_ar else "City", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

# زر التنبؤ
if st.sidebar.button("ابدأ التنبؤ" if is_ar else "Start Prediction"):
    with st.spinner("🔄 " + ("جاري تحميل البيانات..." if is_ar else "Fetching weather data...")):
        start_date = (date.today() - timedelta(days=730)).isoformat()
        end_date = date.today().isoformat()
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
            f"&timezone=auto"
        )

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"]
            })
        except Exception as e:
            st.error("فشل تحميل البيانات." if is_ar else f"Failed to fetch data: {e}")
            st.stop()

    # معالجة القيم المفقودة
    def fill_with_avg_of_neighbors(series):
        series = series.copy()
        for i in range(1, len(series) - 1):
            if pd.isna(series[i]) and not pd.isna(series[i - 1]) and not pd.isna(series[i + 1]):
                series[i] = (series[i - 1] + series[i + 1]) / 2
        return series

    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = fill_with_avg_of_neighbors(df[col])
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
        df[col] = df[col].apply(lambda x: int(x + 0.5))

    # النمذجة وتوقع الثلاث متغيرات
    look_back = 72
    hours_ahead = 24
    variables = ["temperature", "humidity", "wind_speed"]
    forecast_results = {}

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

    for var in variables:
        X, y = [], []
        data_arr = df[[var]].values
        for i in range(len(data_arr) - look_back):
            X.append(data_arr[i:i+look_back].flatten())
            y.append(data_arr[i+look_back][0])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            st.warning("البيانات غير كافية للتدريب." if is_ar else "Not enough data to train.")
            st.stop()

        X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)

        # تدريب النموذجين
        for model in models.values():
            model.fit(X_train, y_train)

        # التنبؤ بالساعات القادمة
        current_sequence = df[[var]].values[-look_back:].flatten().reshape(1, -1)
        hourly_preds = []
        for _ in range(hours_ahead):
            preds = [model.predict(current_sequence)[0] for model in models.values()]
            avg_pred = sum(preds) / len(preds)
            hourly_preds.append(avg_pred)
            current_sequence = np.append(current_sequence[:, 1:], [[avg_pred]], axis=1)

        forecast_results[var] = hourly_preds

    # أوقات الساعات
    start_time = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    hourly_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]

    # جدول التوقعات النهائي
    df_forecast = pd.DataFrame({
        "Time": hourly_times,
        "Temperature (°C)": forecast_results["temperature"],
        "Humidity (%)": forecast_results["humidity"],
        "Wind Speed (km/h)": forecast_results["wind_speed"]
    })

    # العرض
    st.subheader("🌤️ " + ("توقعات الطقس لكل ساعة غدًا" if is_ar else "Hourly Weather Forecast for Tomorrow"))
    st.markdown(f"📍 {city}, {country}")
    st.markdown(f"📅 {date.today() + timedelta(days=1)}")

    st.line_chart(df_forecast.set_index("Time")[["Temperature (°C)"]])
    st.line_chart(df_forecast.set_index("Time")[["Humidity (%)"]])
    st.line_chart(df_forecast.set_index("Time")[["Wind Speed (km/h)"]])

    st.dataframe(df_forecast.style.format({
        "Temperature (°C)": "{:.1f}",
        "Humidity (%)": "{:.0f}",
        "Wind Speed (km/h)": "{:.1f}"
    }))
