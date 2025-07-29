import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
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

    # معالجة القيم المفقودة بطريقة المتوسط بين الجارتين
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

    # تجهيز البيانات للنمذجة
    look_back = 72
    target = "temperature"
    X, y = [], []
    data_arr = df[[target]].values
    for i in range(len(data_arr) - look_back):
        X.append(data_arr[i:i+look_back].flatten())
        y.append(data_arr[i+look_back][0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.warning("البيانات غير كافية للتدريب." if is_ar else "Not enough data to train.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

    results = {}
    times = {}
    predictions = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        elapsed = time.time() - start
        mae = mean_absolute_error(y_test, pred)

        results[name] = mae
        times[name] = elapsed
        predictions.append(pred)

    # توقع الغد
    last_sequence = df[[target]].values[-look_back:].flatten().reshape(1, -1)
    tomorrow_preds = [model.predict(last_sequence)[0] for model in models.values()]
    tomorrow_temp = sum(tomorrow_preds) / len(tomorrow_preds)

    # عرض النتائج بشكل مبسط وإنساني
    st.success("✅ " + ("تم التنبؤ بنجاح!" if is_ar else "Prediction completed!"))
    st.markdown("---")

    st.subheader("🌤️ " + ("توقع درجة الحرارة ليوم الغد" if is_ar else "Tomorrow's Temperature Forecast"))
    st.markdown(f"📍 {city}, {country}")
    st.markdown(f"📅 {date.today() + timedelta(days=1)}")
    st.markdown(f"🌡️ **{tomorrow_temp:.1f}°C**")

    st.markdown("---")
    st.subheader("📈 " + ("أداء النماذج" if is_ar else "Model Performance"))
    perf_df = pd.DataFrame({
        "MAE": results,
        "Time (s)": times
    })
    st.dataframe(perf_df.style.format({"MAE": "{:.2f}", "Time (s)": "{:.2f}"}))

    st.bar_chart(perf_df["MAE"])
