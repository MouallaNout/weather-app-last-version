import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# 🗣 Language Option
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("توقع طقس الغد بالذكاء الاصطناعي" if is_ar else "AI-Powered Tomorrow's Weather Forecast")

# 🌍 City Coordinates
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

# 📍 User Input
country = st.sidebar.selectbox("الدولة" if is_ar else "Country", list(city_coords.keys()))
city = st.sidebar.selectbox("المدينة" if is_ar else "City", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

if st.sidebar.button("ابدأ التوقع" if is_ar else "Start Forecast"):
    with st.spinner("🔄 جاري تحميل بيانات آخر سنتين..." if is_ar else "Fetching last 2 years of data..."):
        today = datetime.today().date()
        start_date = (today - timedelta(days=730)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
        )

        try:
            response = requests.get(api_url)
            data = response.json()
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"]
            })
        except:
            st.error("فشل تحميل البيانات!" if is_ar else "Failed to fetch data!")
            st.stop()

    # 🧹 Clean
    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = df[col].apply(lambda x: int(x + 0.5))

    # 🎯 Prepare Data
    look_back = 72  # 72 ساعة (3 أيام)
    data = df[["temperature"]].values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back].flatten())
        y.append(data[i + look_back][0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.warning("البيانات غير كافية للتدريب." if is_ar else "Not enough data to train.")
        st.stop()

    # 🔮 Use last 72 hours to predict tomorrow
    last_72_hours = data[-look_back:].flatten().reshape(1, -1)

    # 🤖 Models
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict(last_72_hours)[0]
        predictions[name] = pred

    ensemble_prediction = np.mean(list(predictions.values()))

    # ✅ Show Forecast
    st.markdown("## ☀️ " + ("توقع درجة حرارة الغد" if is_ar else "Tomorrow's Temperature Forecast"))
    st.success(
        f"📍 **{city}, {country}**\n\n"
        f"📅 {today + timedelta(days=1)}\n\n"
        f"🌡️ **{ensemble_prediction:.1f} °C** (تقدير متوسط)" if is_ar else
        f"📍 **{city}, {country}**\n\n"
        f"📅 {today + timedelta(days=1)}\n\n"
        f"🌡️ **{ensemble_prediction:.1f} °C** (Ensemble Estimate)"
    )

    # Optional: Breakdown
    st.markdown("### 🤖 " + ("تفاصيل النماذج" if is_ar else "Model Estimates"))
    st.write(pd.DataFrame(predictions, index=["Predicted Temp (°C)"]).T)
