import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================== App ====================
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("AI-Based Weather Forecast" if not is_ar else "توقع الطقس باستخدام الذكاء الاصطناعي")

import json

with open("world_cities_full.json", "r", encoding="utf-8") as f:
    city_coords = json.load(f),
    "Turkey": {
        "Istanbul": (41.0082, 28.9784),
        "Ankara": (39.9208, 32.8541),
        "Izmir": (38.4192, 27.1287)
    },
    "Egypt": {
        "Cairo": (30.0444, 31.2357),
        "Alexandria": (31.2001, 29.9187),
        "Giza": (30.0131, 31.2089)
    },
    "United Kingdom": {
        "London": (51.5074, -0.1278),
        "Manchester": (53.4808, -2.2426),
        "Birmingham": (52.4862, -1.8904)
    },
    "UAE": {
        "Dubai": (25.2048, 55.2708),
        "Abu Dhabi": (24.4539, 54.3773),
        "Sharjah": (25.3463, 55.4209)
    },
    "India": {
        "New Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946)
    },
    "Jordan": {
        "Amman": (31.9539, 35.9106),
        "Irbid": (32.5569, 35.8473),
        "Zarqa": (32.0728, 36.0880)
    },
    "Lebanon": {
        "Beirut": (33.8938, 35.5018),
        "Tripoli": (34.4333, 35.8333),
        "Sidon": (33.5606, 35.3758)
    },
    "Morocco": {
        "Casablanca": (33.5731, -7.5898),
        "Rabat": (34.0209, -6.8416),
        "Marrakesh": (31.6295, -7.9811)
    },
    "Algeria": {
        "Algiers": (36.7538, 3.0588),
        "Oran": (35.6971, -0.6308),
        "Constantine": (36.3650, 6.6147)
    },
    "Qatar": {
        "Doha": (25.276987, 51.520008)
    },
    "Kuwait": {
        "Kuwait City": (29.3759, 47.9774)
    },
    "Oman": {
        "Muscat": (23.5880, 58.3829)
    },
    "Bahrain": {
        "Manama": (26.2285, 50.5860)
    }
}

st.sidebar.markdown("### 🌍 " + ("اختر الدولة والمدينة" if is_ar else "Select Country and City"))
country = st.sidebar.selectbox("Country / الدولة", list(city_coords.keys()))
city = st.sidebar.selectbox("City / المدينة", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

st.sidebar.markdown("### 🔧 " + ("ماذا تريد أن يتم التنبؤ به؟" if is_ar else "Select what to predict"))
all_vars = {
    "🌡️ " + ("Temperature" if not is_ar else "درجة الحرارة"): "temperature",
    "💧 " + ("Humidity" if not is_ar else "الرطوبة"): "humidity",
    "🌬️ " + ("Wind Speed" if not is_ar else "سرعة الرياح"): "wind_speed"
}
selected_display = st.sidebar.multiselect("", list(all_vars.keys()), default=list(all_vars.keys()))
selected_vars = [all_vars[d] for d in selected_display]

st.sidebar.markdown("### 🔢 " + ("Select units" if not is_ar else "اختر وحدات القياس"))
unit_temp = st.sidebar.radio("Temperature", ["C", "F"], index=0)
unit_wind = st.sidebar.radio("Wind Speed", ["km/h", "m/s"], index=0)

if st.sidebar.button("Start Prediction" if not is_ar else "ابدأ التنبؤ"):
    start_date = (date.today() - timedelta(days=730)).isoformat()
    end_date = date.today().isoformat()
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
    )

    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["windspeed_10m"]
    })

    def fill_with_avg_of_neighbors(series):
        series = series.copy()
        for i in range(1, len(series) - 1):
            if pd.isna(series[i]) and not pd.isna(series[i - 1]) and not pd.isna(series[i + 1]):
                series[i] = (series[i - 1] + series[i + 1]) / 2
        return series

    for col in df.columns[1:]:
        df[col] = fill_with_avg_of_neighbors(df[col])
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    look_back = 72
    hours_ahead = 24
    forecast_results = {}

    for var in selected_vars:
        X, y = [], []
        data_arr = df[[var]].values
        for i in range(len(data_arr) - look_back):
            X.append(data_arr[i:i+look_back].flatten())
            y.append(data_arr[i+look_back][0])
        X, y = np.array(X), np.array(y)

        X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)

        current_sequence = df[[var]].values[-look_back:].flatten().reshape(1, -1)
        hourly_preds = []
        for _ in range(hours_ahead):
            pred = model.predict(current_sequence)[0]
            hourly_preds.append(pred)
            current_sequence = np.append(current_sequence[:, 1:], [[pred]], axis=1)

        forecast_results[var] = hourly_preds

    start_time = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    hourly_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]
    df_forecast = pd.DataFrame({"Time": hourly_times})

    if "temperature" in forecast_results:
        temp = forecast_results["temperature"]
        if unit_temp == "F":
            temp = [(t * 9/5) + 32 for t in temp]
        df_forecast[f"Temperature ({unit_temp})"] = temp

    if "humidity" in forecast_results:
        df_forecast["Humidity (%)"] = forecast_results["humidity"]

    if "wind_speed" in forecast_results:
        wind = forecast_results["wind_speed"]
        if unit_wind == "m/s":
            wind = [w / 3.6 for w in wind]
        df_forecast[f"Wind Speed ({unit_wind})"] = wind

    def plot_line_chart(df, column, title):
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df[column], marker='o')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Hourly Weather Forecast for Tomorrow" if not is_ar else "توقعات الطقس لكل ساعة غدًا")
    st.markdown(f"📍 {city}, {country}")
    st.markdown(f"📅 {date.today() + timedelta(days=1)}")

    for col in df_forecast.columns:
        if col != "Time":
            label = col.split(" (")[0]
            emoji = "🌡️" if "Temp" in col else "💧" if "Humidity" in col else "🌬️"
            title = emoji + " " + (f"تغير {label}" if is_ar else f"{label} Throughout the Day")
            plot_line_chart(df_forecast, col, title)

    st.dataframe(df_forecast.style.format(precision=1))
