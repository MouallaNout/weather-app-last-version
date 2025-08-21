import streamlit as st
import csv
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import date, timedelta, datetime
from astral import LocationInfo
from astral.sun import sun
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================== App ====================
lang = st.sidebar.selectbox("اللغة / Language", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("Weather forecasting system using machine learning" if not is_ar else "نظام التنبؤ بالطقس باستخدام التعلم الآلي")

# تحميل بيانات المدن
city_coords = {}
with open("worldcities.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        country = row["country"]
        city = row["city"]
        lat = float(row["lat"])
        lng = float(row["lng"])
        if country not in city_coords:
            city_coords[country] = {}
        city_coords[country][city] = (lat, lng)

st.sidebar.markdown("اختر الدولة والمدينة" if is_ar else "Select Country and City")
country = st.sidebar.selectbox("الدولة" if is_ar else "Country", list(city_coords.keys()))
city = st.sidebar.selectbox("المدينة" if is_ar else "City", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

st.sidebar.markdown("ماذا تريد أن يتم التنبؤ به؟" if is_ar else "Select what to predict")
all_vars = {
    ("Temperature" if not is_ar else "درجة الحرارة"): "temperature",
    ("Humidity" if not is_ar else "الرطوبة"): "humidity",
    ("Wind Speed" if not is_ar else "سرعة الرياح"): "wind_speed"
}
selected_display = st.sidebar.multiselect("", list(all_vars.keys()), default=list(all_vars.keys()))
selected_vars = [all_vars[d] for d in selected_display]

st.sidebar.markdown("Select units" if not is_ar else "اختر وحدات القياس")
unit_temp = st.sidebar.radio(
    "درجة الحرارة" if is_ar else "Temperature",
    ["°م" if is_ar else "°C", "°ف" if is_ar else "°F"],
    index=0
)
unit_wind = st.sidebar.radio(
    "سرعة الرياح" if is_ar else "Wind Speed",
    ["كم/سا" if is_ar else "km/h", "م/ث" if is_ar else "m/s"],
    index=0
)

# ================== جلب البيانات ====================
if st.sidebar.button("ابدأ التنبؤ" if is_ar else "Start Prediction"):
    start_date = (date.today() - timedelta(days=730)).isoformat()
    end_date = date.today().isoformat()
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
    )

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            st.error("فشل في الاتصال بواجهة الطقس." if is_ar else "Failed to connect to weather API.")
            st.stop()

        try:
            data = response.json()
        except ValueError:
            st.error("الاستجابة من API غير صالحة." if is_ar else "Invalid response from API.")
            st.stop()

        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["windspeed_10m"]
        })

    except Exception as e:
        st.error("حدث خطأ أثناء تحميل البيانات." if is_ar else f"An error occurred while fetching data: {e}")
        st.stop()

    # معالجة القيم المفقودة
    def fill_with_avg_of_neighbors(series):
        series = series.copy()
        for i in range(1, len(series) - 1):
            if pd.isna(series[i]) and not pd.isna(series[i - 1]) and not pd.isna(series[i + 1]):
                series[i] = (series[i - 1] + series[i + 1]) / 2
        return series

    for col in df.columns[1:]:
        df[col] = fill_with_avg_of_neighbors(df[col])
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    # ================== حساب أوقات الشروق والغروب ====================
    city_info = LocationInfo(city, country, "America/New_York", lat, lon)
    selected_day = date.today() + timedelta(days=1)
    sun_times = sun(city_info.observer, date=selected_day, tzinfo=city_info.timezone)

    sunrise_hour = sun_times["sunrise"].hour
    sunset_hour = sun_times["sunset"].hour

    # ================== تقسيم اليوم إلى نهار وليل ====================
    daytime_data = df[(df["datetime"].dt.hour >= sunrise_hour) & (df["datetime"].dt.hour < sunset_hour)]
    nighttime_data = df[(df["datetime"].dt.hour < sunrise_hour) | (df["datetime"].dt.hour >= sunset_hour)]

    # ================== حساب المتوسطات ====================
    day_avg_temp = daytime_data["temperature"].mean()
    night_avg_temp = nighttime_data["temperature"].mean()

    day_avg_wind = daytime_data["wind_speed"].mean()
    night_avg_wind = nighttime_data["wind_speed"].mean()

    day_avg_humidity = daytime_data["humidity"].mean()
    night_avg_humidity = nighttime_data["humidity"].mean()

    # ================== عرض المتوسطات ====================
    st.subheader("متوسطات الطقس لليوم التالي")
    st.markdown(f"**متوسط درجة الحرارة في النهار**: {day_avg_temp:.2f}°C")
    st.markdown(f"**متوسط درجة الحرارة في الليل**: {night_avg_temp:.2f}°C")
    st.markdown(f"**متوسط سرعة الرياح في النهار**: {day_avg_wind:.2f} {unit_wind}")
    st.markdown(f"**متوسط سرعة الرياح في الليل**: {night_avg_wind:.2f} {unit_wind}")
    st.markdown(f"**متوسط الرطوبة في النهار**: {day_avg_humidity:.2f}%")
    st.markdown(f"**متوسط الرطوبة في الليل**: {night_avg_humidity:.2f}%")

    # ================== رسم بياني ====================
    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"].dt.hour, df["temperature"], marker="o", label="Hourly Temperature")
    plt.axhline(y=day_avg_temp, color="orange", linestyle="--", label=f"Day Avg Temp = {day_avg_temp:.2f}°C")
    plt.axhline(y=night_avg_temp, color="blue", linestyle="--", label=f"Night Avg Temp = {night_avg_temp:.2f}°C")
    plt.title(f"Temperature for {selected_day}")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(24))
    plt.legend()
    st.pyplot()

    # ================== عرض البيانات ====================
    st.dataframe(df.style.format(precision=1))
