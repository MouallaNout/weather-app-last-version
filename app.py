import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import date, timedelta, datetime
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

            df = pd.DataFr
