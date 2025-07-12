import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# ------------------- Language Toggle -------------------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = (lang == "العربية")

# ------------------- Country/City Dropdown -------------------
cities = {
    "USA": {"New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24)},
    "Saudi Arabia": {"Riyadh": (24.71, 46.67), "Jeddah": (21.54, 39.17)}
}

country = st.sidebar.selectbox("Select Country" if not is_ar else "اختر الدولة", list(cities.keys()))
city = st.sidebar.selectbox("Select City" if not is_ar else "اختر المدينة", list(cities[country].keys()))
lat, lon = cities[country][city]

# ------------------- Fake Model (replace with real one) -------------------
# Here we'll simulate an ML prediction (e.g., using the past 24 hours average temperature)
@st.cache_data
def get_prediction():
    # Replace with your ML prediction
    return round(25 + np.random.randn(), 1)

prediction = get_prediction()

# ------------------- Display Results -------------------
st.title("Weather Prediction" if not is_ar else "توقع الطقس")
st.subheader(f"{'City' if not is_ar else 'المدينة'}: {city}")
st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'}: 🌡️ {prediction}°C")

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Powered by Streamlit • Weather ML Demo")
