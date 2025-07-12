import streamlit as st
import numpy as np
from utils.model_helper import load_model, generate_dummy_input

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

# ------------------- Load and Run Model -------------------
model = load_model()
input_data = generate_dummy_input()  # replace this later with real inputs
prediction = model.predict(input_data)[0]

# ------------------- Display -------------------
st.title("Weather Prediction" if not is_ar else "توقع الطقس")
st.subheader(f"{'City' if not is_ar else 'المدينة'}: {city}")
st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'}: 🌡️ {round(prediction, 2)}°C")

st.markdown("---")
st.caption("Powered by Streamlit • Weather ML Demo")
