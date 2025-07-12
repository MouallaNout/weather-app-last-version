import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.model_helper import load_model, generate_dummy_input

# ---------- Language Toggle ----------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"

# ---------- Country/City Dropdown ----------
world_cities = {
    "USA": {"New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24)},
    "UK": {"London": (51.51, -0.13), "Manchester": (53.48, -2.24)},
    "Saudi Arabia": {"Riyadh": (24.71, 46.67), "Jeddah": (21.54, 39.17)},
    "Egypt": {"Cairo": (30.05, 31.25), "Alexandria": (31.20, 29.92)},
    "India": {"Delhi": (28.61, 77.20), "Mumbai": (19.07, 72.87)},
    "Germany": {"Berlin": (52.52, 13.40), "Munich": (48.14, 11.58)},
    "France": {"Paris": (48.85, 2.35), "Lyon": (45.76, 4.84)},
    "Japan": {"Tokyo": (35.68, 139.69), "Osaka": (34.69, 135.50)}
}

country_label = "Select Country" if not is_ar else "اختر الدولة"
city_label = "Select City" if not is_ar else "اختر المدينة"

country = st.sidebar.selectbox(country_label, list(world_cities.keys()))
city = st.sidebar.selectbox(city_label, list(world_cities[country].keys()))
lat, lon = world_cities[country][city]

# ---------- Prediction Button ----------
if st.sidebar.button("Predict" if not is_ar else "تنفيذ التوقع"):

    # Load the model
    model = load_model("model/temperature_model.pkl")

    # Generate dummy input (real use: pass recent weather data)
    X_input = generate_dummy_input()

    # Run prediction
    prediction = model.predict(X_input)[0]
    prediction = round(float(prediction), 1)

    # ---------- Display Results ----------
    st.title("Weather Prediction" if not is_ar else "توقع الطقس")
    st.subheader(f"{'City' if not is_ar else 'المدينة'}: {city}")
    st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'}: 🌡️ {prediction}°C")

    # Footer
    st.markdown("---")
    st.caption("Powered by Streamlit • Weather ML Demo")
else:
    st.title("Weather Prediction" if not is_ar else "توقع الطقس")
    st.write("👈 Please choose a city and click Predict." if not is_ar else "👈 اختر مدينة واضغط على تنفيذ التوقع.")
