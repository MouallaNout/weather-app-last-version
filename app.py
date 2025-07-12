import streamlit as st
import pandas as pd
from utils.model_helper import load_model, generate_dummy_input

# ---------------- Language Toggle ----------------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = (lang == "العربية")

# ---------------- Country & City ----------------
cities = {
    "USA": {"New York": (40.71, -74.01)},
    "Saudi Arabia": {"Riyadh": (24.71, 46.67)}
}
country = st.sidebar.selectbox("Select Country" if not is_ar else "اختر الدولة", list(cities.keys()))
city = st.sidebar.selectbox("Select City" if not is_ar else "اختر المدينة", list(cities[country].keys()))

# ---------------- Prediction ----------------
model = load_model("model/temperature_model.pkl")
X_input = generate_dummy_input()
prediction = model.predict(X_input)[0]

# ---------------- Display ----------------
st.title("Weather Prediction" if not is_ar else "توقع الطقس")
st.subheader(f"{'City' if not is_ar else 'المدينة'}: {city}")
st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'} 🌡️: {prediction:.1f}°C")
