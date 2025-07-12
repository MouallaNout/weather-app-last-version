import streamlit as st
import pandas as pd
import numpy as np
from utils.model_helper import load_model, generate_dummy_input
from countryinfo import CountryInfo
import pycountry

# ------------------- Language Toggle -------------------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = (lang == "العربية")

# ------------------- Country Dropdown -------------------
all_countries = [country.name for country in pycountry.countries]
country_name = st.sidebar.selectbox("Select Country" if not is_ar else "اختر الدولة", sorted(all_countries))

# ------------------- Province/State Dropdown -------------------
try:
    country_info = CountryInfo(country_name)
    provinces = country_info.provinces()
except:
    provinces = []

if provinces:
    city_name = st.sidebar.selectbox("Select State/Province" if not is_ar else "اختر المنطقة/الولاية", sorted(provinces))
else:
    city_name = st.sidebar.text_input("Enter City Name" if not is_ar else "أدخل اسم المدينة")

# ------------------- Prediction Button -------------------
if st.button("Start Prediction" if not is_ar else "ابدأ التوقع"):
    # Load model
    model = load_model("model/temperature_model.pkl")
    X_input = generate_dummy_input()
    prediction = model.predict(X_input)[0]

    # Show result
    st.title("Weather Prediction" if not is_ar else "توقع الطقس")
    st.subheader(f"{city_name}, {country_name}")
    st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'}: 🌡️ {round(prediction, 1)}°C")

    # Footer
    st.markdown("---")
    st.caption("Powered by Streamlit • Weather ML Demo")
