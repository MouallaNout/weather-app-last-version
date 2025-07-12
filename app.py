import streamlit as st
import pandas as pd
import numpy as np
from utils.model_helper import load_model, generate_dummy_input
from countryinfo import CountryInfo
import pycountry

# ------------------- اللغة -------------------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = (lang == "العربية")
lang_code = "ar" if is_ar else "en"

# ------------------- جميع الدول بأسمائها المحلية -------------------
def get_country_display_name(country):
    try:
        name_translations = pycountry.countries.get(name=country)
        if name_translations and hasattr(name_translations, 'translations'):
            return name_translations.translations.get(lang_code, country)
    except:
        pass
    return country

# قائمة الدول (عرض مترجم فقط)
all_countries = sorted([country.name for country in pycountry.countries])
translated_countries = [get_country_display_name(name) for name in all_countries]
country_map = dict(zip(translated_countries, all_countries))

# اختيار الدولة
display_country = st.sidebar.selectbox("اختر الدولة" if is_ar else "Select Country", translated_countries)
country_name = country_map[display_country]

# ------------------- عرض الولايات/المدن -------------------
try:
    country_info = CountryInfo(country_name)
    provinces = country_info.provinces()
except:
    provinces = []

if provinces:
    city_options = sorted(provinces)
    city_selected = st.sidebar.selectbox("اختر الولاية / المدينة" if is_ar else "Select State / City", city_options)

    # إظهار الاسم العربي أو الإنجليزي حسب اللغة
    if is_ar:
        city_name = city_selected  # نعرض الاسم كما هو
    else:
        city_name = city_selected  # نعرض الاسم كما هو (بدون تغيير حالياً)
else:
    city_name = st.sidebar.text_input("أدخل اسم المدينة" if is_ar else "Enter City Name")

# ------------------- زر التوقع -------------------
predict = st.sidebar.button("ابدأ التوقع" if is_ar else "Start Prediction")

# ------------------- النتيجة -------------------
if predict:
    model = load_model("model/temperature_model.pkl")
    X_input = generate_dummy_input()
    prediction = model.predict(X_input)[0]

    st.title("توقع الطقس" if is_ar else "Weather Prediction")
    st.subheader(f"{city_name}, {display_country}")
    st.write(f"{'درجة الحرارة المتوقعة' if is_ar else 'Predicted Temperature'}: 🌡️ {round(prediction, 1)}°C")

    st.markdown("---")
    st.caption("تم التطوير باستخدام Streamlit • Weather ML Demo")
