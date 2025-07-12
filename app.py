import streamlit as st
from utils.location_data import get_country_list, get_cities_for_country
from utils.model_helper import load_model, generate_dummy_input

# ---------- Language Toggle ----------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"

# ---------- Country and City Dropdown ----------
country_sel = st.sidebar.selectbox("Select Country" if not is_ar else "اختر الدولة",
                                   [name for code, name in get_country_list()])
country_code = [code for code, name in get_country_list() if name == country_sel][0]
city_list = [city for city, lat, lon in get_cities_for_country(country_code)]
city_sel = st.sidebar.selectbox("Select City" if not is_ar else "اختر المدينة", city_list)
lat_lon = {city: (lat, lon) for city, lat, lon in get_cities_for_country(country_code)}
lat, lon = lat_lon[city_sel]

# ---------- Prediction Button ----------
if st.sidebar.button("Predict" if not is_ar else "تنبؤ"):
    model = load_model("model/temperature_model.pkl")
    X_dummy = generate_dummy_input()  # Replace later with real input
    prediction = model.predict(X_dummy)[0]

    # ---------- Display Result ----------
    st.title("Weather Prediction" if not is_ar else "توقع الطقس")
    st.subheader(f"{'City' if not is_ar else 'المدينة'}: {city_sel}")
    st.write(f"{'Predicted Temperature' if not is_ar else 'درجة الحرارة المتوقعة'}: 🌡️ {prediction:.1f}°C")

    st.markdown("---")
    st.caption("Powered by Streamlit & Machine Learning")
