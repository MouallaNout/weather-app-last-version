# app.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 🗣 Language Option
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("توقع الطقس باستخدام الذكاء الاصطناعي" if is_ar else "AI-Based Weather Forecast")

# 🌍 Country/City Dropdown
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

country = st.sidebar.selectbox("الدولة" if is_ar else "Country", list(city_coords.keys()))
city = st.sidebar.selectbox("المدينة" if is_ar else "City", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

if st.sidebar.button("ابدأ التنبؤ" if is_ar else "Start Prediction"):
    with st.spinner("🔄 جاري تحميل البيانات..." if is_ar else "🔄 Fetching weather data..."):
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date=2024-12-31"
            f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
        )
        try:
            response = requests.get(api_url)
            data = response.json()
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"]
            })
        except:
            st.error("فشل تحميل البيانات!" if is_ar else "Failed to fetch data!")
            st.stop()

    # 🧹 Clean
    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = df[col].apply(lambda x: int(x + 0.5))

    # 🎯 Prepare Data
    look_back = 72
    X, y = [], []
    data = df[["temperature"]].values
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back].flatten())
        y.append(data[i+look_back][0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        st.warning("البيانات غير كافية للتدريب." if is_ar else "Not enough data to train.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # 🤖 Models
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR()
    }

    results = {}
    predictions = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions.append(pred)
        results[name] = mean_absolute_error(y_test, pred)

    ensemble_pred = np.mean(predictions, axis=0)
    results["Ensemble"] = mean_absolute_error(y_test, ensemble_pred)

    best_model = min(results, key=results.get)
    best_mae = results[best_model]

    # ✅ Summary
    st.markdown("## 📊 " + ("ملخص التوقعات" if is_ar else "Prediction Summary"))
    st.success(f"""
    📍 **{city}, {country}**  
    📅 Jan 2023 → Dec 2024  
    🤖 Best Model: `{best_model}`  
    📉 MAE: {best_mae:.2f} °C  
    """)

    # 📈 Plot
    st.markdown("### 📈 " + ("درجة الحرارة المتوقعة مقابل الفعلية" if is_ar else "Predicted vs Actual Temperature"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test[:200], label="Actual", color="skyblue", linewidth=2)
    ax.plot(ensemble_pred[:200], label="Predicted", color="salmon", linestyle="--", linewidth=2)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("First 200 Hours Forecast")
    ax.legend()
    st.pyplot(fig)
