import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ---------------------- إعداد اللغة ----------------------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
title = "توقع الطقس باستخدام الذكاء الاصطناعي" if is_ar else "AI-Based Weather Forecast"
st.title(title)

# ---------------------- اختيار الدولة والمدينة ----------------------
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

# ---------------------- زر التنبؤ ----------------------
if st.sidebar.button("ابدأ التنبؤ" if is_ar else "Start Prediction"):
    with st.spinner("🔄 " + ("جاري تحميل البيانات..." if is_ar else "Fetching weather data...")):
        st.write("📡 Connecting to weather API...")
        start_str = "2023-01-01"
        end_str = "2024-12-31"
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_str}&end_date={end_str}"
            f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
            f"&timezone=auto"
        )

        try:
            resp = requests.get(api_url)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"]
            })
            st.success("✅ تم تحميل بيانات الطقس!" if is_ar else "✅ Weather data loaded successfully!")
            st.write("📊 Raw data shape:", df.shape)
        except Exception as e:
            st.error("فشل تحميل البيانات. تأكد من الاتصال بالإنترنت." if is_ar else f"Failed to fetch data: {e}")
            st.stop()

    # ---------------------- تنظيف البيانات ----------------------
    st.write("🪄 Cleaning data...")
    df_original = df.copy()
    for col, cond in [
        ("temperature", (df["temperature"] < -60) | (df["temperature"] > 60)),
        ("humidity", (df["humidity"] < 0) | (df["humidity"] > 100)),
        ("wind_speed", (df["wind_speed"] < 0) | (df["wind_speed"] > 60))
    ]:
        for idx in df[cond].index:
            if 0 < idx < len(df) - 1:
                df.loc[idx, col] = (df.loc[idx - 1, col] + df.loc[idx + 1, col]) / 2

    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = df[col].apply(lambda x: int(x + 0.5))

    st.success("✅ Step 1: Data cleaned")

    # ---------------------- التجهيز للنماذج ----------------------
    look_back = 72
    target = "temperature"
    X, y = [], []
    data = df[[target]].values
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back].flatten())
        y.append(data[i+look_back][0])
    X, y = np.array(X), np.array(y)

    st.success(f"✅ Step 2: Features prepared. X shape = {X.shape}, y shape = {y.shape}")

    if len(X) == 0:
        st.warning("البيانات غير كافية للتدريب." if is_ar else "Not enough data to train.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Ensure float32 for XGBoost
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    st.success("✅ Step 3: Data split into training/testing")

    # ---------------------- تدريب النماذج ----------------------
    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    }

    results = {}
    times = {}
    predictions = []

    for name, model in models.items():
        try:
            st.info(f"🧠 Training model: {name}")
            start = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            elapsed = time.time() - start
            mae = mean_absolute_error(y_test, pred)

            results[name] = mae
            times[name] = elapsed
            predictions.append(pred)

            st.success(f"✅ {name} MAE: {mae:.3f}, Time: {elapsed:.2f}s")
        except Exception as e:
            st.error(f"❌ Error training {name}: {e}")

    if not predictions:
        st.error("❌ No predictions were successful.")
        st.stop()

    # ---------------------- Ensemble ----------------------
    st.info("🔄 Creating ensemble average...")
    final_prediction = np.mean(predictions, axis=0)
    final_mae = mean_absolute_error(y_test, final_prediction)

    results["Ensemble Average"] = final_mae
    times["Ensemble Average"] = 0

    df_results = pd.DataFrame({
        "MAE": results,
        "Time (s)": times
    })

    # ---------------------- عرض النتائج ----------------------
    st.success("✅ All models trained and evaluated.")

    st.markdown("### ⚙️ نتائج النماذج" if is_ar else "### ⚙️ Model Performance")
    st.dataframe(df_results.style.format({"MAE": "{:.2f}", "Time (s)": "{:.2f}"}))

    st.markdown("### 📊 المقارنة بين النماذج" if is_ar else "### 📊 Model Comparison")
    st.bar_chart(df_results["MAE"])

    st.markdown("### ⏱️ الزمن المستغرق في التدريب" if is_ar else "### ⏱️ Training Time")
    st.bar_chart(df_results.drop(index="Ensemble Average")["Time (s)"])
