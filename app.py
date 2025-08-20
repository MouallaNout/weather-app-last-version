import streamlit as st
import csv
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================== App ====================
lang = st.sidebar.selectbox("اللغة / Language", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("Weather forecasting system using machine learning" if not is_ar else " نظام التنبؤ بالطقس باستخدام التعلم الآلي")

city_coords = {}
with open("worldcities.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        country = row["country"]
        city = row["city"]
        lat = float(row["lat"])
        lng = float(row["lng"])
        if country not in city_coords:
            city_coords[country] = {}
        city_coords[country][city] = (lat, lng)

st.sidebar.markdown("اختر الدولة والمدينة" if is_ar else "Select Country and City")
country = st.sidebar.selectbox("الدولة" if is_ar else "Country", list(city_coords.keys()))
city = st.sidebar.selectbox("المدينة" if is_ar else "City", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

st.sidebar.markdown("ماذا تريد أن يتم التنبؤ به؟" if is_ar else "Select what to predict")
all_vars = {
    ("Temperature" if not is_ar else "درجة الحرارة"): "temperature",
    ("Humidity" if not is_ar else "الرطوبة"): "humidity",
    ("Wind Speed" if not is_ar else "سرعة الرياح"): "wind_speed"
}
selected_display = st.sidebar.multiselect("", list(all_vars.keys()), default=list(all_vars.keys()))
selected_vars = [all_vars[d] for d in selected_display]

st.sidebar.markdown("Select units" if not is_ar else "اختر وحدات القياس")
unit_temp = st.sidebar.radio(
    "درجة الحرارة" if is_ar else "Temperature",
    ["°م" if is_ar else "°C", "°ف" if is_ar else "°F"],
    index=0
)
unit_wind = st.sidebar.radio(
    "سرعة الرياح" if is_ar else "Wind Speed",
    ["كم/سا" if is_ar else "km/h", "م/ث" if is_ar else "m/s"],
    index=0
)

# بدء جلب البيانات
if st.sidebar.button("ابدأ التنبؤ" if is_ar else "Start Prediction"):
    start_date = (date.today() - timedelta(days=730)).isoformat()
    end_date = date.today().isoformat()
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
    )

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            st.error("فشل في الاتصال بواجهة الطقس." if is_ar else "Failed to connect to weather API.")
            st.stop()

        try:
            data = response.json()
        except ValueError:
            st.error("الاستجابة من API غير صالحة." if is_ar else "Invalid response from API.")
            st.stop()
    
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["windspeed_10m"]
        })

    except Exception as e:
        st.error("حدث خطأ أثناء تحميل البيانات." if is_ar else f"An error occurred while fetching data: {e}")
        st.stop()


    # معالجة القيم المفقودة
    def fill_with_avg_of_neighbors(series):
        series = series.copy()
        for i in range(1, len(series) - 1):
            if pd.isna(series[i]) and not pd.isna(series[i - 1]) and not pd.isna(series[i + 1]):
                series[i] = (series[i - 1] + series[i + 1]) / 2
        return series

    for col in df.columns[1:]:
        df[col] = fill_with_avg_of_neighbors(df[col])
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    # تدريب النماذج والتنبؤ
    look_back = 72
    hours_ahead = 24
    forecast_results = {}

    for var in selected_vars:
        X, y = [], []
        data_arr = df[[var]].values
        for i in range(len(data_arr) - look_back):
            X.append(data_arr[i:i+look_back].flatten())
            y.append(data_arr[i+look_back][0])
        X, y = np.array(X), np.array(y)

        X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)

        current_sequence = df[[var]].values[-look_back:].flatten().reshape(1, -1)
        hourly_preds = []
        for _ in range(hours_ahead):
            pred = model.predict(current_sequence)[0]
            hourly_preds.append(pred)
            current_sequence = np.append(current_sequence[:, 1:], [[pred]], axis=1)

        forecast_results[var] = hourly_preds

    # تجهيز بيانات التوقيت والغد
    start_time = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    hourly_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]
    df_forecast = pd.DataFrame({ "Time": hourly_times })

    if "temperature" in forecast_results:
        temp = forecast_results["temperature"]
        if unit_temp == "°F":
            temp = [(t * 9/5) + 32 for t in temp]
        df_forecast[f"Temperature ({unit_temp})"] = temp

    if "humidity" in forecast_results:
        df_forecast["Humidity (%)"] = forecast_results["humidity"]

    if "wind_speed" in forecast_results:
        wind = forecast_results["wind_speed"]
        if unit_wind == "m/s":
            wind = [w / 3.6 for w in wind]
        df_forecast[f"Wind Speed ({unit_wind})"] = wind

    # الرسم البياني
    def plot_line_chart(df, column, title):
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df[column], marker='o')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Hourly Weather Forecast for Tomorrow" if not is_ar else "توقعات الطقس لكل ساعة غداً")
    st.markdown(f"{city}, {country}")
    st.markdown(f"{date.today() + timedelta(days=1)}")

    for col in df_forecast.columns:
        if col != "Time":
            label = col.split(" (")[0]
            title = label if is_ar else f"{label} Throughout the Day"
            plot_line_chart(df_forecast, col, title)

    st.dataframe(df_forecast.style.format(precision=1))
