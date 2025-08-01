import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import io
import os
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import matplotlib.pyplot as plt

# ================= Google Drive Integration =================
@st.cache_resource
def create_drive_service():
    credentials = service_account.Credentials.from_service_account_info({
        "type": "service_account",
        "project_id": st.secrets["GDRIVE_PROJECT_ID"],
        "private_key_id": "",
        "private_key": st.secrets["GDRIVE_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": st.secrets["GDRIVE_CLIENT_EMAIL"],
        "client_id": st.secrets["GDRIVE_CLIENT_ID"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{st.secrets['GDRIVE_CLIENT_EMAIL'].replace('@', '%40')}"
    })
    return build('drive', 'v3', credentials=credentials)

def find_drive_file(service, filename):
    query = f"name='{filename}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def download_model(service, filename):
    file_id = find_drive_file(service, filename)
    if not file_id:
        return None
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pickle.load(fh)

def upload_model(service, filename, model):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    file_metadata = {"name": filename}
    media = MediaFileUpload(filename, mimetype="application/octet-stream")
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    os.remove(filename)

# ================== App ====================
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
is_ar = lang == "العربية"
st.title("AI-Based Weather Forecast" if not is_ar else "توقع الطقس باستخدام الذكاء الاصطناعي")

city_coords = {
    "USA": {"New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24)},
    "Saudi Arabia": {"Riyadh": (24.7136, 46.6753), "Jeddah": (21.4858, 39.1925)},
    "Germany": {"Berlin": (52.52, 13.4050), "Munich": (48.1351, 11.5820)}
}

st.sidebar.markdown("### 🌍 " + ("اختر الدولة والمدينة" if is_ar else "Select Country and City"))
country = st.sidebar.selectbox("Country / الدولة", list(city_coords.keys()))
city = st.sidebar.selectbox("City / المدينة", list(city_coords[country].keys()))
lat, lon = city_coords[country][city]

if st.sidebar.button("Start Prediction" if not is_ar else "ابدأ التنبؤ"):
    with st.spinner("Fetching weather data..." if not is_ar else "جاري جلب البيانات..."):
        start_date = (date.today() - timedelta(days=730)).isoformat()
        end_date = date.today().isoformat()
        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m&timezone=auto"
        )

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"]
            })
        except Exception as e:
            st.error("Failed to fetch data" if not is_ar else "فشل جلب البيانات")
            st.stop()

    # Fill missing data
    def fill_with_avg_of_neighbors(series):
        series = series.copy()
        for i in range(1, len(series) - 1):
            if pd.isna(series[i]) and not pd.isna(series[i - 1]) and not pd.isna(series[i + 1]):
                series[i] = (series[i - 1] + series[i + 1]) / 2
        return series

    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = fill_with_avg_of_neighbors(df[col])
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    look_back = 72
    hours_ahead = 24
    selected_vars = ["temperature", "humidity", "wind_speed"]
    forecast_results = {}
    service = create_drive_service()

    for var in selected_vars:
        X, y = [], []
        data_arr = df[[var]].values
        for i in range(len(data_arr) - look_back):
            X.append(data_arr[i:i+look_back].flatten())
            y.append(data_arr[i+look_back][0])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            st.warning("Not enough data to train." if not is_ar else "البيانات غير كافية للتدريب")
            st.stop()

        X_train, _, y_train, _ = train_test_split(X, y, shuffle=False, test_size=0.2)
        model_filename = f"{country}_{city}_{var}.pkl"
        model = download_model(service, model_filename)

        if model:
            st.info(f"✅ Loaded saved model for {city} - {var}")
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            upload_model(service, model_filename, model)
            st.success(f"🆕 Trained and uploaded model for {city} - {var}")

        current_sequence = df[[var]].values[-look_back:].flatten().reshape(1, -1)
        hourly_preds = []
        for _ in range(hours_ahead):
            pred = model.predict(current_sequence)[0]
            hourly_preds.append(pred)
            current_sequence = np.append(current_sequence[:, 1:], [[pred]], axis=1)

        forecast_results[var] = hourly_preds

    start_time = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
    hourly_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]

    df_forecast = pd.DataFrame({"Time": hourly_times})
    if "temperature" in forecast_results:
        df_forecast["Temperature (°C)"] = forecast_results["temperature"]
    if "humidity" in forecast_results:
        df_forecast["Humidity (%)"] = forecast_results["humidity"]
    if "wind_speed" in forecast_results:
        df_forecast["Wind Speed (km/h)"] = forecast_results["wind_speed"]

    def plot_line_chart(df, column, title):
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df[column], marker='o')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Hourly Weather Forecast for Tomorrow" if not is_ar else "توقع الطقس لكل ساعة غدًا")
    st.markdown(f"📍 {city}, {country}")
    st.markdown(f"📅 {date.today() + timedelta(days=1)}")

    for col in df_forecast.columns:
        if col != "Time":
            label = col.split(" (")[0]
            emoji = "🌡️" if "Temp" in col else "💧" if "Humidity" in col else "💨"
            title = emoji + " " + (f"تغير {label}" if is_ar else f"{label} Throughout the Day")
            plot_line_chart(df_forecast, col, title)

    st.dataframe(df_forecast.style.format(precision=1))
