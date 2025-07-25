import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("nyc_hourly_cleaned.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Prepare features using past 24 hours
look_back = 24
X, y = [], []
data = df[["temperature"]].values

for i in range(len(data) - look_back):
    X.append(data[i:i+look_back].flatten())
    y.append(data[i+look_back][0])

X = np.array(X)
y = np.array(y)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/temperature_model.pkl")
print("✅ Model trained and saved inside Streamlit environment.")
