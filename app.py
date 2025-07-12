import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned dataset
df = pd.read_csv("nyc_hourly_cleaned.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Create features from past 24 hours
look_back = 24
target = "temperature"

X, y = [], []
data = df[[target]].values
for i in range(len(data) - look_back):
    X.append(data[i:i+look_back].flatten())
    y.append(data[i+look_back][0])

X = np.array(X)
y = np.array(y)

# Train simple model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "model/temperature_model.pkl")
print("✅ Model saved!")
