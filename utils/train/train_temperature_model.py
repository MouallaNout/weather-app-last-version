import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# تحميل البيانات
df = pd.read_csv("data/nyc_hourly_cleaned.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# تجهيز بيانات التسلسل
look_back = 24
target = "temperature"
X, y = [], []
data = df[[target]].values
for i in range(len(data) - look_back):
    X.append(data[i:i+look_back].flatten())
    y.append(data[i+look_back][0])

X, y = np.array(X), np.array(y)

# تدريب النموذج
model = LinearRegression()
model.fit(X, y)

# حفظ النموذج
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/temperature_model.pkl")
print("✅ Model saved!")
