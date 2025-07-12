import joblib
import numpy as np
import os
import streamlit as st

@st.cache_resource
def load_model():
    path = os.path.join("model", "temperature_model.pkl")
    return joblib.load(path)

def generate_dummy_input():
    # Simulate last 24 hours temp (example), you can replace this with real historical data
    return np.array([25]*24).reshape(1, -1)
