import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def generate_dummy_input():
    # 24 dummy hourly values
    return np.zeros((1, 24))
