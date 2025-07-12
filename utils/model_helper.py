import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def generate_dummy_input():
    # إدخال وهمي بطول 24 ساعة (للتجربة)
    return np.random.rand(1, 24)
