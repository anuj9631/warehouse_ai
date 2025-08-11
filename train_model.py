import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Dummy training data (distance, weight → time)
X = np.array([
    [5, 2],
    [10, 5],
    [2, 1],
    [8, 3]
])
y = np.array([10, 25, 5, 20])  # delivery times

# Train the model
model = LinearRegression()
model.fit(X, y)

# Dummy class_map (robot IDs → names)
class_map = {
    0: "Robot Alpha",
    1: "Robot Beta",
    2: "Robot Gamma"
}

# Save as dictionary
bundle = {
    'model': model,
    'class_map': class_map
}

joblib.dump(bundle, "models/robot_assignment_model.pkl")
print("✅ Model + class_map saved to models/robot_assignment_model.pkl")
