# ui/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import os
import joblib
from utils.data_loader import load_orders, load_robots
from ml_model import train_and_save
from math import hypot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT, 'models', 'robot_assignment.pkl')
DATA_DIR = os.path.join(ROOT, 'data')

st.set_page_config(layout="wide")
st.title("warehouse_ai — Dispatcher")

# Ensure model exists (auto-train)
if not os.path.exists(MODEL_PATH):
    st.info("No model found. Training model automatically...")
    train_and_save()
    st.success("Model trained.")

# Load model
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
le = bundle['label_encoder']

menu = st.sidebar.selectbox("Menu", ["Assign Order", "View Data", "Train Model", "Simulate"])

if menu == "Assign Order":
    st.header("Assign an order (ML)")
    orders = load_orders()
    robots = load_robots()
    col1, col2, col3 = st.columns(3)
    with col1:
        distance = st.number_input("Distance (grid units)", min_value=0.0, value=5.0)
        # You can compute distance from robots and order location; simplified input here
    with col2:
        weight = st.number_input("Weight (kg)", min_value=0.1, value=2.0)
    with col3:
        fragile = st.checkbox("Fragile?")
    if st.button("Assign"):
        fragile_flag = 1 if fragile else 0
        # model expects [distance, weight, fragile, (urgency optional - not used here)]
        feat = [[distance, weight, fragile_flag]]
        pred = model.predict(feat)[0]
        robot_id = le.inverse_transform([int(pred)])[0]
        st.success(f"Assigned robot: {robot_id}")

elif menu == "View Data":
    st.header("Data")
    st.write("Orders")
    st.dataframe(load_orders())
    st.write("Robots")
    st.dataframe(load_robots())

elif menu == "Train Model":
    st.header("Retrain Model")
    if st.button("Retrain Now"):
        train_and_save()
        st.success("Model retrained.")

elif menu == "Simulate":
    st.header("Run Simulation")
    st.write("Simulation will open in a separate matplotlib window.")
    if st.button("Start Simulation"):
        from simulation.robot_simulator import simulate_one_run
        simulate_one_run()
        st.success("Simulation completed (or closed).")
