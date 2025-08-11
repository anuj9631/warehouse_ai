import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import euclidean_distance



import streamlit as st
import pandas as pd
import os
from joblib import load
from utils import euclidean_distance

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.title("warehouse_ai - Simple Order Dispatcher")

if st.button("Show orders and robots"):
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    robots = pd.read_csv(f"{DATA_DIR}/robots.csv")
    st.subheader("Orders")
    st.dataframe(orders)
    st.subheader("Robots")
    st.dataframe(robots)

st.write("---")
st.header("Assign a single order (ML)")

order_id = st.number_input("OrderID", min_value=1, max_value=100, value=1, step=1)
if st.button("Assign"):
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    robots = pd.read_csv(f"{DATA_DIR}/robots.csv")
    order = orders[orders['OrderID'] == order_id]
    if order.empty:
        st.error("Order not found")
    else:
        bundle = load(f"{MODEL_DIR}/robot_assignment_model.pkl")
        model = bundle['model']
        class_map = bundle['class_map']
        inv_map = {v:k for k,v in class_map.items()}
        # pick best robot by logic similar to simulation
        best_idx = None
        best_dist = 1e9
        for idx, r in robots.iterrows():
            if r['LoadCapacity(kg)'] < order.iloc[0]['Weight']:
                continue
            dist = euclidean_distance((r['CurrentX'], r['CurrentY']),
                                      (order.iloc[0]['LocationX'], order.iloc[0]['LocationY']))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            st.success(f"Assigned robot {robots.loc[best_idx,'RobotID']} (idx {best_idx})")
        else:
            st.warning("No capable robot found — choose manual assignment")
