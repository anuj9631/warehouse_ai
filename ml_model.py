# ml_model.py
import os
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import load_orders, load_robots
from math import hypot

ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'robot_assignment.pkl')

def euclidean(a,b):
    return hypot(a[0]-b[0], a[1]-b[1])

def generate_training_data(orders, robots, n_samples_per_order=1):
    """
    Create training pairs: for each order, choose the best robot by a heuristic:
     - robot must have capacity >= weight
     - score = battery% - distance/speed - (weight/50)
    Then label that robot as target.
    Returns X (features) and y (robot label)
    """
    X = []
    y = []
    for _, order in orders.iterrows():
        order_loc = (order['LocationX'], order['LocationY'])
        candidates = []
        for _, r in robots.iterrows():
            robot_loc = (r['CurrentX'], r['CurrentY'])
            dist = euclidean(robot_loc, order_loc)
            speed = r.get('Speed(m/s)', r.get('Speed', 1.0))
            capacity = r.get('LoadCapacity(kg)', r.get('LoadCapacity', 10))
            battery = r.get('Battery(%)', r.get('Battery', 80))
            if capacity < order['Weight']:
                # can't carry, but still include with penalty
                score = -999
            else:
                score = battery - (dist / speed) - (order['Weight']/50.0)
            candidates.append((score, r['RobotID'], dist, capacity, battery, speed))
        # choose best
        best = max(candidates, key=lambda x: x[0])
        # create feature vector for this order labelled as best robot
        feat = [best[2], order['Weight'], 1 if str(order.get('Fragile','No')).lower().startswith('y') else 0]
        # include urgency if exists
        if 'Urgency' in orders.columns:
            urgency_map = {'low': 0, 'medium':1, 'high':2}
            feat.append(urgency_map.get(order['Urgency'], 0))
        X.append(feat)
        y.append(best[1])
    return np.array(X), np.array(y)

def train_and_save():
    orders = load_orders()
    robots = load_robots()
    X, y = generate_training_data(orders, robots)
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf.fit(X, y_num)
    bundle = {'model': clf, 'label_encoder': le}
    joblib.dump(bundle, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return MODEL_PATH

if __name__ == "__main__":
    train_and_save()
