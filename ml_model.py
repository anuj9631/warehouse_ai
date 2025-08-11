import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from utils import euclidean_distance
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    orders = pd.read_csv(f'{DATA_DIR}/orders.csv')
    robots = pd.read_csv(f'{DATA_DIR}/robots.csv')
    return orders, robots

def create_synthetic_training(orders, robots, n_samples=300):
    """
    Create synthetic samples: randomly pick a robot and order,
    compute features, and mark whether that robot is a 'good' choice.
    We'll label 'good' when robot can carry the load and battery/distance ratio is acceptable.
    Then convert into multiclass by choosing the best robot among all for each synthetic order.
    """
    samples = []
    for _ in range(n_samples):
        # pick random order spot (or jitter existing)
        o = orders.sample(1).iloc[0]
        order_loc = (o['LocationX'], o['LocationY'])
        # create candidate robots features and choose best by heuristic scoring
        scores = []
        robot_features = []
        for _, r in robots.iterrows():
            robot_loc = (r['CurrentX'], r['CurrentY'])
            dist = euclidean_distance(robot_loc, order_loc)
            capacity = r['LoadCapacity(kg)'] if 'LoadCapacity(kg)' in r else r['LoadCapacity']
            battery = r['Battery(%)'] if 'Battery(%)' in r else r['Battery']
            weight = o['Weight']
            # heuristic: can carry weight -> penalty by distance / speed and battery margin
            speed = r['Speed(m/s)'] if 'Speed(m/s)' in r else r['Speed']
            if capacity < weight:
                score = -999  # can't carry
            else:
                # smaller is better
                score = -(dist / speed) + (battery/100.0) - (weight/50.0)
            scores.append(score)
            robot_features.append((dist, weight, battery, capacity))
        # pick best robot index
        best_idx = int(np.argmax(scores))
        # For training we create one sample per robot but label the best one
        for ridx, feat in enumerate(robot_features):
            dist, weight, battery, capacity = feat
            samples.append({
                'dist': dist,
                'weight': weight,
                'battery': battery,
                'capacity': capacity,
                'robot_id': robots.iloc[ridx]['RobotID'],
                'label': 1 if ridx==best_idx else 0,
                'best_robot': robots.iloc[best_idx]['RobotID']
            })
    df = pd.DataFrame(samples)
    # For multiclass classifier we'll aggregate per synthetic order: each sample row representing the "candidate" and best_robot as target.
    # But easiest: create dataset where features are aggregated per (order) by creating many samples and mapping best_robot -> class
    X = df[['dist','weight','battery','capacity']].values
    # We need target as the best robot id mapping - create mapping by selecting only rows that were labeled 1 (one per synthetic order)
    targets = df[df['label']==1]['robot_id'].values
    X_targets = df[df['label']==1][['dist','weight','battery','capacity']].values
    return X_targets, targets

def train_and_save():
    orders, robots = load_data()
    X, y = create_synthetic_training(orders, robots, n_samples=400)
    # map robot ids to numeric classes
    classes = np.unique(y)
    class_map = {rid: i for i,rid in enumerate(classes)}
    y_num = np.array([class_map[r] for r in y])
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X, y_num)
    # save model + class_map
    dump({'model': clf, 'class_map': class_map, 'classes': classes}, f'{MODEL_DIR}/robot_assignment_model.pkl')
    print(f"Saved model to {MODEL_DIR}/robot_assignment_model.pkl. Classes: {class_map}")

if __name__ == "__main__":
    train_and_save()
