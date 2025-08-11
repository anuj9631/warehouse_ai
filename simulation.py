import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from pathfinding import astar
from utils import euclidean_distance

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
GRID_SIZE = 12

def load_resources():
    orders = pd.read_csv(f'{DATA_DIR}/orders.csv')
    robots = pd.read_csv(f'{DATA_DIR}/robots.csv')
    model_bundle = load(f'{MODEL_DIR}/robot_assignment_model.pkl')
    model = model_bundle['model']
    class_map = model_bundle['class_map']  # mapping robotid->int
    inv_map = {v:k for k,v in class_map.items()}
    return orders, robots, model, inv_map

def pick_robot_ml(order, robots, model, inv_map):
    """
    Decide best robot using model. If model predicts a robot that can't carry -> fallback.
    """
    best_score = None
    chosen_robot_id = None
    # features: dist, weight, battery, capacity
    # We'll compute features for each robot, pick robot with predicted class that matches robot id.
    # Simpler: for each robot compute feature vector, ask which robot is predicted (we trained based on best robot features).
    best_candidates = []
    for _, r in robots.iterrows():
        dist = euclidean_distance((r['CurrentX'], r['CurrentY']), (order['LocationX'], order['LocationY']))
        weight = order['Weight']
        battery = r['Battery(%)']
        capacity = r['LoadCapacity(kg)'] if 'LoadCapacity(kg)' in r else r['LoadCapacity']
        feat = np.array([[dist, weight, battery, capacity]])
        pred = model.predict(feat)[0]
        robot_id_pred = inv_map[pred]
        # If predicted this robot is the best, consider it
        if robot_id_pred == r['RobotID'] and capacity >= weight:
            best_candidates.append((r['RobotID'], dist, r))
    # If one or more candidates, pick nearest
    if best_candidates:
        best_candidates.sort(key=lambda x:x[1])
        return best_candidates[0][2].name  # index in robots dataframe
    # fallback: nearest robot that can carry
    possible = []
    for idx, r in robots.iterrows():
        if r['LoadCapacity(kg)'] >= order['Weight']:
            dist = euclidean_distance((r['CurrentX'], r['CurrentY']), (order['LocationX'], order['LocationY']))
            possible.append((dist, idx))
    if possible:
        possible.sort(key=lambda x:x[0])
        return possible[0][1]
    # final fallback: any robot (even if can't carry) -> index 0
    return 0

def plot_setup():
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1, GRID_SIZE+1)
    ax.set_ylim(-1, GRID_SIZE+1)
    ax.set_xticks(range(0, GRID_SIZE+1))
    ax.set_yticks(range(0, GRID_SIZE+1))
    ax.grid(True)
    return fig, ax

def draw(ax, orders, robots_df):
    ax.clear()
    ax.set_xlim(-1, GRID_SIZE+1)
    ax.set_ylim(-1, GRID_SIZE+1)
    ax.grid(True)
    # plot orders
    ax.scatter(orders['LocationX'], orders['LocationY'], c='red', marker='o', label='Orders')
    # plot robots
    ax.scatter(robots_df['CurrentX'], robots_df['CurrentY'], c='blue', marker='s', label='Robots')
    for _, r in robots_df.iterrows():
        ax.text(r['CurrentX']+0.1, r['CurrentY']+0.1, r['RobotID'], fontsize=8)
    ax.legend(loc='upper right')
    plt.pause(0.05)

def simulate():
    orders, robots, model, inv_map = load_resources()
    fig, ax = plot_setup()
    blocked = set()  # no obstacles for now; add positions in future
    for _, order in orders.iterrows():
        idx = pick_robot_ml(order, robots, model, inv_map)
        robot = robots.loc[idx]
        print(f"Assigning {robot['RobotID']} (idx {idx}) to Order {order['OrderID']}")
        start = (int(robot['CurrentX']), int(robot['CurrentY']))
        goal = (int(order['LocationX']), int(order['LocationY']))
        path = astar(start, goal, blocked, GRID_SIZE)
        if not path:
            # fallback to straight move if no path found
            path = []
            x,y = start
            gx,gy = goal
            while (x,y) != (gx,gy):
                if x < gx: x+=1
                elif x > gx: x-=1
                if y < gy: y+=1
                elif y > gy: y-=1
                path.append((x,y))
        # move robot along path
        for (nx,ny) in path:
            robots.at[idx, 'CurrentX'] = nx
            robots.at[idx, 'CurrentY'] = ny
            draw(ax, orders, robots)
            time.sleep(0.05)
        # simulate pickup -> delivery to packing station (we'll assume packing at 0,0)
        packing = (0,0)
        path2 = astar(goal, packing, blocked, GRID_SIZE) or []
        for (nx,ny) in path2:
            robots.at[idx, 'CurrentX'] = nx
            robots.at[idx, 'CurrentY'] = ny
            draw(ax, orders, robots)
            time.sleep(0.05)
        # after delivery, remove order marker (simulate completed)
        orders = orders[orders['OrderID'] != order['OrderID']]
        draw(ax, orders, robots)
        time.sleep(0.2)
    print("All orders processed.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    simulate()
