# simulation/robot_simulator.py
import os
import time
import matplotlib.pyplot as plt
from utils.data_loader import load_orders, load_robots
from pathfinding.a_star import astar
from math import hypot


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT, 'models', 'robot_assignment.pkl')

def draw_grid(ax, orders_df, robots_df, grid_size=12):
    ax.clear()
    ax.set_xlim(-1, grid_size+1)
    ax.set_ylim(-1, grid_size+1)
    ax.set_xticks(range(0, grid_size+1))
    ax.set_yticks(range(0, grid_size+1))
    ax.grid(True)
    ax.scatter(orders_df['LocationX'], orders_df['LocationY'], c='red', marker='o', label='Orders')
    ax.scatter(robots_df['CurrentX'], robots_df['CurrentY'], c='blue', marker='s', label='Robots')
    for _, r in robots_df.iterrows():
        ax.text(r['CurrentX']+0.1, r['CurrentY']+0.1, r['RobotID'], fontsize=8)
    ax.legend()

def euclidean(a,b): return hypot(a[0]-b[0], a[1]-b[1])

def simulate_one_run(model_bundle=None, grid_size=12, pause=0.05):
    orders = load_orders().copy()
    robots = load_robots().copy()
    blocked = set()  # add blocked cells if you want
    fig, ax = plt.subplots(figsize=(6,6))
    draw_grid(ax, orders, robots, grid_size)
    plt.ion()
    plt.show()
    # simple heuristic assignment: nearest robot that can carry
    while not orders.empty:
        order = orders.iloc[0]
        # pick robot
        best_idx = None
        best_dist = 1e9
        for idx, r in robots.iterrows():
            if r['LoadCapacity(kg)'] < order['Weight']:
                continue
            d = euclidean((r['CurrentX'], r['CurrentY']), (order['LocationX'], order['LocationY']))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None:
            best_idx = 0  # fallback
        start = (int(robots.at[best_idx,'CurrentX']), int(robots.at[best_idx,'CurrentY']))
        goal = (int(order['LocationX']), int(order['LocationY']))
        path = astar(start, goal, blocked, grid_size)
        if not path:
            # straight-line path
            x,y = start
            path = []
            gx,gy = goal
            while (x,y) != (gx,gy):
                if x < gx: x += 1
                elif x > gx: x -= 1
                if y < gy: y += 1
                elif y > gy: y -= 1
                path.append((x,y))
        for (nx,ny) in path:
            robots.at[best_idx,'CurrentX'] = nx
            robots.at[best_idx,'CurrentY'] = ny
            draw_grid(ax, orders, robots, grid_size)
            plt.pause(pause)
        # simulate pickup -> deliver to origin (0,0)
        path_back = astar(goal, (0,0), blocked, grid_size) or []
        for (nx,ny) in path_back:
            robots.at[best_idx,'CurrentX'] = nx
            robots.at[best_idx,'CurrentY'] = ny
            draw_grid(ax, orders, robots, grid_size)
            plt.pause(pause)
        orders = orders[orders['OrderID'] != order['OrderID']]
    plt.ioff()
    plt.show()
