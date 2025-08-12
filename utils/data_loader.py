# utils/data_loader.py
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')

def load_orders(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'orders.csv')
    return pd.read_csv(path)

def load_robots(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'robots.csv')
    return pd.read_csv(path)
