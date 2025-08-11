import math

def euclidean_distance(a, b):
    """Compute Euclidean distance between points a=(x,y) and b=(x,y)."""
    return math.hypot(a[0]-b[0], a[1]-b[1])
