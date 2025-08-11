# Simple grid A* pathfinding (4-neighbor)
import heapq

def astar(start, goal, grid, grid_size):
    """
    start, goal: (x,y)
    grid: set of blocked coordinates e.g. {(x1,y1), (x2,y2)}
    grid_size: (width, height)
    returns list of (x,y) or empty list if no path
    """
    def neighbors(node):
        x,y = node
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx,ny) not in grid:
                yield (nx,ny)

    def h(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # manhattan

    open_set = []
    heapq.heappush(open_set, (h(start,goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            # reconstruct
            path = []
            n = current
            while n:
                path.append(n)
                n = came_from[n]
            return list(reversed(path))
        for nb in neighbors(current):
            tentative_g = gscore[current] + 1
            if nb not in gscore or tentative_g < gscore[nb]:
                gscore[nb] = tentative_g
                f = tentative_g + h(nb, goal)
                heapq.heappush(open_set, (f, tentative_g, nb, current))
    return []
