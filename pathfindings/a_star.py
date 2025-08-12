# pathfinding/a_star.py
import heapq

def astar(start, goal, blocked, grid_size):
    """
    Simple A* for grid with 4-neighbors. 
    start, goal: tuples (x,y)
    blocked: set of (x,y) coordinates
    grid_size: int (grid is 0..grid_size-1)
    """
    def neighbors(n):
        x,y = n
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx,ny) not in blocked:
                yield (nx,ny)

    def h(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_heap = []
    heapq.heappush(open_heap, (h(start,goal), 0, start))
    came_from = {}
    gscore = {start: 0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct path
            path = []
            n = current
            while n != start:
                path.append(n)
                n = came_from[n]
            path.reverse()
            return path
        for nb in neighbors(current):
            tentative = g + 1
            if nb not in gscore or tentative < gscore[nb]:
                gscore[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative + h(nb, goal), tentative, nb))
    return []
