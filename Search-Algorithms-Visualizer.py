import pygame
import heapq
import math
import time
import sys

# ─── Constants ─────────────────────────────────────────────
GRID_ROWS, GRID_COLS = 8, 8
CELL_SIZE = 80
SIDE_PANEL_WIDTH = 280
WINDOW_WIDTH = GRID_COLS * CELL_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE

FPS = 30
STEP_DELAY = 120

WHITE   = (255, 255, 255)
BLACK   = (30,  30,  30)
GRAY    = (180, 180, 180)
DKGRAY  = (60,  60,  60)
START_C = (50,  200, 80)
GOAL_C  = (220, 50,  50)
WALL_C  = (40,  40,  40)
FRONT_C = (255, 220, 0)
VISIT_C = (100, 149, 237)
PATH_C  = (50,  220, 130)
PANEL_C = (25,  25,  40)
BTN_C   = (60,  80,  140)
BTN_H   = (90,  120, 200)
BTN_ACT = (40,  180, 100)
TEXT_C  = (230, 230, 230)
TITLE_C = (180, 210, 255)

EMPTY, WALL, START, GOAL = 0, 1, 2, 3

# ─── Grid ──────────────────────────────────────────────────
def build_grid():
    grid = [[EMPTY]*GRID_COLS for _ in range(GRID_ROWS)]
    for i in range(2, 6):
        grid[i][3] = WALL
        grid[i][5] = WALL
    for j in range(1, 4):
        grid[4][j] = WALL
    for j in range(4, 7):
        grid[6][j] = WALL
    grid[1][1] = START
    grid[6][6] = GOAL
    return grid

START_POS = (1, 1)
GOAL_POS  = (6, 6)

DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def get_neighbors(pos, grid):
    r, c = pos
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

# ─── Heuristics ────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ─── Algorithms ────────────────────────────────────────────
def gbfs(grid, start, goal, heuristic):
    open_set = [(heuristic(start, goal), 0, start)]
    counter = 1
    came_from = {start: None}
    visited_nodes = set()
    explored_count = 0
    start_time = time.time()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited_nodes:
            continue
        visited_nodes.add(current)
        explored_count += 1
        yield ('visit', current, visited_nodes.copy(),
               set(n for _,_,n in open_set), None,
               explored_count, 0, 0)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            cost = len(path) - 1
            elapsed = (time.time() - start_time) * 1000
            yield ('done', None, visited_nodes, set(),
                   path, explored_count, cost, elapsed)
            return

        for nb in get_neighbors(current, grid):
            if nb not in visited_nodes and nb not in came_from:
                came_from[nb] = current
                heapq.heappush(open_set, (heuristic(nb, goal), counter, nb))
                counter += 1

        yield ('frontier', current, visited_nodes.copy(),
               set(n for _,_,n in open_set),
               None, explored_count, 0, 0)

    yield ('done', None, visited_nodes, set(),
           None, explored_count, 0,
           (time.time()-start_time)*1000)

def astar(grid, start, goal, heuristic):
    open_set = [(heuristic(start, goal), 0, start)]
    counter = 1
    came_from = {start: None}
    g_score = {start: 0}
    visited_nodes = set()
    explored_count = 0
    start_time = time.time()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited_nodes:
            continue
        visited_nodes.add(current)
        explored_count += 1
        yield ('visit', current, visited_nodes.copy(),
               set(n for _,_,n in open_set),
               None, explored_count, 0, 0)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            cost = g_score[goal]
            elapsed = (time.time() - start_time) * 1000
            yield ('done', None, visited_nodes, set(),
                   path, explored_count,
                   round(cost,2), elapsed)
            return

        for nb in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal)
                came_from[nb] = current
                heapq.heappush(open_set, (f, counter, nb))
                counter += 1

        yield ('frontier', current, visited_nodes.copy(),
               set(n for _,_,n in open_set),
               None, explored_count, 0, 0)

    yield ('done', None, visited_nodes, set(),
           None, explored_count, 0,
           (time.time()-start_time)*1000)

# ─── Main ──────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("GBFS / A* Search Visualizer")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("segoeui", 18, bold=True)
    font_med = pygame.font.SysFont("segoeui", 15, bold=True)
    font_sm  = pygame.font.SysFont("segoeui", 14)

    grid = build_grid()

    algo_index = 0
    heuristic_index = 0

    visited_nodes = set()
    frontier_nodes = set()
    final_path = []

    explored_count = 0
    path_cost = 0
    time_taken = 0.0

    app_state = 'idle'
    search_generator = None
    last_step = 0

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if app_state == 'running' and search_generator and (now - last_step) >= STEP_DELAY:
            last_step = now
            try:
                kind, _, vis, fron, pth, nv, pc, el = next(search_generator)
                visited_nodes = vis
                frontier_nodes = fron
                explored_count = nv

                if kind == 'done':
                    time_taken = el
                    if pth:
                        final_path = pth
                        path_cost = pc
                        app_state = 'done'
                    else:
                        app_state = 'no_path'
                    search_generator = None
            except StopIteration:
                app_state = 'done'
                search_generator = None

        screen.fill(BLACK)

        path_set = set(final_path) if final_path else set()

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x, y = c*CELL_SIZE, r*CELL_SIZE
                cell = grid[r][c]

                if cell == WALL:
                    color = WALL_C
                elif cell == START:
                    color = START_C
                elif cell == GOAL:
                    color = GOAL_C
                elif (r,c) in path_set:
                    color = PATH_C
                elif (r,c) in visited_nodes:
                    color = VISIT_C
                elif (r,c) in frontier_nodes:
                    color = FRONT_C
                else:
                    color = WHITE

                pygame.draw.rect(screen, color,
                                 (x+1, y+1, CELL_SIZE-2, CELL_SIZE-2),
                                 border_radius=4)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()