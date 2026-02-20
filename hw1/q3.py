"""
q3.py — Repeated BACKWARD A* (Backward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- ESC or close window : quit

Maze file loader (optional helper): readFile(fname) reads 0/1 tokens (space-separated), 1=blocked, 0=free.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path (agent actually walked)
YELLOW = start + current agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG
from q2 import repeated_forward_astar


# ---------------- FILE LOADER ----------------
def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    mazes: List[List[List[int]]] = []
    for idx, grid in enumerate(data):
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = [[int(v) for v in row] for row in grid]
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)
    return mazes

def repeated_backward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # TODO: Implement Backward A* with max_g tie-braking strategy.
    # Use heapq for standard priority queue implementation and name your max_g heap class as `CustomPQ_maxG` and use it. 

    rows = len(actual_maze)
    cols = len(actual_maze[0])
    
    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
        return neighbors

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def compute_priority(g_val: int, h_val: int) -> int:
        f_val = g_val + h_val

        # For larger g: 
        C = 10 * rows * cols
        return C * f_val - g_val
        
    # Tracking variables
    current = start
    agent_path = [start]
    known_blocked = set()
    
    g = {}
    search = {}
    tree = {}
    
    counter = 0
    total_expansions = 0
    num_searches = 0

    while current != goal:
        counter += 1
        num_searches += 1

        g[goal] = 0
        search[goal] = counter
        g[current] = float('inf')
        search[current] = counter

        open_list = CustomPQ_maxG()
        
        h_start = manhattan_distance(goal, current)
        open_list.insert(compute_priority(0, h_start), goal)

        # Compute Path
        while not open_list.is_empty():
            min_priority, min_pos = open_list.peek()
            
            target_g = g.get(current, float('inf'))
            if target_g <= min_priority:
                if target_g != float('inf'):
                    break
                
            _, current_pos = open_list.extract_min()
            total_expansions += 1
            
            if visualize_callbacks and 'visited' in visualize_callbacks:
                visualize_callbacks['visited'](current_pos)

            r, c = current_pos
            for neighbor in get_neighbors(r, c):
                if neighbor in known_blocked:
                    continue
                
                if search.get(neighbor, 0) < counter:
                    g[neighbor] = float('inf')
                    search[neighbor] = counter
                
                new_g = g[current_pos] + 1
                
                if new_g < g[neighbor]:
                    g[neighbor] = new_g
                    tree[neighbor] = current_pos 
                    
                    h_val = manhattan_distance(neighbor, current)
                    priority = compute_priority(new_g, h_val)
                    open_list.insert(priority, neighbor)

        # Break if no path found
        if g.get(current, float('inf')) == float('inf'):
            return False, agent_path, total_expansions, num_searches

        # Get path
        path = []
        temp_pos = current
        while temp_pos != goal:
            temp_pos = tree[temp_pos]
            path.append(temp_pos)

        for next_step in path:
            # Check for blockage before moving
            nr, nc = next_step
            if actual_maze[nr][nc] == 1:
                known_blocked.add(next_step)
                break # Replan
            
            # Move agent
            current = next_step
            agent_path.append(current)
            
            # Check neighbors
            for neighbor in get_neighbors(current[0], current[1]):
                if actual_maze[neighbor[0]][neighbor[1]] == 1:
                    known_blocked.add(neighbor)
            
            if current == goal:
                return True, agent_path, total_expansions, num_searches

    return True, agent_path, total_expansions, num_searches

def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    # This function should display the maze used, the agent's knowledge, and the search process as the agent plans and executes.
    # As a reference, this function takes pygame Surface 'win' to draw on, the actual maze grid, the algorithm name for labeling, 
    # and optional parameters for controlling the visualization speed and saving a screenshot.
    # You are free to use other visualization libraries other than pygame. 
    # You can call repeated_forward_astar with visualize_callbacks that update the Pygame display as the agent plans and executes.
    # In the end it should store the visualization as a PNG file if save_path is provided, or default to "vis_{algo}.png".
    # print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    if save_path is None:
        save_path = f"vis_{algo}.png"

    # If 'win' is the display surface (it is), this works:
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Repeated Backward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q3.json",
                        help="Path to output JSON results file")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q3-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []
    total_bwd_expansions = 0

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_backward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
        )
        t1 = time.perf_counter()

        entry["bwd"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }
        total_bwd_expansions += expanded

        '''
        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_forward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
        )
        t1 = time.perf_counter()

        entry["fwd"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }
        '''
        results.append(entry)

    if len(mazes) > 0:
        avg_bwd_expansions = total_bwd_expansions / len(mazes)
        print(f"\nResults for {len(mazes)} mazes:")
        print(f"Average Cell Expansions (Backward): {avg_bwd_expansions:.2f}")

    if args.show_vis:
        # In case, PyGame is used for visualization, this code initializes a window and runs the visualization for the selected maze and algorithm.
        # Feel free to modify this code if you use a different visualization library or approach.
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Repeated Backward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()