"""
gen_test_json.py — Generate N random 101x101 mazes and save as mazes.json. Uses same algorithm as maze_generator.py.

Usage:
    python gen_test_json.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
import random
from constants import ROWS
from tqdm import tqdm
import argparse
import numpy as np

# set random seed for reproducibility
random.seed(42)

def create_maze() -> list:
    # TODO: Implement this function to generate and return a random maze as a 2D list of 0s and 1s.
    # 0 = Unblocked, 1 = Blocked
    grid = np.zeros((ROWS, ROWS), dtype=np.int8)
    visited = np.zeros((ROWS, ROWS), dtype=bool)

    row_pos = random.randint(0, ROWS - 1)
    col_pos = random.randint(0, ROWS - 1)
    
    visited[row_pos, col_pos] = True
    grid[row_pos, col_pos] = 0
    stack = [(row_pos, col_pos)]
    
    while not np.all(visited):
        if not stack:
            # Re-seed the stack if it empties before all cells are visited
            unvisited_positions = np.argwhere(visited == False)
            random_idx = random.randint(0, len(unvisited_positions) - 1)
            row_pos, col_pos = unvisited_positions[random_idx]
            
            visited[row_pos, col_pos] = True
            # Even for new roots, we apply the block probability
            if random.random() < 0.3:
                grid[row_pos, col_pos] = 1
            else:
                grid[row_pos, col_pos] = 0
                stack.append((row_pos, col_pos))
        else:
            r, c = stack[-1]
            
            # Check unvisited neighbors
            unvisited_neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < ROWS and not visited[nr, nc]:
                    unvisited_neighbors.append((nr, nc))
            
            if unvisited_neighbors:
                # Pick random neighbor
                nr, nc = random.choice(unvisited_neighbors)
                visited[nr, nc] = True
                
                if random.random() < 0.3:
                    grid[nr, nc] = 1
                    # Do not add to stack if it's blocked
                else:
                    grid[nr, nc] = 0
                    stack.append((nr, nc))
            else:
                # No unvisited neighbors
                stack.pop()

    return grid.tolist()

def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)
    
    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):  
        mazes.append(create_maze())

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)
    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")

if __name__ == "__main__":
    main()
