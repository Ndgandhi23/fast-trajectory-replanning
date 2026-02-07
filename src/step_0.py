import numpy as np #grid arrays
import random #for the tie-breaks
from dataclasses import dataclass #makes Cell class cleaner
from typing import List #type hints

@dataclass
class Cell: 
    row: int 
    col: int

class MazeGenerator: 
    '''Generates maze grid worlds 50 by 50s'''

    def __init__(self, rows: int = 101, cols: int = 101, prob: float = 0.3):
        self.rows = rows 
        self.cols = cols 
        self.prob = prob
        self.grid = None 
        self.visited = None 
    
    def is_valid(self, row: int, col: int) -> bool: 
        if -1 < row < self.rows and -1 < col < self.cols:
            return True
        return False
    
    def get_neighbors(self, cell: Cell) -> List[Cell]:
        neighbors = []
        possible_neighbors = [
            (cell.row - 1, cell.col),
            (cell.row + 1, cell.col),
            (cell.row, cell.col - 1),
            (cell.row, cell.col + 1)
        ]

        for neighbor in possible_neighbors: 
            if self.is_valid(*neighbor): 
                neighbors.append(Cell(*neighbor))
        return neighbors
    
    def get_unvisited_neighbors(self, cell: Cell) -> List[Cell]: 
        unvisited_neighbors = []
        neighbors = self.get_neighbors(cell)
        for neighbor in neighbors: 
            if self.visited[neighbor.row, neighbor.col] == False:
                unvisited_neighbors.append(neighbor)
        return unvisited_neighbors
    
    def generate(self) -> np.ndarray: 
        #generate the field
        self.grid = np.zeros((self.rows, self.cols), dtype = np.int8)
        self.visited = np.zeros((self.rows, self.cols), dtype = bool)

        #start stuff
        row_pos = random.randint(0, self.rows -1)
        col_pos = random.randint(0, self.cols -1)
        cell = Cell(row_pos, col_pos)

        self.visited[row_pos, col_pos] = True
        self.grid[row_pos, col_pos] = 0

        stack = [cell]

        #going through the maze 
        while not np.all(self.visited):
            #if stack is empty 
            if not stack: 
                unvisited_positions = np.argwhere(self.visited == False)
                random_idx = random.randint(0, len(unvisited_positions)-1)
                row_pos, col_pos = unvisited_positions[random_idx]
                self.visited[row_pos, col_pos] = True
                stack.append(Cell(row_pos, col_pos))
            else: 
                cell = stack[-1]
                unvisited_neighbors = self.get_unvisited_neighbors(cell)
                if unvisited_neighbors: 
                    random_idx = random.randint(0, len(unvisited_neighbors) - 1)
                    neighbor = unvisited_neighbors[random_idx]
                    self.visited[neighbor.row, neighbor.col] = True
                    if random.random() < self.prob: 
                        self.grid[neighbor.row, neighbor.col] = 1
                    else: 
                        stack.append(Cell(neighbor.row, neighbor.col))
                else: 
                    stack.pop()
        return self.grid 
    


if __name__ == "__main__":
    print("Testing Step 0: Maze Generation")
    print("=" * 50)

    # Test with a small 10x10 maze first
    print("\nGenerating 10x10 test maze...")
    generator = MazeGenerator(rows=10, cols=10, prob=0.3)
    maze = generator.generate()

    print("Generated maze:")
    print(maze)

    print("\nMaze stats:")
    print(f"  Total cells: {maze.size}")
    print(f"  Blocked cells: {np.sum(maze == 1)}")
    print(f"  Unblocked cells: {np.sum(maze == 0)}")
    print(f"  Blocked percentage: {np.sum(maze == 1) / maze.size * 100:.1f}%")
