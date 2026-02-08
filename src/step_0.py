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
        #init ground rules on Maze 
        self.rows = rows 
        self.cols = cols 
        self.prob = prob
        #grid will show blocks 
        self.grid = None 
        #used in dfs for making maze to check if cell is visited
        self.visited = None 
    
    def is_valid(self, row: int, col: int) -> bool: 
        '''checks if cell is in range of maze'''
        if -1 < row < self.rows and -1 < col < self.cols:
            return True
        return False
    
    def get_neighbors(self, cell: Cell) -> List[Cell]:
        '''gets all possible neighbors'''
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
        '''gets all unvisited neighbors'''
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

        #inital cell when making map
        row_pos = random.randint(0, self.rows -1)
        col_pos = random.randint(0, self.cols -1)
        cell = Cell(row_pos, col_pos)

        self.visited[row_pos, col_pos] = True
        self.grid[row_pos, col_pos] = 0

        stack = [cell]

        #going through the maze 
        while not np.all(self.visited):
            #if stack is empty we will add a random unvisited to stack and mark it as not blocked 
            if not stack: 
                unvisited_positions = np.argwhere(self.visited == False)
                random_idx = random.randint(0, len(unvisited_positions)-1)
                row_pos, col_pos = unvisited_positions[random_idx]
                self.visited[row_pos, col_pos] = True
                stack.append(Cell(row_pos, col_pos))
            else: 
                '''
                look at top of stack and check unvisited neighbors 
                if unvisited neighbors exists, pick a random one, decide if blocked
                if no unvisted neighbors pop from stack 
                '''

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
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    print("=" * 60)
    print("CS 440 Assignment 1 - Step 0: Environment Generation")
    print("=" * 60)
    
    # Test with a small 10x10 maze first
    print("\n[TEST] Generating 10x10 test maze...")
    generator = MazeGenerator(rows=10, cols=10, prob=0.3)
    test_maze = generator.generate()
    
    print("Generated test maze:")
    print(test_maze)
    print(f"\nTest maze stats:")
    print(f"  Total cells: {test_maze.size}")
    print(f"  Blocked cells: {np.sum(test_maze == 1)}")
    print(f"  Unblocked cells: {np.sum(test_maze == 0)}")
    print(f"  Blocked percentage: {np.sum(test_maze == 1) / test_maze.size * 100:.1f}%")
    
    # Ask user if they want to generate all 50 mazes
    print("\n" + "=" * 60)
    response = input("Generate all 50 mazes (101x101)? (y/n): ").strip().lower()
    
    if response == 'y':
        # Create output directory
        output_dir = "environments"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating 50 mazes of size 101x101...")
        print(f"Block probability: 30%")
        print("=" * 60)
        
        generator = MazeGenerator(rows=101, cols=101, prob=0.3)
        stats = []
        
        for i in range(50):
            print(f"Generating maze {i+1}/50...", end=" ")
            
            # Generate the maze
            maze = generator.generate()
            
            # Calculate stats
            blocked_pct = (np.sum(maze == 1) / maze.size) * 100
            stats.append(blocked_pct)
            
            # Save it
            filename = os.path.join(output_dir, f"maze_{i:02d}.npy")
            np.save(filename, maze)
            print(f"✓ Saved ({blocked_pct:.1f}% blocked)")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"\nSummary Statistics:")
        print(f"  Average blocked: {np.mean(stats):.2f}%")
        print(f"  Min blocked: {np.min(stats):.2f}%")
        print(f"  Max blocked: {np.max(stats):.2f}%")
        print(f"  Std dev: {np.std(stats):.2f}%")
        print(f"\nAll 50 mazes saved to {output_dir}/")
        
        # Visualize a few sample mazes
        print("\n" + "=" * 60)
        viz_response = input("Visualize sample mazes? (y/n): ").strip().lower()
        
        if viz_response == 'y':
            # Color map: white for unblocked, black for blocked
            cmap = ListedColormap(['white', 'black'])
            
            # Visualize first 3 mazes
            for i in range(min(3, 50)):
                filename = os.path.join(output_dir, f"maze_{i:02d}.npy")
                maze = np.load(filename)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(maze, cmap=cmap, interpolation='nearest')
                ax.set_title(f"Maze {i} ({maze.shape[0]}x{maze.shape[1]})", 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                ax.grid(True, alpha=0.3, linewidth=0.5)
                plt.tight_layout()
                
                # Save visualization
                viz_dir = "../results"
                os.makedirs(viz_dir, exist_ok=True)
                viz_filename = os.path.join(viz_dir, f"maze_{i:02d}_visualization.png")
                plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
                print(f"Saved visualization: {viz_filename}")
                plt.close()
            
            print(f"\nVisualizations saved to {viz_dir}/")
    
    print("\n" + "=" * 60)
    print("Step 0 Complete!")
    print("=" * 60)