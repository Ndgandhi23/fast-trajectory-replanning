import numpy as np 
from typing import List, Tuple

class GridWorld: 
    def __init__(self, grid: np.ndarray): 
        self.grid = grid
        self.rows, self.cols = self.grid.shape

    def is_valid(self, row: int, col: int) -> bool: 
        if -1 < row < self.rows and -1 < col < self.cols: 
            return True
        return False
    
    def is_blocked(self, row, col):
        if self.grid[row, col] == 1: 
            return True
        return False

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        '''gets all possible neighbors'''
        neighbors = []
        possible_neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]

        for neighbor in possible_neighbors: 
            if self.is_valid(*neighbor): 
                neighbors.append(neighbor)
        return neighbors

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    

        
