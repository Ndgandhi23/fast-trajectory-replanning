import numpy as np 
from typing import Tuple, Dict, Set, List
from GridWorld import GridWorld
from binary_heap import BinaryHeap

class RepeatedForwardAStar: 
    def __init__(self, gridworld: GridWorld, start: Tuple[int, int], goal: Tuple[int, int], tie_breaking: str):
        self.gridworld = gridworld
        self.start = start
        self.goal = goal 
        self.tie_breaking = tie_breaking

        self.total_expansions = 0
        self.num_searches = 0

        self.known_blocked = set()

        self.counter = 0
        self.search = {}  # Maps position -> search number
        self.g = {}       # Maps position -> g-value
        self.tree = {}    # Maps position -> parent position

        self.C = 10 * GridWorld.rows * GridWorld.cols
    
    #helper 
    def heuristic(self, pos: Tuple[int, int]) -> int:
        return self.gridworld.manhattan_distance(pos, self.goal)
    
    def compute_priority(self, pos: Tuple[int, int], g_val: float) -> float: 
        f_val = g_val + self.heuristic(pos)

        #for step 2 comparison
        if self.tie_breaking == 'larger_g':
            return self.C * f_val - g_val
        else:
            return self.C * f_val + g_val
        
    def compute_path(self, current: Tuple[int, int]): 
        open = BinaryHeap()
        closed = set()

        priority = self.compute_priority(current, 0)
        open.insert(priority, current)

        while not open.is_empty(): 
            goal_g = self.g.get(self.goal, float('inf'))

            if open.peek() is None:
                break

            min_priority, min_pos = open.peek()
            min_f = self.g[min_pos] + self.heuristic(min_pos)

            if goal_g <= min_f:
                break
            
            _, current_pos = open.extract_min()
            
            if current_pos in closed:
                continue
            
            closed.add(current_pos)
            self.total_expansions += 1
            
            current_g = self.g[current_pos]
            
            for neighbor in self.gridworld.get_neighbors(current_pos[0], current_pos[1]):
                if neighbor in self.known_blocked:
                    continue
                
                if self.search.get(neighbor, 0) < self.counter:
                    self.g[neighbor] = float('inf')
                    self.search[neighbor] = self.counter
                
                new_g = current_g + 1  
                
                if new_g < self.g[neighbor]:
                    self.g[neighbor] = new_g
                    self.tree[neighbor] = current_pos
                    
                    new_priority = self.compute_priority(neighbor, new_g)
                    open.insert(new_priority, neighbor)
    
    def run(self):
        current = self.start
        
        while current != self.goal:
            self.counter += 1
            self.num_searches += 1
            
            self.g[current] = 0
            self.g[self.goal] = float('inf')
            self.search[current] = self.counter
            self.search[self.goal] = self.counter
            
            self.compute_path(current)
            
            if self.g[self.goal] == float('inf'):
                return False 
            
            path = []
            pos = self.goal
            while pos != current:
                path.append(pos)
                if pos not in self.tree:
                    return False 
                pos = self.tree[pos]
            path.reverse()
            
            for next_pos in path:
                for neighbor in self.gridworld.get_neighbors(current[0], current[1]):
                    if self.gridworld.is_blocked(neighbor[0], neighbor[1]):
                        self.known_blocked.add(neighbor)
            
                if next_pos in self.known_blocked:
                    break 
                
                current = next_pos
            
                if current == self.goal:
                    return True
    
        return True


class AdaptiveAStar(RepeatedForwardAStar):
    """
    Adaptive A* - improves on Repeated Forward A* by learning better heuristics.
    Updates h-values after each search to make future searches more efficient.
    """
    
    def __init__(self, gridworld: GridWorld, start: Tuple[int, int], 
                 goal: Tuple[int, int], tie_breaking: str = 'larger_g'):
        # Call parent constructor
        super().__init__(gridworld, start, goal, tie_breaking)
        
        self.h = {}

        for row in range(gridworld.rows):
            for col in range(gridworld.cols):
                pos = (row, col)
                self.h[pos] = gridworld.manhattan_distance(pos, goal)
    
    def heuristic(self, pos: Tuple[int, int]) -> int:
        """
        Override parent's heuristic to use learned h-values.
        Falls back to Manhattan distance if not yet computed.
        """
        if pos in self.h:
            return self.h[pos]
        return self.gridworld.manhattan_distance(pos, self.goal)
    
    def compute_path(self, current: Tuple[int, int]):

        open_list = BinaryHeap()
        closed = set()

        priority = self.compute_priority(current, 0)
        open_list.insert(priority, current)

        expanded_cells = []

        while not open_list.is_empty(): 
            goal_g = self.g.get(self.goal, float('inf'))

            if open_list.peek() is None:
                break

            min_priority, min_pos = open_list.peek()
            min_f = self.g[min_pos] + self._heuristic(min_pos)

            if goal_g <= min_f:
                break
            
            _, current_pos = open_list.extract_min()
            
            if current_pos in closed:
                continue
            
            closed.add(current_pos)
            expanded_cells.append(current_pos)
            self.total_expansions += 1
            
            current_g = self.g[current_pos]
            
            for neighbor in self.gridworld.get_neighbors(current_pos[0], current_pos[1]):
                if neighbor in self.known_blocked:
                    continue
                
                if self.search.get(neighbor, 0) < self.counter:
                    self.g[neighbor] = float('inf')
                    self.search[neighbor] = self.counter
                
                new_g = current_g + 1  
                
                if new_g < self.g[neighbor]:
                    self.g[neighbor] = new_g
                    self.tree[neighbor] = current_pos
                    
                    new_priority = self.compute_priority(neighbor, new_g)
                    open_list.insert(new_priority, neighbor)

        self._update_heuristics(expanded_cells)
    
    def _update_heuristics(self, expanded_cells: List[Tuple[int, int]]):
        goal_g = self.g.get(self.goal, float('inf'))
        if goal_g == float('inf'):
            return
        
        for cell in expanded_cells:
            if cell in self.g:
                new_h = goal_g - self.g[cell]
                
                if new_h > self.h.get(cell, 0):
                    self.h[cell] = new_h

            




