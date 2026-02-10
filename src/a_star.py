import numpy as np 
from typing import Tuple, Dict, Set, List
from GridWorld import GridWorld
from binary_heap import BinaryHeap

#helper for debugging
def fmt(pos):
    return f"({pos[0]},{pos[1]})"

class RepeatedForwardAStar: 
    def __init__(self, gridworld: GridWorld, start: Tuple[int, int], goal: Tuple[int, int], tie_breaking: str, debug:bool):
        self.gridworld = gridworld
        self.start = start
        self.goal = goal 
        self.tie_breaking = tie_breaking
        self.debug = debug

        self.total_expansions = 0
        self.num_searches = 0

        self.known_blocked = set()

        self.counter = 0
        self.search = {}  # Maps position -> search number
        self.g = {}       # Maps position -> g-value
        self.tree = {}    # Maps position -> parent position

        self.C = 10 * self.gridworld.rows * self.gridworld.cols
    
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

            #DEBUGGING
            if self.debug:
                print(
                    f"EXPAND {fmt(current_pos)} "
                    f"g={self.g[current_pos]} "
                    f"h={self.heuristic(current_pos)} "
                    f"f={self.g[current_pos] + self.heuristic(current_pos)}"
                )
            
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

                    #DEBUGGING
                    if self.debug:
                        print(
                            f"  UPDATE {fmt(neighbor)} "
                            f"g={new_g} parent={fmt(current_pos)}"
                        )
    
    def run(self):
        current = self.start
        
        while current != self.goal:
            self.counter += 1
            self.num_searches += 1
            
            #Moved to be before compute path so it actually looks at neighbors before it moves
            for neighbor in self.gridworld.get_neighbors(current[0], current[1]):
                    if self.gridworld.is_blocked(neighbor[0], neighbor[1]):
                        #DEBUGGING
                        if self.debug:
                            if neighbor not in self.known_blocked:
                                print(f"OBSERVE BLOCKED at {fmt(neighbor)}")
                        self.known_blocked.add(neighbor)

            #DEBUGGING
            if self.debug:
                print("\n==============================")
                print(f"NEW SEARCH #{self.counter}")
                print(f"Agent position: {fmt(current)}")
                print(f"Known blocked cells: {self.known_blocked}")
            
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
                #DEBUGGING
                if self.debug:
                    print(f"MOVE {fmt(current)} -> {fmt(next_pos)}")
            
                if next_pos in self.known_blocked:
                    #DEBUGGING
                    if self.debug:
                        print(f"REPLAN triggered: {fmt(next_pos)} is blocked")
                    break 
                
                current = next_pos

                for neighbor in self.gridworld.get_neighbors(current[0], current[1]):
                    if self.gridworld.is_blocked(neighbor[0], neighbor[1]):
                        #DEBUGGING
                        if self.debug:
                            if neighbor not in self.known_blocked:
                                print(f"OBSERVE BLOCKED at {fmt(neighbor)}")
                        self.known_blocked.add(neighbor)
            
                if current == self.goal:
                    return True
    
        return True


class AdaptiveAStar(RepeatedForwardAStar):
    """
    Adaptive A* - improves on Repeated Forward A* by learning better heuristics.
    Updates h-values after each search to make future searches more efficient.
    """
    
    def __init__(self, gridworld: GridWorld, start: Tuple[int, int], 
                 goal: Tuple[int, int], debug:bool, tie_breaking: str = 'larger_g'):
        # Call parent constructor
        super().__init__(gridworld, start, goal, tie_breaking, debug)
        
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
        expanded_cells = []

        priority = self.compute_priority(current, 0)
        open_list.insert(priority, current)

        while not open_list.is_empty(): 
            goal_g = self.g.get(self.goal, float('inf'))

            if open_list.peek() is None:
                break

            min_priority, min_pos = open_list.peek()
            min_f = self.g[min_pos] + self.heuristic(min_pos)

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

        #self._update_heuristics(expanded_cells)
        goal_g = self.g.get(self.goal, float('inf'))
        if goal_g < float("inf"):
            for c in expanded_cells:
                self.h[c] = goal_g - self.g[c]
    
    def _update_heuristics(self, expanded_cells: List[Tuple[int, int]]):
        goal_g = self.g.get(self.goal, float('inf'))
        if goal_g == float('inf'):
            return
        
        for cell in expanded_cells:
            if cell in self.g:
                new_h = goal_g - self.g[cell]
                
                if new_h > self.h.get(cell, 0):
                    self.h[cell] = new_h

class RepeatedBackwardAStar(RepeatedForwardAStar): 
    def heuristic(self, pos: Tuple[int, int]) -> int:
        return self.gridworld.manhattan_distance(self.goal, pos)
        
    def compute_path(self, current: Tuple[int, int]): 
        open = BinaryHeap()
        closed = set()

        self.g[self.goal]=0
        self.search[self.goal]= self.counter

        priority = self.compute_priority(self.goal, 0)
        open.insert(priority, self.goal)

        while not open.is_empty(): 
            agent_g = self.g.get(current, float('inf'))

            if open.peek() is None:
                break

            min_priority, min_pos = open.peek()
            min_f = self.g[min_pos] + self.heuristic(min_pos)

            if agent_g <= min_f:
                break
            
            _, current_pos = open.extract_min()
            
            if current_pos in closed:
                continue
            
            closed.add(current_pos)
            self.total_expansions += 1

            #DEBUGGING
            if self.debug:
                print(
                    f"EXPAND {fmt(current_pos)} "
                    f"g={self.g[current_pos]} "
                    f"h={self.heuristic(current_pos)} "
                    f"f={self.g[current_pos] + self.heuristic(current_pos)}"
                )
            
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

                    #DEBUGGING
                    if self.debug:
                        print(
                            f"  UPDATE {fmt(neighbor)} "
                            f"g={new_g} parent={fmt(current_pos)}"
                        )

    
    def run(self):
        current = self.start
        
        while current != self.goal:
            self.counter += 1
            self.num_searches += 1

            for neighbor in self.gridworld.get_neighbors(current[0], current[1]):
                    if self.gridworld.is_blocked(neighbor[0], neighbor[1]):
                        #DEBUGGING
                        if self.debug:
                            if neighbor not in self.known_blocked:
                                print(f"OBSERVE BLOCKED at {fmt(neighbor)}")
                        self.known_blocked.add(neighbor)
            
            #DEBUGGING
            if self.debug:
                print("\n==============================")
                print(f"NEW SEARCH #{self.counter}")
                print(f"Agent position: {fmt(current)}")
                print(f"Known blocked cells: {self.known_blocked}")

            self.g[current] = float('inf')
            self.g[self.goal] = 0
            self.search[current] = self.counter
            self.search[self.goal] = self.counter
            
            self.compute_path(current)
            
            if self.g[current] == float('inf'):
                return False 
            
            path = []
            pos = current
            while pos != self.goal:
                if pos not in self.tree:
                    return False 
                pos = self.tree[pos]
                path.append(pos)
            
            for next_pos in path:
                #DEBUGGING
                if self.debug:
                    print(f"MOVE {fmt(current)} -> {fmt(next_pos)}")
            
                if next_pos in self.known_blocked:
                    #DEBUGGING
                    if self.debug:
                        print(f"REPLAN triggered: {fmt(next_pos)} is blocked")
                    break 
                
                current = next_pos

                for neighbor in self.gridworld.get_neighbors(current[0], current[1]):
                    if self.gridworld.is_blocked(neighbor[0], neighbor[1]):
                        #DEBUGGING
                        if self.debug:
                            if neighbor not in self.known_blocked:
                                print(f"OBSERVE BLOCKED at {fmt(neighbor)}")
                        self.known_blocked.add(neighbor)

                if current == self.goal:
                    return True
    
        return True
    

#DEBUGGING TESTS
def main():
    # 0 = free, 1 = blocked
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ])

    gw = GridWorld(grid)

    start = (4, 2)
    goal = (4, 4)

    print("\n-------------------------------")
    print("RUNNING REPEATED FORWARD A*")
    print("-------------------------------")

    forward_astar = RepeatedForwardAStar(
        gridworld=gw,
        start=start,
        goal=goal,
        tie_breaking="larger_g",
        debug=True
    )

    forward_result = forward_astar.run()

    print("\nFORWARD A* RESULT:", forward_result)
    print("Total searches:", forward_astar.num_searches)
    print("Total expansions:", forward_astar.total_expansions)

    print("\n-------------------------------")
    print("RUNNING APDATIVE A*")
    print("-------------------------------")

    adaptive_astar = AdaptiveAStar(
        gridworld=gw,
        start=start,
        goal=goal,
        debug=True,
        tie_breaking="larger_g"
    )

    adaptive_result = adaptive_astar.run()

    print("\nAdaptive A* RESULT:", adaptive_result)
    print("Total searches:", adaptive_astar.num_searches)
    print("Total expansions:", adaptive_astar.total_expansions)

    print("\n-------------------------------")
    print("RUNNING REPEATED BACKWARD A*")
    print("-------------------------------")

    backward_astar = RepeatedBackwardAStar(
        gridworld=gw,
        start=start,
        goal=goal,
        tie_breaking="larger_g",
        debug=True
    )

    backward_result = backward_astar.run()

    print("\nBACKWARD A* RESULT:", backward_result)
    print("Total searches:", backward_astar.num_searches)
    print("Total expansions:", backward_astar.total_expansions)


if __name__ == "__main__":
    main()





