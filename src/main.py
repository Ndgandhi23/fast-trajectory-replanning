import numpy as np 
import random
from GridWorld import GridWorld
from a_star import RepeatedForwardAStar

def load_maze(maze_number: int) -> np.ndarray:
    if not (-1 < maze_number < 50):
        raise ValueError(f"maze_number must be 0-49, got {maze_number}")
    padded_maze_number = f"{maze_number:02d}"
    maze = np.load(f'environments/maze_{padded_maze_number}.npy')
    return maze

def find_random_start_goal(gridworld: GridWorld):
    #runs until different
    while True: 
        #Finds start pos
        unblocked_positions = np.argwhere(gridworld.grid == 0)
        random_idx = random.randint(0, len(unblocked_positions)-1)
        row_pos, col_pos = unblocked_positions[random_idx]
        start = (row_pos, col_pos)
        #Finds end pos
        unblocked_positions = np.argwhere(gridworld.grid == 0)
        random_idx = random.randint(0, len(unblocked_positions)-1)
        row_pos, col_pos = unblocked_positions[random_idx]
        end = (row_pos, col_pos)

        
        if start != end: 
            return start, end

def run_experiment(maze_number):
    maze = load_maze(maze_number)
    grid = GridWorld(maze)
    start, end = find_random_start_goal(grid)
    #step 2
    # Run with smaller_g
    astar1 = RepeatedForwardAStar(grid, start, goal, 'smaller_g')
    success1 = astar1.run()
    
    # Run with larger_g
    astar2 = RepeatedForwardAStar(grid, start, goal, 'larger_g')
    success2 = astar2.run()
    
    # Return results
    return {
        'maze_number': maze_number,
        'start': start,
        'goal': goal,
        'smaller_g': {
            'success': success1,
            'expansions': astar1.total_expansions,
            'searches': astar1.num_searches
        },
        'larger_g': {
            'success': success2,
            'expansions': astar2.total_expansions,
            'searches': astar2.num_searches
        }
    }

def main():
    """Run experiments on all 50 mazes."""
    print("CS 440 Assignment 1 - Part 2: The Effects of Ties")    
    # Store results
    all_results = []
    
    # Run experiments on all 50 mazes
    for maze_num in range(50):
        try:
            result = run_experiment(maze_num)
            all_results.append(result)
            
            # Print quick summary
            print(f"  Maze {maze_num}: smaller_g={result['smaller_g']['expansions']}, "
                  f"larger_g={result['larger_g']['expansions']}")
        except Exception as e:
            print(f"  Maze {maze_num}: ERROR - {e}")
    
    print("RESULTS SUMMARY")
    
    # Calculate statistics
    smaller_g_expansions = [r['smaller_g']['expansions'] for r in all_results 
                            if r['smaller_g']['success']]
    larger_g_expansions = [r['larger_g']['expansions'] for r in all_results 
                           if r['larger_g']['success']]
    
    if smaller_g_expansions and larger_g_expansions:
        print(f"\nsmaller_g strategy:")
        print(f"  Average expansions: {np.mean(smaller_g_expansions):.2f}")
        print(f"  Median expansions: {np.median(smaller_g_expansions):.2f}")
        print(f"  Min expansions: {np.min(smaller_g_expansions)}")
        print(f"  Max expansions: {np.max(smaller_g_expansions)}")
        
        print(f"\nlarger_g strategy:")
        print(f"  Average expansions: {np.mean(larger_g_expansions):.2f}")
        print(f"  Median expansions: {np.median(larger_g_expansions):.2f}")
        print(f"  Min expansions: {np.min(larger_g_expansions)}")
        print(f"  Max expansions: {np.max(larger_g_expansions)}")
        
        print(f"\nComparison:")
        avg_smaller = np.mean(smaller_g_expansions)
        avg_larger = np.mean(larger_g_expansions)
        difference = avg_larger - avg_smaller

        print(f"  Difference (larger_g - smaller_g): {difference:.2f}")
        
        if difference < 0:
            print(f"larger_g expanded fewer cells")
        else:
            print(f"smaller_g expanded fewer cells")


if __name__ == "__main__":
    main()



        
    



 