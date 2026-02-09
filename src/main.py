import numpy as np 
import random
from GridWorld import GridWorld
from a_star import RepeatedForwardAStar
from a_star import RepeatedBackwardAStar

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
        goal = (row_pos, col_pos)

        
        if start != goal: 
            return start, goal

def run_g_experiment(maze_number):
    maze = load_maze(maze_number)
    grid = GridWorld(maze)
    start, goal = find_random_start_goal(grid)
    #step 2
    # Run with smaller_g
    astar1 = RepeatedForwardAStar(grid, start, goal, 'smaller_g', False)
    success1 = astar1.run()
    
    # Run with larger_g
    astar2 = RepeatedForwardAStar(grid, start, goal, 'larger_g', False)
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

def analyze_g_results(all_results):
    # Filter successful runs
    smaller = [r['smaller_g']['expansions']
               for r in all_results if r['smaller_g']['success']]
    larger = [r['larger_g']['expansions']
              for r in all_results if r['larger_g']['success']]

    print("\n==============================")
    print("PART 2 RESULTS: Tie-breaking Strategies")
    print("==============================")

    if not smaller or not larger:
        print("No valid results to analyze.")
        return

    avg_smaller = np.mean(smaller)
    avg_larger = np.mean(larger)

    print("\nsmaller_g strategy:")
    print(f"  Average expansions: {avg_smaller:.2f}")
    print(f"  Median expansions: {np.median(smaller):.2f}")
    print(f"  Min expansions: {np.min(smaller)}")
    print(f"  Max expansions: {np.max(smaller)}")

    print("\nlarger_g strategy:")
    print(f"  Average expansions: {avg_larger:.2f}")
    print(f"  Median expansions: {np.median(larger):.2f}")
    print(f"  Min expansions: {np.min(larger)}")
    print(f"  Max expansions: {np.max(larger)}")

    diff = avg_larger - avg_smaller
    print(f"\nAverage difference (larger_g − smaller_g): {diff:.2f}")

    if diff < 0:
        print("Conclusion: Breaking ties in favor of larger g-values expands fewer cells on average.")
    else:
        print("Conclusion: Breaking ties in favor of smaller g-values expands fewer cells on average.")


def run_forward_backward_experiment(maze_number):
    maze = load_maze(maze_number)
    grid = GridWorld(maze)
    start, goal = find_random_start_goal(grid)

    results = {}

    # Forward A*
    forward = RepeatedForwardAStar(grid, start, goal, 'larger_g', False)
    forward.run()
    results["forward"] = forward.total_expansions

    # Backward A*
    backward = RepeatedBackwardAStar(grid, start, goal, 'larger_g', False)
    backward.run()
    results["backward"] = backward.total_expansions

    return results

def analyze_forward_backward_results(results):
    forward = [r['forward'] for r in results]
    backward = [r['backward'] for r in results]

    print("\n==============================")
    print("PART 3 RESULTS: Forward vs Backward A*")
    print("==============================")

    if not forward or not backward:
        print("No valid results to analyze.")
        return

    avg_forward = np.mean(forward)
    avg_backward = np.mean(backward)

    print(f"Average Forward A* expansions: {avg_forward:.2f}")
    print(f"Average Backward A* expansions: {avg_backward:.2f}")

    diff = avg_backward - avg_forward
    print(f"Average difference (Backward − Forward): {diff:.2f}")

    if diff < 0:
        print("Conclusion: Backward A* expands fewer nodes on average than Forward A*.")
    else:
        print("Conclusion: Forward A* expands fewer nodes on average than Backward A*.")

def main():
    """Run experiments on all 50 mazes."""
    print("CS 440 Assignment 1 - Part 2: The Effects of Ties")    
    # Store results
    g_results = []
    forward_backward_results = []
    
    # Run experiments on all 50 mazes
    for maze_num in range(5):
        try:
            #g_res = run_g_experiment(maze_num)
            #g_results.append(g_res)

            forward_backward_res = run_forward_backward_experiment(maze_num)
            forward_backward_results.append(forward_backward_res)
        except Exception as e:
            print(f"  Maze {maze_num}: ERROR - {e}")

    #analyze_g_results(g_results)
    analyze_forward_backward_results(forward_backward_results)
    

if __name__ == "__main__":
    main()


        
    



 