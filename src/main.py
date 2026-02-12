import os
import numpy as np
import random
from typing import List, Tuple, Optional, Set

from GridWorld import GridWorld
from a_star import RepeatedForwardAStar, RepeatedBackwardAStar, AdaptiveAStar
from visualization import save_replan_image, get_replan_dir

def load_maze(maze_number: int) -> np.ndarray:
    if not (-1 < maze_number < 50):
        raise ValueError(f"maze_number must be 0-49, got {maze_number}")
    padded_maze_number = f"{maze_number:02d}"
    maze = np.load(f'environments/maze_{padded_maze_number}.npy')
    return maze

def find_random_start_goal(gridworld: GridWorld):
    #runs until different
    while True: 
        #finds start pos
        unblocked_positions = np.argwhere(gridworld.grid == 0)
        random_idx = random.randint(0, len(unblocked_positions)-1)
        row_pos, col_pos = unblocked_positions[random_idx]
        start = (row_pos, col_pos)
        #finds end pos
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
    #run with smaller_g
    astar1 = RepeatedForwardAStar(grid, start, goal, 'smaller_g', False)
    success1 = astar1.run()
    
    #run with larger_g
    astar2 = RepeatedForwardAStar(grid, start, goal, 'larger_g', False)
    success2 = astar2.run()
    
    #return results
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
    #filter successful runs
    smaller = [r['smaller_g']['expansions']
               for r in all_results if r['smaller_g']['success']]
    larger = [r['larger_g']['expansions']
              for r in all_results if r['larger_g']['success']]

    print("\n-------------------------------")
    print("PART 2 RESULTS: Tie-breaking Strategies")
    print("-------------------------------")

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

    #forward A*
    forward = RepeatedForwardAStar(grid, start, goal, 'larger_g', False)
    success1 = forward.run()

    #backward A*
    backward = RepeatedBackwardAStar(grid, start, goal, 'larger_g', False)
    success2 = backward.run()

    return {
        'forward': {
            'success': success1,
            'expansions': forward.total_expansions,
            'searches': forward.num_searches
        },
        'backward': {
            'success': success2,
            'expansions': backward.total_expansions,
            'searches': backward.num_searches
        }
    }

def run_forward_adaptive_experiment(maze_number):
    maze = load_maze(maze_number)
    grid = GridWorld(maze)
    start, goal = find_random_start_goal(grid)

    results = {}

    #forward A*
    forward = RepeatedForwardAStar(grid, start, goal, 'larger_g', False)
    success1 = forward.run()

    #adaptive A*
    adaptive = AdaptiveAStar(grid, start, goal, False, 'larger_g')
    success2 = adaptive.run()

    return {
        'forward': {
            'success': success1,
            'expansions': forward.total_expansions,
            'searches': forward.num_searches
        },
        'adaptive': {
            'success': success2,
            'expansions': adaptive.total_expansions,
            'searches': adaptive.num_searches
        }
    }

def analyze_forward_backward_results(results):
    forward = [r['forward']['expansions'] 
               for r in results if r['forward']['success']]
    backward = [r['backward']['expansions'] 
                for r in results if r['backward']['success']]

    forward_search = [r['forward']['searches'] 
                      for r in results if r['forward']['success']]
    backward_search = [r['backward']['searches'] 
                       for r in results if r['backward']['success']]

    print("\n-------------------------------")
    print("PART 3 RESULTS: Forward vs Backward A*")
    print("-------------------------------")

    if not forward or not backward:
        print("No valid results to analyze.")
        return

    avg_forward = np.mean(forward)
    avg_backward = np.mean(backward)

    print("\nForward A* expansions:")
    print(f"  Average expansions: {avg_forward:.2f}")
    print(f"  Median expansions: {np.median(forward):.2f}")
    print(f"  Min expansions: {np.min(forward)}")
    print(f"  Max expansions: {np.max(forward)}")
    print(f"  Average searches: {np.mean(forward_search)}")

    print("\nBackward A* expansions:")
    print(f"  Average expansions: {avg_backward:.2f}")
    print(f"  Median expansions: {np.median(backward):.2f}")
    print(f"  Min expansions: {np.min(backward)}")
    print(f"  Max expansions: {np.max(backward)}")
    print(f"  Average searches: {np.mean(backward_search)}")

    diff = avg_backward - avg_forward
    print(f"Average difference (Backward − Forward): {diff:.2f}")



    if diff < 0:
        print("Conclusion: Backward A* expands fewer nodes on average than Forward A*.")
    else:
        print("Conclusion: Forward A* expands fewer nodes on average than Backward A*.")

def analyze_forward_adaptive_results(results):
    forward = [r['forward']['expansions'] 
               for r in results if r['forward']['success']]
    adaptive = [r['adaptive']['expansions'] 
                for r in results if r['adaptive']['success']]

    forward_search = [r['forward']['searches'] 
                      for r in results if r['forward']['success']]
    adaptive_search = [r['adaptive']['searches'] 
                       for r in results if r['adaptive']['success']]

    print("\n-------------------------------")
    print("PART 3 RESULTS: Forward vs adaptive A*")
    print("-------------------------------")

    if not forward or not adaptive:
        print("No valid results to analyze.")
        return

    avg_forward = np.mean(forward)
    avg_adaptive = np.mean(adaptive)

    print("\nForward A* expansions:")
    print(f"  Average expansions: {avg_forward:.2f}")
    print(f"  Median expansions: {np.median(forward):.2f}")
    print(f"  Min expansions: {np.min(forward)}")
    print(f"  Max expansions: {np.max(forward)}")
    print(f"  Average searches: {np.mean(forward_search)}")

    print("\nAdaptive A* expansions:")
    print(f"  Average expansions: {avg_adaptive:.2f}")
    print(f"  Median expansions: {np.median(adaptive):.2f}")
    print(f"  Min expansions: {np.min(adaptive)}")
    print(f"  Max expansions: {np.max(adaptive)}")
    print(f"  Average searches: {np.mean(adaptive_search)}")

    diff = avg_adaptive - avg_forward
    print(f"Average difference (Adpative − forward): {diff:.2f}")



    if diff < 0:
        print("Conclusion: Adaptive A* expands fewer nodes on average than Forward A*.")
    else:
        print("Conclusion: Forward A* expands fewer nodes on average than Adaptive A*.")

def run_single_maze(maze_number: int, start: Tuple[int, int], goal: Tuple[int, int], algorithm: str, tie_breaking: str, save_replan_images: bool = False,) -> bool:
    maze = load_maze(maze_number)
    grid = GridWorld(maze)
    on_replan = None
    if save_replan_images:
        replan_dir = get_replan_dir()
        replan_num = [0]
        def on_replan(current: Tuple[int, int], known_blocked: Set[Tuple[int, int]]) -> None:
            replan_num[0] += 1
            filepath = os.path.join(replan_dir, f"replan_{replan_num[0]:02d}.png")
            save_replan_image(maze, start, goal, current, known_blocked, filepath, title=f"Replan {replan_num[0]}")
    if algorithm == "forward":
        runner = RepeatedForwardAStar(grid, start, goal, tie_breaking, False, on_replan=on_replan)
    elif algorithm == "backward":
        runner = RepeatedBackwardAStar(grid, start, goal, tie_breaking, False, on_replan=on_replan)
    else:
        runner = AdaptiveAStar(grid, start, goal, False, tie_breaking, on_replan=on_replan)
    success = runner.run()
    print(f"Success: {success}, Searches: {runner.num_searches}, Expansions: {runner.total_expansions}")
    return success

def main():
    print("What would you like to do?")
    print(" 1) Run experiment")
    print(" 2) Run single maze")
    choice = input("Choice: ").strip()

    if choice == "1":
        print("Which experiment?")
        print(" 1) Tie-breaking (smaller_g vs larger_g)")
        print(" 2) Forward A* vs Backward A*")
        print(" 3) Forward A* vs Adaptive A*")
        exp = input("Choice: ").strip()
        if exp == "1":
            g_results = []
            for maze_num in range(50):
                try:
                    g_results.append(run_g_experiment(maze_num))
                except Exception as e:
                    print(f"Maze {maze_num}: ERROR - {e}")
            analyze_g_results(g_results)
        elif exp == "2":
            forward_backward_results = []
            for maze_num in range(50):
                try:
                    forward_backward_results.append(run_forward_backward_experiment(maze_num))
                except Exception as e:
                    print(f"Maze {maze_num}: ERROR - {e}")
            analyze_forward_backward_results(forward_backward_results)
        else:
            forward_adaptive_results = []
            for maze_num in range(50):
                try:
                    forward_adaptive_results.append(run_forward_adaptive_experiment(maze_num))
                except Exception as e:
                    print(f"Maze {maze_num}: ERROR - {e}")
            analyze_forward_adaptive_results(forward_adaptive_results)

    elif choice == "2":
        maze_number = int(input("Maze number (0-49): ").strip())
        maze = load_maze(maze_number)
        grid = GridWorld(maze)
        use_random = input("Random start and goal? (y/n): ").strip().lower()
        if use_random == "y":
            start, goal = find_random_start_goal(grid)
            print(f"Start: {start}, Goal: {goal}")
        else:
            r0, c0 = map(int, input("Start row col: ").split())
            r1, c1 = map(int, input("Goal row col: ").split())
            start, goal = (r0, c0), (r1, c1)
        print("Algorithm: 1=forward  2=backward  3=adaptive")
        algo_choice = input("Choice: ").strip()
        if algo_choice == "1":
            algorithm = "forward"
        elif algo_choice == "2":
            algorithm = "backward"
        else:
            algorithm = "adaptive"
        print("Tie-breaking: 1=smaller_g  2=larger_g")
        tie_choice = input("Choice: ").strip()
        tie_breaking = "smaller_g" if tie_choice == "1" else "larger_g"
        save_replan = input("Save image at each replan? (y/n): ").strip().lower() == "y"
        run_single_maze(maze_number, start, goal, algorithm, tie_breaking, save_replan_images=save_replan)

    else:
        print("Unknown choice.")

if __name__ == "__main__":
    main()


        
    



 