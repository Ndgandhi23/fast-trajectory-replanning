"""CS 440 Assignment 1 - Fast Trajectory Replanning"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("CS 440 Assignment 1 - Fast Trajectory Replanning")
    print("NumPy version:", np.__version__)
    print("Setup successful! ✓")
    
    # Test visualization
    grid = np.zeros((5, 5))
    grid[1:3, 2:4] = 1  # Some blocked cells
    
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap='binary')
    plt.title("Test Grid")
    plt.savefig('results/test_grid.png')
    print("Test grid saved to results/test_grid.png")

if __name__ == "__main__":
    main()
