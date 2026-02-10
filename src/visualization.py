import os
from typing import Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _repo_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_replan_dir() -> str:
    #return path to replan folder in repo
    replan_dir = os.path.join(_repo_dir(), "replan")
    os.makedirs(replan_dir, exist_ok=True)
    return replan_dir

def save_replan_image(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], current: Tuple[int, int], known_blocked: Set[Tuple[int, int]], filepath: str,title: str = "Replan",) -> None:
    cmap = ListedColormap(["white", "black"])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap=cmap, interpolation="nearest")
    ax.set_title(f"{title} ({maze.shape[0]}x{maze.shape[1]})", fontsize=16, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for (r, c) in known_blocked:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="gray", alpha=0.7))
    ax.plot(start[1], start[0], "s", color="green", markersize=8, label="Start")
    ax.plot(goal[1], goal[0], "*", color="red", markersize=12, label="Goal")
    ax.plot(current[1], current[0], "o", color="blue", markersize=8, label="Current")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filepath}")
