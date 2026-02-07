# CS 440 Assignment 1 - Fast Trajectory Replanning

## Team Members
- Neil Gandhi
- Dev
- Sandy

## Setup Instructions

### Prerequisites
Install `uv`:
- Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Installation
```bash
git clone <repo-url>
cd fast-trajectory-replanning
uv sync
```

### Run
```bash
uv run python src/main.py
```

## Project Structure
```
src/          # Source code
tests/        # Test files
environments/ # Generated gridworlds
results/      # Experimental results
docs/         # Report files
```
