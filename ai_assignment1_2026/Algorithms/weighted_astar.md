# Implementation Notes - AI Assignment 1

## Weighted A* Search (`weighted_astar.py`)

I have implemented the **Weighted A* Search** algorithm with the following features:

### 1. Algorithm Structure
- Uses a `PriorityQueue` (from `SMP.motion_planner.queue`) to maintain the frontier.
- Implements the evaluation function: $f(n) = g(n) + w \cdot h(n)$.
- $g(n)$ is retrieved directly from the `CostNode.cost` attribute, which is automatically updated by `SequentialSearch.take_step` using the provided cost function.
- Maintains a `visited` dictionary (closed set) to prune states that have already been explored with a better or equal cost. States are represented by a tuple of (rounded) position $(x, y)$ and velocity.

### 2. Heuristics
Two heuristic functions $h(n)$ are supported:
- **Euclidean Distance**: Uses the base class method `calc_euclidean_distance`, which calculates the distance to the nearest point of the goal region.
- **Manhattan Distance**: A custom implementation calculating $|x_1 - x_2| + |y_1 - y_2|$ to the center of the goal.

### 3. Parameters
- **Weight ($w$)**: The default weight is set to `1.0`. It can be easily modified in the `__init__` method of the `weighted_astar` class for experimentation ($w=0, w=1, w>1$).
- **Heuristic Type**: Can be toggled between `"euclidean"` and `"manhattan"` in the `__init__` method.

### 4. Output
The implementation prints the required information to the console after a successful search:
- **Visited Nodes number**: The total number of nodes expanded (popped from the frontier and processed).
- **Path**: The sequence of $(x, y)$ coordinates from the start to the goal.
- **Estimated Cost**: The total accumulated cost $g(n)$ of the final path.

### 5. Verification
- The implementation was verified by running `main.py` with Scenario 1, successfully finding a path with 22 visited nodes and an estimated cost of 31.5.
