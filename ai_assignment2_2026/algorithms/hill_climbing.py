import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# coordinate of the points/cities
coordinate = np.array(
    [[1, 2], [30, 21], [56, 23], [8, 18], [20, 50], [3, 4], [11, 6], [6, 7], [15, 20], [10, 9], [12, 12]])


# adjacency matrix for a weighted graph based on the given coordinates
def generate_matrix(coordinate):
    matrix = []
    for i in range(len(coordinate)):
        for j in range(len(coordinate)):
            p = np.linalg.norm(coordinate[i] - coordinate[j])
            matrix.append(p)
    matrix = np.reshape(matrix, (len(coordinate), len(coordinate)))
    print(matrix)
    return matrix


# finds a random solution
def solution(matrix):
    points = list(range(0, len(matrix)))
    solution = []
    for i in range(0, len(matrix)):
        random_point = points[random.randint(0, len(points) - 1)]
        solution.append(random_point)
        points.remove(random_point)
    return solution


# calculate the path based on the random solution
def path_length(matrix, solution):
    cycle_length = 0
    for i in range(0, len(solution)):
        cycle_length += matrix[solution[i]][solution[i - 1]]
    return cycle_length


# generate neighbors of the random solution by swapping cities and returns the best neighbor
def neighbors(matrix, solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbor = solution.copy()
            neighbor[i] = solution[j]
            neighbor[j] = solution[i]
            neighbors.append(neighbor)

    # assume that the first neighbor in the list is the best neighbor
    best_neighbor = neighbors[0]
    best_path = path_length(matrix, best_neighbor)

    # check if there is a better neighbor
    for neighbor in neighbors:
        current_path = path_length(matrix, neighbor)
        if current_path < best_path:
            best_path = current_path
            best_neighbor = neighbor
    return best_neighbor, best_path


def hill_climbing(coordinate):
    matrix = generate_matrix(coordinate)

    current_solution = solution(matrix)
    current_path = path_length(matrix, current_solution)
    neighbor = neighbors(matrix, current_solution)[0]
    best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)

    while best_neighbor_path < current_path:
        current_solution = best_neighbor
        current_path = best_neighbor_path
        neighbor = neighbors(matrix, current_solution)[0]
        best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)

    return current_path, current_solution


def visualize_solution(coordinate, solution_path, path_length_val, title="TSP Solution"):
    """
    Visualize the TSP solution with cities and the path taken
    """
    plt.figure(figsize=(10, 8))

    # Plot cities
    plt.scatter(coordinate[:, 0], coordinate[:, 1], c='red', s=200, zorder=5, label='Cities')

    # Label cities with their indices
    for i, coord in enumerate(coordinate):
        plt.annotate(f'{i}', (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Plot the path
    for i in range(len(solution_path)):
        start = coordinate[solution_path[i]]
        end = coordinate[solution_path[(i + 1) % len(solution_path)]]  # Connect last to first
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, alpha=0.6)

    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'{title}\nPath Length: {path_length_val:.2f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    return plt


final_solution = hill_climbing(coordinate)
print("The solution is \n", final_solution[1])
print(f"Final path length: {final_solution[0]:.2f}")

# Visualize the solution
visualize_solution(coordinate, final_solution[1], final_solution[0], "Hill Climbing TSP Solution")
plt.show()
