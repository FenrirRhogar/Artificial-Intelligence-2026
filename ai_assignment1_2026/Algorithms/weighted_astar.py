from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import PriorityQueue

class weighted_astar(SequentialSearch):
    """
    Class for Weighted A* Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        # Default weight and heuristic type
        # These can be modified for experimentation
        self.w = 1.0 
        self.heuristic_type = "euclidean" # options: "euclidean", "manhattan"

    def execute_search(self, time_pause):
        """
        Implementation of Weighted A* Search.
        f(n) = g(n) + w * h(n)
        """
        # Initialize search and get the initial node
        node_initial = self.initialize_search(time_pause=time_pause, cost=True)
        
        # Priority Queue for the frontier
        frontier = PriorityQueue()
        
        # Calculate initial f(n)
        f_initial = node_initial.cost + self.w * self.heuristic_function(node_initial)
        frontier.insert(node_initial, f_initial)
        
        # Closed set to keep track of visited states and their minimum cost g(n)
        # Using a dictionary: state_representation -> min_g_cost
        # For simplicity in this motion planning context, we use the final position of the node
        visited = {} 
        
        # Counter for expanded nodes
        visited_nodes_count = 0
        
        while not frontier.empty():
            # Pop the node with the lowest f(n)
            node_current = frontier.pop()
            
            # Check if we have already visited this state with a better or equal cost
            # State is represented by position and velocity (since velocity affects future costs/moves)
            state_last = node_current.list_paths[-1][-1]
            state_key = (round(state_last.position[0], 2), 
                         round(state_last.position[1], 2), 
                         round(state_last.velocity, 2))
            
            if state_key in visited and visited[state_key] <= node_current.cost:
                continue
            
            visited[state_key] = node_current.cost
            visited_nodes_count += 1
            
            # Check all possible successors from the current maneuver automaton
            for primitive_successor in node_current.get_successors():
                
                # Check if goal is reached via this primitive
                # We do this BEFORE take_step to match the provided goal_reached structure
                # which handles visualization and pruning if goal is hit.
                if self.goal_reached(successor=primitive_successor, node_current=node_current):
                    # Reconstruct path and info for final output
                    # Note: goal_reached internally handles the solution plotting
                    
                    # We need the child node to get the final cost and path
                    _, child = self.take_step(successor=primitive_successor, node_current=node_current, cost=True)
                    
                    print(f"Weighted A* (w={self.w}):")
                    print(f"    Visited Nodes number: {visited_nodes_count}")
                    
                    path = self.get_node_path(child)
                    path_str = "->".join([f"({p[0]:.2f},{p[1]:.2f})" for p in path])
                    print(f"    Path: {path_str}")
                    print(f"    Estimated Cost: {child.cost:.4f}")
                    
                    return True

                # Execute step and check for collisions
                collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current, cost=True)
                
                if collision_flag:
                    continue

                # Calculate f(n) = g(n) + w * h(n)
                # g(n) is child.cost (accumulated cost from start)
                f_child = child.cost + self.w * self.heuristic_function(child)
                
                # Add child to frontier
                frontier.insert(child, f_child)
                
        print("Path not found.")
        return False
    
    def heuristic_function(self, node_current):
        """
        Heuristic function h(x) calculation of distance from node_current to goal.
        """
        if self.heuristic_type == "euclidean":
            # Uses the provided method in SearchBaseClass which handles goal regions
            return self.calc_euclidean_distance(node_current)
        
        elif self.heuristic_type == "manhattan":
            # Custom Manhattan distance to the center of the goal
            pos_current = node_current.list_paths[-1][-1].position
            goal_info = self.get_goal_information() # [x, y, length, width]
            pos_goal = [goal_info[0], goal_info[1]]
            
            # |x1 - x2| + |y1 - y2|
            distance = abs(pos_current[0] - pos_goal[0]) + abs(pos_current[1] - pos_goal[1])
            return distance
            
        return 0
