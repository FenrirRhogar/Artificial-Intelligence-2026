from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig
import math
import copy


class LRTAstar(SequentialSearch):
    """
    Class for LRTA* (Learning Real-Time A*) Search algorithm.
    Korf, 1990: interleaves acting and heuristic value updates.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         automaton=automaton, plot_config=plot_config)
        # H-table: stores learned heuristic values per visited position
        # key: tuple(position), value: float
        self.h_table = {}

    def heuristic_function(self, node_current):
        """
        h(n): Euclidean distance from node_current to the goal center.
        Checks the h_table first — if a learned value exists, use it.
        """
        pos = tuple(self.get_node_information(node_current))

        # Return learned (updated) heuristic if available
        if pos in self.h_table:
            return self.h_table[pos]

        goal_center = self.get_goal_information()   # [x, y, length, width]
        dx = goal_center[0] - pos[0]
        dy = goal_center[1] - pos[1]
        return math.sqrt(dx**2 + dy**2)

    def evaluation_function(self, node_current):
        """f(n) = g(n) + h(n)"""
        return self.cost_function(node_current) + self.heuristic_function(node_current)

    def execute_search(self, time_pause):
        """
        LRTA* main loop.

        At each state:
          1. Check all successors for goal.
          2. Compute f(s') = cost_to_reach(s') + h(s') for each successor s'.
          3. Update h(current) <- min f(s')  [the learning step].
          4. Move to the successor s* = argmin f(s').
        Repeats until goal is reached or no successors remain.
        """
        node_current = self.initialize_search(time_pause=time_pause, cost=True)

        while True:
            successors = node_current.get_successors()
            if not successors:
                # Dead end — no valid moves
                return False

            best_child = None
            best_f = float('inf')
            best_successor = None
            collision_free_found = False

            # evaluate all successors 
            for primitive_successor in successors:

                if self.goal_reached(successor=primitive_successor,
                                     node_current=node_current):
                    return True

                collision_flag, child = self.take_step(
                    successor=primitive_successor,
                    node_current=node_current,
                    cost=True
                )

                if collision_flag:
                    continue  # Skip colliding primitives

                collision_free_found = True
                f_val = self.cost_function(child) + self.heuristic_function(child)

                if f_val < best_f:
                    best_f = f_val
                    best_child = child
                    best_successor = primitive_successor

            if not collision_free_found or best_child is None:
                return False  

            #  LRTA* learning — update h(current) 
            current_pos = tuple(self.get_node_information(node_current))
            current_h = self.heuristic_function(node_current)
            # h(s) <- max(h(s), min_f) 
            self.h_table[current_pos] = max(current_h, best_f)

            # move to best successor 
            node_current = best_child