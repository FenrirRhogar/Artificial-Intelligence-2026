from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig
class IDDFS(SequentialSearch):
    """
    Class for Iterative Deepening Depth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        self._visited_count = 0
        self._found_node = None

    def recursive_IDDFS(self, node_current, depth_limit):
        # parse through all successors available starting from the current_node
        if depth_limit == 0:
            return False
        for primitive_successor in node_current.get_successors():

            # execute step from node_current to primitive_successor
            collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current, cost=True)

            # if it collides with an obstacle or boundary skip this successor
            if collision_flag:
                continue

            self._visited_count += 1

            # check whether goal is reached
            goal_flag = self.goal_reached(successor=primitive_successor, node_current=node_current)
            # if goal is reached, store the node and return
            if goal_flag:
                self._found_node = child
                return True

            goal_found = self.recursive_IDDFS(node_current=child, depth_limit=depth_limit - 1)

            # if a recursive successor returns with goal reached, no further recursion is required
            if goal_found:
                return True
        return False

    def execute_search(self, time_pause):
        self._visited_count = 0
        self._found_node = None
        node_initial = self.initialize_search(time_pause=time_pause, cost=True)
        depth_limit = 0

        while True:
            found_path = self.recursive_IDDFS(node_current=node_initial, depth_limit=depth_limit)
            if found_path:
                path = self.get_node_path(self._found_node)
                path_str = "->".join([f"({p[0]:.2f},{p[1]:.2f})" for p in path])
                print(f"Iterative-Deepening DFS:")
                print(f"    Visited Nodes number: {self._visited_count}")
                print(f"    Path: {path_str}")
                print(f"    Estimated Cost: {self._found_node.cost:.4f}")
                return True
            depth_limit += 1
      