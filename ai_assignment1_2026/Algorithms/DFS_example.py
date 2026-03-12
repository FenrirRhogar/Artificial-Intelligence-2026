
from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig

class DepthFirstSearch(SequentialSearch):
    """
    Class for Depth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

    def recursive_DFS(self, node_current):
        # parse through all successors available starting from the current_node
        for primitive_successor in node_current.get_successors():

            # execute step from node_current to primitive_successor
            collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current)
            # print("Node position is: ", self.get_node_information(child))
            # print("And path to get here is: ", self.get_node_path(child))

            # if it collides with an obstacle or boundary skip this successor
            if collision_flag:
                continue

            # check whether goal is reached
            goal_flag = self.goal_reached(successor=primitive_successor,
                                                                     node_current=node_current)
            # if goal is reached, return back with the solution path
            if goal_flag:
                return True

            # if goal is not reached, continue the search recursive
            goal_found = self.recursive_DFS(node_current=child)

            # if a recursive successor returns with goal reached and a solution path, no further recursion is required
            if goal_found:
                return True
        return False

    def execute_search(self, time_pause):
        node_initial = self.initialize_search(time_pause=time_pause)
        # print(self.get_obstacles_information())
        # print(self.get_goal_information())
        # print(self.get_node_information(node_initial))
        found_path = self.recursive_DFS(node_current=node_initial)
        return found_path