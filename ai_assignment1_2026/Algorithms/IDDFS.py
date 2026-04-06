from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig
class IDDFS(SequentialSearch):
    """
    Class for Iterative Deepening Depth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        
    def recursive_IDDFS(self, node_current,depth_limit ):
        # parse through all successors available starting from the current_node
        if depth_limit == 0:
            return False
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

            if depth_limit > 0:
                next_depth = None if depth_limit is None else depth_limit - 1
                goal_found = self.recursive_IDDFS(node_current=child, depth_limit=next_depth)

            # if a recursive successor returns with goal reached and a solution path, no further recursion is required
            if goal_found:
                return True
        return False
    
        
    def execute_search(self, time_pause):
        node_initial = self.initialize_search(time_pause=time_pause)
        depth_limit = 0

        while True:
            # now self.recursive_DFS is available from DepthFirstSearch
            found_path = self.recursive_IDDFS(node_current=node_initial, depth_limit = depth_limit)
            if found_path:
                return True
            depth_limit += 1
      