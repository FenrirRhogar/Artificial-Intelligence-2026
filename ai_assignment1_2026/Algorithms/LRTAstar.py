from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig

class LRTAstar(SequentialSearch):
    """
    Class for LRTA* Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
    
    def execute_search(self, time_pause):
        # TODO: implement the execute_search method for LRTA* Search algorithm
        print("TODO: implement the execute_search method for LRTA* Search algorithm")
        return True
    
    def heuristic_function(self, node_current):
        """
        TODO: Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """
        distance = 0
        return distance
 