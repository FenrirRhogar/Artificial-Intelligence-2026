from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig

class IDDFS(SequentialSearch):
    """
    Class for Iterative Deepening Depth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        
    def execute_search(self, time_pause):
        # TODO: implement the execute_search method for Iterative Deepening Depth First Search algorithm
        print("TODO: implement the execute_search method for Iterative Deepening Depth First Search algorithm")
        return True