from typing import Union
from enum import Enum, unique
import sys

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
sys.path.append('../../')
from Algorithms.DFS_example import DepthFirstSearch
from Algorithms.weighted_astar import weighted_astar
from Algorithms.LRTAstar import LRTAstar
from Algorithms.IDDFS import IDDFS

@unique
class MotionPlannerType(Enum):
    """
    Enumeration definition of different algorithms.
    """
    DFS = "dfs"
    WEIGHTED_ASTAR = "weighted_astar"
    LRTASTAR = "lrtastar"
    IDDFS = "iddfs"


class MotionPlanner:
    """
    Class to load and execute the specified motion planner.
    """

    class NoSuchMotionPlanner(KeyError):
        """
        Error message when the specified motion planner does not exist.
        """

        def __init__(self, message):
            self.message = message

    dict_motion_planners = dict()
    dict_motion_planners[MotionPlannerType.DFS] = DepthFirstSearch
    dict_motion_planners[MotionPlannerType.WEIGHTED_ASTAR] = weighted_astar
    dict_motion_planners[MotionPlannerType.LRTASTAR] = LRTAstar
    dict_motion_planners[MotionPlannerType.IDDFS] = IDDFS

    @classmethod
    def create(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: ManeuverAutomaton,
               plot_config=DefaultPlotConfig,
               motion_planner_type: MotionPlannerType = MotionPlannerType.DFS) -> Union[DepthFirstSearch,
                                                                                         weighted_astar,
                                                                                         LRTAstar,
                                                                                         IDDFS]:
        """
        Method to instantiate the specified motion planner.
        """
        try:
            return cls.dict_motion_planners[motion_planner_type](scenario, planning_problem, automaton,
                                                                 plot_config=plot_config)
        except KeyError:
            raise cls.NoSuchMotionPlanner(f"MotionPlanner with type <{motion_planner_type}> does not exist.")

    @classmethod
    def weighted_astar(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: ManeuverAutomaton,
                           plot_config=DefaultPlotConfig) -> weighted_astar:
        """
        Method to instantiate a weighted A* Search motion planner.
        """
        return MotionPlanner.create(scenario, planning_problem, automaton, plot_config, MotionPlannerType.WEIGHTED_ASTAR)


    @classmethod
    def DepthFirstSearch(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: ManeuverAutomaton,
                         plot_config=DefaultPlotConfig) -> DepthFirstSearch:
        """
        Method to instantiate a Depth-First-Search motion planner.
        """
        return MotionPlanner.create(scenario, planning_problem, automaton, plot_config, MotionPlannerType.DFS)

    @classmethod
    def LRTAstar(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: ManeuverAutomaton,
                    plot_config=DefaultPlotConfig) -> LRTAstar:
        """
        Method to instantiate an LRTA* Search motion planner.
        """
        return MotionPlanner.create(scenario, planning_problem, automaton, plot_config, MotionPlannerType.LRTASTAR)

    @classmethod
    def IDDFS(cls, scenario: Scenario, planning_problem: PlanningProblem, automaton: ManeuverAutomaton,
                    plot_config=DefaultPlotConfig) -> IDDFS:
        """
        Method to instantiate an IDDFS motion planner.
        """
        return MotionPlanner.create(scenario, planning_problem, automaton, plot_config, MotionPlannerType.IDDFS)
