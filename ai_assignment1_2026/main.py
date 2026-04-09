import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    mpl.use('Qt5Agg')
except ImportError:
    mpl.use('TkAgg')
# Backend will be overridden below if SHOW_VISUAL=False

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# add current directory to python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.motion_planner import MotionPlanner
from SMP.motion_planner.plot_config import StudentScriptPlotConfig


# ── Configuration ─────────────────────────────────────────────────────────────

# Heuristics to simulate for Weighted A* and LRTA*
HEURISTICS = ["euclidean", "manhattan"]

# Set to True to show animated plots, False to run without any visualization
SHOW_VISUAL = False

# ──────────────────────────────────────────────────────────────────────────────


def save_figure(folder: str, name: str) -> None:
    """Save the current matplotlib figure to folder/name.svg."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + '.svg')
    plt.rcParams['svg.fonttype'] = 'none'
    try:
        plt.savefig(path, format='svg', bbox_inches='tight')
    except Exception as e:
        print(f'Saving {path} failed: {e}')
    plt.close('all')


def main():
    scenarios = [
        'Scenarios/scenario1.xml',
        'Scenarios/scenario2.xml',
        'Scenarios/scenario3.xml',
    ]
    file_motion_primitives = 'V_9.0_9.0_Vstep_0_SA_-0.2_0.2_SAstep_0.4_T_0.5_Model_BMW320i.xml'
    if not SHOW_VISUAL:
        plt.switch_backend('Agg')  # renders to memory, no display window
    config_plot = StudentScriptPlotConfig(DO_PLOT=True)
    automaton = ManeuverAutomaton.generate_automaton(file_motion_primitives)

    for i, path_scenario in enumerate(scenarios, start=1):
        # load scenario and planning problem
        scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

        figures_folder = f'Figures/Scenario {i}'

        # ── Iterative-Deepening DFS (no heuristic, run once per scenario) ─────
        print(f"\nScenario {i} | IDDFS")
        planner = MotionPlanner.IDDFS(scenario=scenario, planning_problem=planning_problem,
                                      automaton=automaton, plot_config=config_plot)
        planner.execute_search(time_pause=0.0001)
        save_figure(figures_folder, 'IDDFS')

        for heuristic in HEURISTICS:
            print(f"\nScenario {i} | Heuristic: {heuristic}")

            # ── Weighted A* ───────────────────────────────────────────────────
            for w in [0, 1, 1.25, 1.5, 1.75, 2, 5, 10]:
                planner = MotionPlanner.weighted_astar(scenario=scenario, planning_problem=planning_problem,
                                                       automaton=automaton, plot_config=config_plot)
                planner.w = w
                planner.heuristic_type = heuristic
                planner.execute_search(time_pause=0.0001)
                save_figure(figures_folder, f'WeightedAstar_w{w}_{heuristic}')

            # ── LRTA* ─────────────────────────────────────────────────────────
            planner = MotionPlanner.LRTAstar(scenario=scenario, planning_problem=planning_problem,
                                             automaton=automaton, plot_config=config_plot)
            planner.heuristic_type = heuristic
            planner.execute_search(time_pause=0.0001)
            save_figure(figures_folder, f'LRTAstar_{heuristic}')

    print('\nDone')


if __name__ == '__main__':
    main()
