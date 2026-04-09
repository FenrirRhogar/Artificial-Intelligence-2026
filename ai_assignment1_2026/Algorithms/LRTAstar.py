from Algorithms.Utils.SequentialSearch import SequentialSearch
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.node import CostNode

class LRTAstar(SequentialSearch):
    """
    Class for LRTA* (Learning Real-Time A*) Search algorithm.

    LRTA* is an online, real-time search algorithm that:
    - Moves the agent one step at a time (no full lookahead)
    - Maintains a learned heuristic table H that gets updated as the agent moves
    - At each state, looks ahead at neighbors and picks the move minimizing cost + H
    - Updates H[current] = min over neighbors of (cost_to_neighbor + H[neighbor])
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        self.heuristic_type = "weighted_euclidean"  # options: "euclidean", "manhattan", "weighted_euclidean"
        self.H = {}
       # self.visited = {}  # state_key -> min cost seen

    def _state_key(self, node):
        state = node.list_paths[-1][-1]
        return (
            round(state.position[0], 1),
            round(state.position[1], 1)
        )

    def _get_H(self, node):
        """Return the learned heuristic for a node, initializing from base heuristic if unseen."""
        key = self._state_key(node)
        if key not in self.H:
            self.H[key] = self.heuristic_function(node)
        return self.H[key]

    def execute_search(self, time_pause):
        node_current = self.initialize_search(time_pause=time_pause, cost=True)
        initial_primitive = node_current.list_primitives[-1]
        initial_state = node_current.list_paths[-1][-1]
        visited_nodes_count = 0
        path_taken = [initial_state]

        while True:
            current_key = self._state_key(node_current)
            visited_nodes_count += 1
            successors_info = []

            for primitive_successor in node_current.get_successors():
                collision_flag, child = self.take_step(
                    successor=primitive_successor,
                    node_current=node_current,
                    cost=True
                )
                if collision_flag:
                    continue

                if self.goal_reached(successor=primitive_successor, node_current=node_current):
                    path_taken.append(child.list_paths[-1][-1])
                    path_str = "->".join([f"({s.position[0]:.2f},{s.position[1]:.2f})" for s in path_taken])
                    print(f"LRTA* Search:")
                    print(f"    Visited Nodes number: {visited_nodes_count}")
                    print(f"    Path: {path_str}")
                    print(f"    Estimated Cost: {child.cost:.4f}")
                    return True

                step_cost = child.cost - node_current.cost 
                f_child = step_cost + self._get_H(child)
                successors_info.append((child, primitive_successor, f_child))

            if not successors_info:
                self.H[current_key] = self._get_H(node_current) + 50
                node_current = CostNode(
                    list_paths=[[initial_state]],
                    list_primitives=[initial_primitive],
                    depth_tree=0,
                    cost=0
                )
                path_taken = [initial_state]  # reset path on restart
                continue

            min_f = min(info[2] for info in successors_info)
            self.H[current_key] = max(self._get_H(node_current), min_f)

            best_child, _, _ = min(successors_info, key=lambda x: x[2])
            last_state = best_child.list_paths[-1][-1]
            path_taken.append(last_state)  # ← correctly placed AFTER best_child is defined

            node_current = CostNode(
                list_paths=best_child.list_paths,
                list_primitives=best_child.list_primitives,
                depth_tree=best_child.depth_tree,  # ← not depth_t
                cost=best_child.cost
            )

    def heuristic_function(self, node_current):
        """
        Heuristic function h(x): distance from node_current to goal.
        Returns distance normalized to be comparable with cost function measurements.
        """
        if self.heuristic_type == "euclidean":
            return self.calc_euclidean_distance(node_current)

        elif self.heuristic_type == "manhattan":
            pos_current = node_current.list_paths[-1][-1].position
            goal_info = self.get_goal_information()  # [x, y, length, width]
            pos_goal = [goal_info[0], goal_info[1]]
            distance = abs(pos_current[0] - pos_goal[0]) + abs(pos_current[1] - pos_goal[1])
            return distance
        
        return 0