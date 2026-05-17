from copy import deepcopy
import numpy as np
from copy import deepcopy
from wrappers.wrappers import ModifyTerminalStateRewardCartPole, ModifyTerminalStateRewardMountainCar
import time

class LocalSearchAgent:
    '''
    This is an 'abstract' implementation of a Local Search Agent.
    Every Local search agent e.g., HiLLClimbing .. should inherit 
    and implement the functionalities of this exact agent.
    '''
    name = 'Local_Search_Agent'

    def __init__(self, num_actions, domain_name):
        self.num_actions = num_actions
        self.domain_name = domain_name
        self.current_value = -np.inf
        pass

    def act(self, env) -> int:
        '''
        The method responsible for generating an action based on the current
        observation/state of the environment. It either returns a valid action
        or -1 en the case that all the actions results to a terminal state
        '''


        raise NotImplementedError()

    def project_act(self, env) -> int:
        """ 
        This method caalls the act function in rder to return an action.
        If this action is not valid (i.e., -1) then it returns a
        random action since it make no difference (all action evaluated within act() 
        result to a terminal state) 
        """

        action = self.act(env)
        return action if action != -1 else np.random.choice(self.num_actions)
    
    
    def calculate_value(self, t, reward, terminated, truncated):
        if self.domain_name == 'CartPole':
            #In cart pole, we would like the agent to survive as long as it can
            return reward + 0.1 * truncated + t * (not terminated)
        #however, in Mountain Car we would like to reach the goal
        #the faster that it can
        return reward + 0.1 * truncated - t * (not terminated)

    def find_best_route(self, env, seed):
        '''
        method that simulates a whole run.
        it returns the initial state, 
        the path of the actions taken by the agent (when removing the no op action)
        and if the episode terminated succesfully (truncated) or terminated  
        '''
        visulalize = True
        terminated = False
        truncated = False
        action = -1
        states = []
        rewards = []
        actions = []
        values = []
        #in that way, the initial point is always the same
        obs, info = env.reset(seed = seed)
        
        while not (terminated or truncated):
            
            
            """ while hasattr(sim_env, "env"):
                sim_env = sim_env.env """
            #sim_env = sim_env.unwrapped
            if self.domain_name == "CartPole":
                sim_env = ModifyTerminalStateRewardCartPole(deepcopy(env.get_planning_env()))
            elif self.domain_name == "MountainCar":
                sim_env = ModifyTerminalStateRewardMountainCar(deepcopy(env.get_planning_env()))

            action = self.project_act(sim_env)
            state, reward, terminated, truncated, info = env.step(action)
            value = self.calculate_value(
                t = env.t,
                reward = reward,
                terminated = terminated,
                truncated = truncated
            )

            env.render()
            values.append(value)
            self.current_value = value
            states.append(state); rewards.append(reward); actions.append(action)

        return states, rewards, actions, terminated, truncated



class TetsingAgent(LocalSearchAgent):
    def __init__(self, num_actions, domain_name):
        super().__init__(num_actions, domain_name)

    def act(self, env):
        return np.random.choice(self.num_actions)

class HillClimbingAgent(LocalSearchAgent):
    name = 'Hill_Climbing'
    # TODO


class SimulatedAnnealingAgent(LocalSearchAgent):
    name = 'Simulated_Annealing'

    def __init__(self, num_actions, domain_name,
                 initial_temp=0.01, cooling_rate=0.99, min_temp=1e-3):
        super().__init__(num_actions, domain_name)
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def act(self, env) -> int:
        """
        Simulated Annealing act implementation using a robust 1-step deepcopy with strong heuristics.
        """
        t = getattr(env, 't', 0)

        action_values = {}
        for action in range(self.num_actions):
            # We must use deepcopy because we need the physics engine to update the state correctly.
            # The environment is non-stationary, so we must sync parameters.
            sim = deepcopy(env)
            self._sync_attrs(env, sim)
            
            obs, reward, terminated, truncated, _ = sim.step(action)
            
            # Extract state
            state = obs['state'] if isinstance(obs, dict) else obs
            
            # Use our custom strong heuristic for CartPole
            if self.domain_name == 'CartPole':
                x, x_dot, theta, theta_dot = state
                if terminated:
                    heuristic_reward = -100.0
                else:
                    # Penalize angle heavily, and velocity moderately
                    heuristic_reward = 10.0 - (10.0 * (theta ** 2) + 0.1 * (x ** 2) + 0.5 * (theta_dot ** 2))
            elif self.domain_name == 'MountainCar':
                pos, vel = state
                heuristic_reward = pos + 15.0 * (vel ** 2)
            else:
                heuristic_reward = reward

            val = self.calculate_value(t + 1, heuristic_reward, terminated, truncated)
            action_values[action] = val

        if not action_values:
            return -1

        # 3. SA Decision Logic
        max_val = max(action_values.values())
        best_actions = [a for a, v in action_values.items() if v == max_val]
        best_action = np.random.choice(best_actions)

        # For SA, compare best against a random neighbor
        next_action = np.random.choice(list(action_values.keys()))
        next_value = action_values[next_action]

        delta = next_value - max_val

        if delta >= 0:
            chosen = next_action
        else:
            temp = max(self.temperature, self.min_temp)
            prob = np.exp(delta / temp)
            if np.random.random() < prob:
                chosen = next_action
            else:
                chosen = best_action

        self.temperature = max(self.temperature * self.cooling_rate, self.min_temp)
        return chosen

    def _sync_attrs(self, source_env, target_env):
        """Syncs ns-gym non-stationary parameters from source to target environment."""
        source = source_env.unwrapped
        target = target_env.unwrapped
        
        # Common attributes across domains
        attrs = ['gravity', 'tau', 'force_mag', 'masspole', 'masscart', 'length', 
                 'power', 'speed', 'min_position', 'max_position', 'goal_position']
        
        for attr in attrs:
            if hasattr(source, attr):
                try:
                    setattr(target, attr, getattr(source, attr))
                except AttributeError:
                    pass