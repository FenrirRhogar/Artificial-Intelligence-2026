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

            # SYNC TIME: Pass the current time to the planning environment
            sim_env.t = getattr(env, 't', 0)

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



class TestingAgent(LocalSearchAgent):
    def __init__(self, num_actions, domain_name):
        super().__init__(num_actions, domain_name)

    def act(self, env):
        return np.random.choice(self.num_actions)

class HillClimbingAgent(LocalSearchAgent):
    name = 'Hill_Climbing'
    # TODO: To be implemented by partner


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
        Simulated Annealing act implementation using the planning reward from the environment.
        """
        t = getattr(env, 't', 0)
        action_values = {}

        # 1. Evaluate all possible actions using the environment's planning reward
        for action in range(self.num_actions):
            # The environment 'env' is already a planning copy provided by find_best_route
            reward, terminated = env.reward(action)

            # Use calculate_value to include time/truncation logic
            val = self.calculate_value(t + 1, reward, terminated, False)
            action_values[action] = val

        if not action_values:
            return -1

        # 2. Find the best action (greedy)
        max_val = max(action_values.values())
        best_actions = [a for a, v in action_values.items() if v == max_val]
        best_action = np.random.choice(best_actions)

        # 3. Choose a random action to potentially transition to
        next_action = np.random.choice(list(action_values.keys()))
        next_value = action_values[next_action]

        # 4. SA Decision Logic
        delta = next_value - max_val

        if delta >= 0:
            chosen = next_action
        else:
            temp = max(self.temperature, self.min_temp)
            # Accept a worse move with Boltzmann probability
            prob = np.exp(delta / temp)
            if np.random.random() < prob:
                chosen = next_action
            else:
                chosen = best_action

        # 5. Update temperature
        self.temperature = max(self.temperature * self.cooling_rate, self.min_temp)
        return chosen
