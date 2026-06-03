import math
import random
import numpy as np
from copy import deepcopy
import ns_gym.base as base


class MCTSNode:
    def __init__(self, state, parent=None, action=None, n_actions=2):
        self.state = np.array(state)
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.untried_actions = list(range(n_actions))
        self.terminal = False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def uct(self, c):
        if self.N == 0:
            return float('inf')
        return self.Q + c * math.sqrt(math.log(self.parent.N) / self.N)

    def fully_expanded(self):
        return len(self.untried_actions) == 0 or self.terminal


def get_state(obs):
    return np.array(obs['state']) if isinstance(obs, dict) else np.array(obs)


def restore(env, state):
    env.unwrapped.state = state.copy()
    if hasattr(env.unwrapped, 'steps_beyond_terminated'):
        env.unwrapped.steps_beyond_terminated = None


class MCTSAgent(base.Agent):
    def __init__(self, env, n_simulations=50, c=math.sqrt(2), max_depth=50, gamma=0.99):
        self.n_actions = env.action_space.n
        self.n_simulations = n_simulations
        self.c = c
        self.max_depth = max_depth
        self.gamma = gamma

    def act(self, observation, env):
        root = MCTSNode(get_state(observation), n_actions=self.n_actions)
        sim_env = deepcopy(env)

        for _ in range(self.n_simulations):
            node = root

            # Selection
            restore(sim_env, node.state)
            while node.fully_expanded() and not node.terminal:
                node = max(node.children.values(), key=lambda n: n.uct(self.c))
                restore(sim_env, node.state)

            if node.terminal:
                self._backprop(node, 0.0)
                continue

            # Expansion
            action = node.untried_actions.pop()
            restore(sim_env, node.state)
            obs, reward, terminated, truncated, _ = sim_env.step(action)
            child = MCTSNode(get_state(obs), parent=node, action=action, n_actions=self.n_actions)
            child.terminal = terminated or truncated
            node.children[action] = child

            # Rollout
            rollout_return = self._rollout(sim_env, child)
            self._backprop(child, reward + self.gamma * rollout_return)

        if not root.children:
            return env.action_space.sample()
        return max(root.children.values(), key=lambda n: n.Q).action

    def _rollout(self, sim_env, child):
        if child.terminal:
            return 0.0
        restore(sim_env, child.state)
        total, discount = 0.0, 1.0
        for _ in range(self.max_depth):
            action = self._rollout_policy(sim_env, sim_env.unwrapped.state)
            obs, r, terminated, truncated, _ = sim_env.step(action)
            total += discount * r
            discount *= self.gamma
            if terminated or truncated:
                break
        return total

    def _rollout_policy(self, sim_env, state):
        return sim_env.action_space.sample()

    def _backprop(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent


class InformedMCTSAgent(MCTSAgent):
    def __init__(self, env, guidance_agent, noise_level=0.0, **kwargs):
        super().__init__(env, **kwargs)
        self.guidance_agent = guidance_agent
        self.noise_level = noise_level

    def _rollout_policy(self, sim_env, state):
        probs = self.guidance_agent.get_guidance(state, self.noise_level).ravel()
        probs = np.clip(probs, 1e-10, None)
        probs /= probs.sum()
        return int(np.random.choice(self.n_actions, p=probs))


class RandomAgent(base.Agent):
    def __init__(self, env):
        self.env = env

    def act(self, observation, env):
        return self.env.action_space.sample()
