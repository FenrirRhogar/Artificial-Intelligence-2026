import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg

import ns_gym.base as base
import random
import math


class RandomAgent(base.Agent):
    """A random agent that samples actions uniformly at random from the action space."""
    def __init__(self, env):
        self.env = env

    def act(self, observation, env):
        return self.env.action_space.sample()


class _Node:
    """Single MCTS tree node. Holds a deepcopy of the env at this state."""
    def __init__(self, env, obs, parent=None, action_in=None, reward=0.0, done=False):
        self.env = env              # env snapshot at this node (source for deepcopy)
        self.obs = obs              # observation at this node (for guidance queries)
        self.parent = parent
        self.action_in = action_in  # action taken from parent to reach here
        self.reward = reward        # step reward earned entering this node
        self.done = done            # terminal (done or truncated)
        self.children = {}          # action -> _Node
        self.untried = [] if done else list(range(env.action_space.n))
        self.priors = None          # cached guidance probs (informed agent only)
        self.N = 0                  # visit count
        self.W = 0.0                # sum of returns

    def fully_expanded(self):
        return not self.untried


class MCTSAgent(base.Agent):
    """Standard Monte Carlo Tree Search with UCT.

    Reuses the search tree across act() calls: after an action is chosen, the
    corresponding child becomes the new root and its entire subtree (with all
    accumulated visit counts and returns) is retained; the old root and its
    other branches are discarded. The re-rooted node is re-grounded to the
    current real env so simulations plan from the live state/dynamics.
    Rollout runs to natural episode end (no depth cap, no discount). Node
    values are min-max normalized to [0,1] so the fixed exploration constant
    c=sqrt(2) is well scaled.
    """
    def __init__(self, env, num_simulations=100, c=math.sqrt(2)):
        self.env = env
        self.num_simulations = num_simulations
        self.c = c
        self.root = None            # persisted tree across act() calls
        self.last_action = None     # action chosen on the previous act()

    # ----- public API ------------------------------------------------------
    def act(self, observation, env):
        root = self._advance_root(observation, env)
        self.root = root
        self._vmin = math.inf
        self._vmax = -math.inf

        for _ in range(self.num_simulations):
            leaf = self._select(root)
            child = self._expand(leaf)
            g = self._rollout(child)
            self._backprop(child, g)

        # robust choice: most visited child
        self.last_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return self.last_action

    def _advance_root(self, observation, env):
        """Reuse the subtree under the previously chosen action as the new root.

        Falls back to a fresh root when there is no prior tree or the chosen
        action was never expanded. The retained node keeps its statistics and
        children but is re-grounded to the live env so new expansions/rollouts
        use the current state and (non-stationary) dynamics.
        """
        if self.root is not None and self.last_action is not None:
            child = self.root.children.get(self.last_action)
            if child is not None:
                child.parent = None          # detach old root + sibling branches
                child.action_in = None
                child.reward = 0.0           # no parent return to shift into
                child.env = deepcopy(env)     # ground to real current state
                child.obs = observation
                child.priors = None          # recompute under current state
                return child
        return _Node(deepcopy(env), obs=observation)

    # ----- MCTS phases -----------------------------------------------------
    def _select(self, node):
        while not node.done and node.fully_expanded():
            node = max(node.children.values(),
                       key=lambda ch: self._score(node, ch))
        return node

    def _expand(self, node):
        if node.done or node.fully_expanded():
            return node
        if node.priors is None:
            node.priors = self._priors(node)
        a = node.untried.pop()
        env2 = deepcopy(node.env)
        obs, r, done, trunc, _ = env2.step(a)
        child = _Node(env2, obs=obs, parent=node, action_in=a,
                      reward=float(r), done=bool(done or trunc))
        node.children[a] = child
        return child

    def _rollout(self, node):
        if node.done:
            return 0.0
        env = deepcopy(node.env)
        total = 0.0
        done = trunc = False
        while not (done or trunc):
            a = env.action_space.sample()
            _, r, done, trunc, _ = env.step(a)
            total += float(r)
        return total

    def _backprop(self, node, rollout_return):
        g = rollout_return
        while node is not None:
            node.N += 1
            node.W += g
            self._track(node.W / node.N)
            g += node.reward          # shift to parent's return perspective
            node = node.parent

    # ----- selection score / normalization ---------------------------------
    def _score(self, parent, child):
        # textbook UCB1:  X̄_i + c * sqrt(ln N / n_i)   (raw average reward)
        q = child.W / child.N
        return q + self.c * math.sqrt(math.log(parent.N) / child.N)

    def _track(self, q):
        self._vmin = min(self._vmin, q)
        self._vmax = max(self._vmax, q)

    def _normalize(self, q):
        if self._vmax > self._vmin:
            return (q - self._vmin) / (self._vmax - self._vmin)
        return 0.0

    # ----- hook for informed variant ---------------------------------------
    def _priors(self, node):
        return None

class InformedMCTSAgent(MCTSAgent):
    """MCTS with AlphaGo-style PUCT selection.

    Identical to MCTSAgent except the node-selection criterion combines the
    normalized value with a DQN-mentor prior P(s,a) (informed UCT).
    """
    def __init__(self, env, guidance_agent, noise_level=0.0,
                 num_simulations=100, c=1):
        super().__init__(env, num_simulations=num_simulations, c=c)
        self.guidance_agent = guidance_agent
        self.noise_level = noise_level

    def _priors(self, node):
        return self.guidance_agent.get_guidance(node.obs, noise_level=self.noise_level)

    def _rollout(self, node, max_depth=200, eps=0.1):

        if node.done:
            return 0.0
        env = deepcopy(node.env)
        obs = node.obs
        total = 0.0
        done = trunc = False
        steps = 0
        while not (done or trunc) and steps < max_depth:
            a = self.guidance_agent.act(obs, eps=eps)
            obs, r, done, trunc, _ = env.step(a)
            total += float(r)
            steps += 1
        return total

    def _score(self, parent, child):
        q = self._normalize(child.W / child.N)
        p = parent.priors[child.action_in] if parent.priors is not None else 1.0
        return q + self.c * p * math.sqrt(parent.N) / (1 + child.N)