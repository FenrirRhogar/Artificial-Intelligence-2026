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

    Re-plans from scratch each act() call. Rollout runs to natural episode end
    (no depth cap, no discount). Node values are min-max normalized to [0,1]
    so the fixed exploration constant c=sqrt(2) is well scaled.
    """
    def __init__(self, env, num_simulations=100, c=math.sqrt(2)):
        self.env = env
        self.num_simulations = num_simulations
        self.c = c

    # ----- public API ------------------------------------------------------
    def act(self, observation, env):
        root = _Node(deepcopy(env), obs=observation)
        self._vmin = math.inf
        self._vmax = -math.inf

        for _ in range(self.num_simulations):
            leaf = self._select(root)
            child = self._expand(leaf)
            g = self._rollout(child)
            self._backprop(child, g)

        self.last_root = root   # kept for render_tree()
        # robust choice: most visited child
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]

    # ----- debug visualization ---------------------------------------------
    def render_tree(self, max_depth=2):
        """Print the last search tree as ASCII. action [N=.. Q=.. P=..]."""
        root = getattr(self, 'last_root', None)
        if root is None:
            print('(no tree yet — call act() first)')
            return

        def walk(node, depth, prefix):
            if depth > max_depth:
                return
            for a, ch in sorted(node.children.items(),
                                key=lambda kv: kv[1].N, reverse=True):
                q = ch.W / ch.N if ch.N else 0.0
                p = node.priors[a] if node.priors is not None else None
                ptxt = f' P={p:.2f}' if p is not None else ''
                term = ' [terminal]' if ch.done else ''
                print(f'{prefix}a={a} [N={ch.N} Q={q:.2f}{ptxt}]{term}')
                walk(ch, depth + 1, prefix + '   ')

        print(f'ROOT [N={root.N}]')
        walk(root, 1, '   ')

    def plot_tree(self, path=None, ax=None, max_depth=2, title=None):
        """Plot the last search tree with matplotlib.

        Layered layout: depth on y, tidy x. Node label = visits N and mean Q
        (and prior P for the informed agent). Saves to `path` if given.
        Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt

        root = getattr(self, 'last_root', None)
        if root is None:
            raise RuntimeError('no tree yet — call act() first')

        # assign positions: x by leaf order, y by -depth
        pos = {}
        self._leaf_x = 0

        def layout(node, depth):
            kids = [ch for _, ch in sorted(node.children.items())]
            if depth >= max_depth or not kids:
                x = self._leaf_x
                self._leaf_x += 1
            else:
                xs = [layout(ch, depth + 1) for ch in kids]
                x = sum(xs) / len(xs)
            pos[id(node)] = (x, -depth)
            return x

        layout(root, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, self._leaf_x * 0.9), 4))

        def draw(node, depth, parent_action=None):
            x, y = pos[id(node)]
            if node.parent is not None:
                px, py = pos[id(node.parent)]
                ax.plot([px, x], [py, y], color='gray', lw=0.8, zorder=1)
            q = node.W / node.N if node.N else 0.0
            if node.parent is None:
                lbl = f'root\nN={node.N}'
            else:
                p = node.parent.priors[node.action_in] if node.parent.priors is not None else None
                lbl = f'a={node.action_in}\nN={node.N}\nQ={q:.1f}'
                if p is not None:
                    lbl += f'\nP={p:.2f}'
            color = '#d9534f' if node.done else '#5bc0de'
            ax.scatter([x], [y], s=600, color=color, zorder=2, edgecolors='k')
            ax.annotate(lbl, (x, y), ha='center', va='center', fontsize=6, zorder=3)
            if depth < max_depth:
                for _, ch in sorted(node.children.items()):
                    draw(ch, depth + 1)

        draw(root, 0)
        ax.set_title(title or f'MCTS tree (sims={self.num_simulations})')
        ax.axis('off')
        if path is not None:
            ax.figure.savefig(path, dpi=150, bbox_inches='tight')
        return ax

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
        q = self._normalize(child.W / child.N)
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

'''
class InformedMCTSAgent(MCTSAgent):
    """MCTS with AlphaGo-style PUCT selection.

    Identical to MCTSAgent except the node-selection criterion combines the
    normalized value with a DQN-mentor prior P(s,a) (informed UCT).
    """
    def __init__(self, env, guidance_agent, noise_level=0.0,
                 num_simulations=100, c=math.sqrt(2)):
        super().__init__(env, num_simulations=num_simulations, c=c)
        self.guidance_agent = guidance_agent
        self.noise_level = noise_level

    def _priors(self, node):
        return self.guidance_agent.get_guidance(node.obs, noise_level=self.noise_level)

    def _score(self, parent, child):
        q = self._normalize(child.W / child.N)
        p = parent.priors[child.action_in] if parent.priors is not None else 1.0
        return q + self.c * p * math.sqrt(parent.N) / (1 + child.N)
'''