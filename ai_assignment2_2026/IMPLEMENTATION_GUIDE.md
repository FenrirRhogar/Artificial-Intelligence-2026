# Implementation Guide: AI-agents-in-ns-gym

This document explains **what you need to implement**, **why each part matters**, and **how the pieces in this repository fit together** for the assignment.

---

## 1) Goal of the assignment

The assignment is about **local search in non-stationary environments** using `ns-gym` on top of `gymnasium`.

You must implement and evaluate three local-search agents:

1. **Hill Climbing**
2. **Random Restart Hill Climbing**
3. **Simulated Annealing**

They are tested on two classic control environments:

- `CartPole-v1`
- `MountainCar-v0`

The environments are made **non-stationary** through `ns-gym` schedulers and update functions, and their reward logic is altered through custom wrappers.

---

## 2) What is already provided

The repository already contains a useful skeleton:

- `agents/local_search_agents.py`
  - Base class: `LocalSearchAgent`
  - Random baseline: `TetsingAgent` (typo in the name, but used as a random agent)
  - A placeholder for `HillClimbingAgent`
  - A partially implemented `SimulatedAnnealingAgent`

- `wrappers/wrappers.py`
  - `ModifyTerminalStateRewardCartPole`
  - `ModifyTerminalStateRewardMountainCar`
  - `ForwardWrapper` helper

- `test/test_ns_gymnasium.py`
  - A simple single-agent test script

- `test/run_experiments.py`
  - A script intended for the experiment plots requested in the assignment

- `algorithms/hill_climbing.py`
  - An unrelated hill-climbing example on coordinates / TSP-style optimization

---

## 3) What you still need to implement

## 3.1 `HillClimbingAgent`

### What to implement

In `agents/local_search_agents.py`, the `HillClimbingAgent` class is still incomplete.
You need to implement its `act()` method.

### Why it matters

This is one of the required algorithms in the assignment.
Without it, you cannot compare the performance of all local-search strategies.

### Expected behavior

A hill-climbing agent should:

- evaluate candidate actions
- compare them using the environment’s simulated/planning reward
- choose the action with the best estimated value
- avoid random moves unless all choices are equally bad or invalid

### Practical note

The agent already has access to helper logic in `LocalSearchAgent`:

- `project_act()`
- `calculate_value()`
- `find_best_route()`

So the missing part is mainly the local decision rule inside `act()`.

---

## 3.2 `RandomRestartHillClimbingAgent`

### What to implement

The assignment PDF requires **Random Restart Hill Climbing**, but this class does not appear in the current `agents/local_search_agents.py`.
You should add it.

### Why it matters

Plain hill climbing can get stuck in local optima.
Random restart hill climbing improves exploration by repeatedly restarting from new random states, increasing the chance of finding a better solution.

### Expected behavior

A random-restart version usually:

- runs a hill-climbing search multiple times
- starts from different random actions / initial candidates
- keeps the best overall solution across restarts

### Practical note

If your current code only uses one-step lookahead, the “restart” part may be implemented as repeated resampling of candidate actions or repeated rollouts with fresh randomness.

---

## 3.3 `SimulatedAnnealingAgent`

### What to implement or verify

A `SimulatedAnnealingAgent` class already exists in your code, but it should be checked carefully.
It must behave like **simulated annealing**, not just greedy hill climbing with noise.

### Why it matters

Simulated annealing is the third required algorithm.
Its key advantage is that it can accept worse moves early on, which helps escape local minima.

### Expected behavior

A correct simulated annealing agent should:

- evaluate candidate actions using the environment’s projected outcome
- compute a value for the current and proposed action
- sometimes accept a worse action with probability depending on temperature
- gradually reduce temperature over time

### Important note

If the code only returns the best action every time, then it is not really simulated annealing.
The temperature schedule and probabilistic acceptance rule are the defining features.

---

## 4) What the wrappers do and why they matter

File: `wrappers/wrappers.py`

These wrappers modify the behavior of the environments.
They are important because the assignment is about **non-stationary** and **altered reward** settings.

### 4.1 `ModifyTerminalStateRewardCartPole`

This wrapper:

- wraps the CartPole environment
- adds small noise to the reward returned by `step()`
- provides a custom `reward()` method for simulated/planning use
- uses CartPole physics variables such as:
  - `force_mag`
  - `masspole`
  - `length`
  - `tau`
  - `x_threshold`
  - `theta_threshold_radians`

### 4.2 `ModifyTerminalStateRewardMountainCar`

This wrapper:

- wraps the MountainCar environment
- adds small reward noise
- provides a custom `reward()` method for planning
- uses MountainCar internals such as:
  - `force`
  - `gravity`
  - `max_speed`
  - `min_position`
  - `goal_position`

### Why the wrappers matter

Local-search agents need a way to **simulate candidate actions**.
These wrappers allow the agent to inspect how an action would behave in a planning copy of the environment.

### Important implementation caution

Your agent code calls `env.get_planning_env()`.
That means wrapper forwarding must work correctly.
If a wrapper blocks that call, the agent will fail with an `AttributeError`.

So for wrappers, the important design goal is:

- forward environment attributes correctly
- preserve access to `get_planning_env()`
- avoid breaking `ns-gym` internals

---

## 5) Why `LocalSearchAgent` exists

File: `agents/local_search_agents.py`

`LocalSearchAgent` is the shared base class for all local-search agents.
It already provides the common machinery:

- storing number of actions
- remembering the domain name
- converting raw reward into a value with `calculate_value()`
- simulating a complete route in `find_best_route()`
- using `project_act()` to avoid invalid actions

### Why this is useful

It avoids repeating the same simulation logic in every agent.
Each algorithm only needs to define **how to choose the next action**.

---

## 6) What the test scripts are supposed to do

### 6.1 `test/test_ns_gymnasium.py`

This is the simplest test.
It runs one agent in one environment and prints:

- the actions selected during the run
- the mean cumulative reward

It is mainly a **sanity check** that the environment, wrappers, and planning logic are connected correctly.

### 6.2 `test/test_agents.py`

This script is intended for comparing **multiple agents** on **multiple domains**.
According to the assignment, the experiments should include:

- multiple agents
- both domains
- different max episode lengths
- plots of results

### 6.3 `test/run_experiments.py`

This file is the closest to the assignment’s experimental part.
It is designed to:

- run repeated episodes
- collect cumulative rewards
- plot mean reward vs `max_timesteps`
- plot reward progression over episodes

---

## 7) What the assignment requires experimentally

From the PDF, the required experiment plan is:

### Experiment A
For each environment and each agent:

- run **100 episodes**
- repeat for `max_timesteps = 10, 100, 500, 1000`
- in each episode, sum all rewards returned by the environment
- compute the mean of the 100 cumulative rewards
- create a graph of the results

### Experiment B
For each environment:

- set `max_timesteps = 100`
- run **1000 episodes**
- create a graph of cumulative reward per episode
- this helps show how agent behavior changes as the environment becomes non-stationary over time

### What graphs you should produce

You should end up with:

- one set of graphs for `CartPole-v1`
- one set of graphs for `MountainCar-v0`

So the report should contain **two environment-specific result sets**.

---

## 8) Recommended implementation order

If you want the project to come together cleanly, implement in this order:

1. **Fix wrappers**
   - ensure `get_planning_env()` and attribute forwarding work
   - make sure CartPole and MountainCar simulations do not crash

2. **Implement `HillClimbingAgent`**
   - simplest required agent

3. **Add `RandomRestartHillClimbingAgent`**
   - extend the hill-climbing logic with multiple restarts

4. **Review `SimulatedAnnealingAgent`**
   - make sure it is truly probabilistic and temperature-based

5. **Run `test/test_ns_gymnasium.py`**
   - verify one-agent behavior

6. **Run the experiment script**
   - generate the final plots and tables

---

## 9) Known issues in the current codebase

### 9.1 Typo in `TetsingAgent`

The baseline random agent is named `TetsingAgent` instead of `TestingAgent`.
It works, but the spelling is confusing.
If you have time, renaming it would improve readability.

### 9.2 Wrapper forwarding can be fragile

If the wrapper does not forward `get_planning_env()` or other environment internals correctly, planning-based agents break.

### 9.3 MountainCar deepcopy warning

There is a known warning in `ns_gym` for MountainCar planning copies.
It does not necessarily mean the code is wrong, but it is a reminder to be careful when using deep copies for simulation.

### 9.4 Rendering

If you use `render_mode="human"`, you should expect a visible window.
That is useful for demonstrations, but slow for long experiments.
For large experiments, use non-rendered execution.

---

## 10) What to include in your report

Your report should explain:

- what each algorithm does
- why the wrapper is needed
- how the environment changes over time
- how the agent uses planning/simulation
- what experiment settings you used
- what the graphs show
- which algorithm performs best in which setting
- why the results make sense

---

## 11) Short summary

### You need to implement:

- `HillClimbingAgent.act()`
- `RandomRestartHillClimbingAgent`
- verify/fix `SimulatedAnnealingAgent`
- make sure `wrappers.py` forwards environment behavior correctly
- run the required experiments and produce the graphs

### Why:

Because the assignment is specifically about comparing local-search methods in non-stationary `ns-gym` environments, and the code must support planning, reward shaping, and reproducible experiments.

---

If you want, I can also turn this into a **cleaner report-style Markdown** with:

- an introduction
- implementation section
- experiment section
- known limitations section
- conclusion

so you can submit it directly as project documentation.

