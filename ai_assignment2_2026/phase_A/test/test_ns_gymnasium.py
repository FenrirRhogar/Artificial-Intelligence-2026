import gymnasium as gym
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.local_search_agents import TestingAgent, HillClimbingAgent, SimulatedAnnealingAgent
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


def main(domain, timesteps):
    horizon = 10
    env = gym.make(domain, render_mode="rgb_array", max_episode_steps= timesteps)
    name, version = domain.split("-")

    # Testing with SimulatedAnnealingAgent
    agent = SimulatedAnnealingAgent(num_actions = env.action_space.n, domain_name=name)

    
    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)

    update_function1= IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)
    if name == "CartPole":
        ##### Step 4: map parameters to update functions
        tunable_params = {"masspole":update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity":update_function1, "force": update_function1}

    ######## Step 5: set notification level and pass environment and parameters into wrapper
    env = NSClassicControlWrapper(env,tunable_params,change_notification=True)

    action = -1
    seed = 123
    
    obs, info = env.reset(seed = seed)
    rewards_per_episode = np.zeros(horizon)
    for i in range(horizon):
        states, rewards, actions, truncated, terminated = agent.find_best_route(env, seed)
        print(actions)
        rewards_per_episode[i] = np.sum(np.array(rewards))

    print(f'Mean per episode Cumulative reward: {np.mean(rewards_per_episode)}')
    env.close()


if __name__ == "__main__":
    #domain = "MountainCar-v0" #"CartPole-v1" #
    domain = "CartPole-v1"
    timesteps = 100
    main(
        domain = domain,
        timesteps = timesteps,
    )
