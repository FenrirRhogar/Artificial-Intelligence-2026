import gymnasium as gym
import numpy as np
import math


class ForwardWrapper(gym.Wrapper):
    def __getattr__(self, name):
        # fallback: forward attribute access to base env
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        # keep wrapper's own attributes local
        if name in ["env", "truncated", "highest_reward", "num_actions", "domain_name"]:
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)


class ModifyTerminalStateRewardMountainCar(ForwardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.highest_reward = 10.0
        self.domain_name = "MountainCar"

    def calculate_modified_reward(self, state, terminated):
        pos, vel = state
        
        # Heuristic: Match original logic exactly
        reward = pos + 15.0 * (vel ** 2)
        return float(reward)

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        # Override reward with our custom logic
        reward = self.calculate_modified_reward(state, terminated)
        
        # Add noise as requested by the assignment
        reward += 0.5 * 2 * (np.random.rand() - 0.5)
        return state, reward, terminated, truncated, info

    def reward(self, action):
        position, velocity = self.env.unwrapped.state
        
        # Physics update
        velocity += (action - 1) * self.env.unwrapped.force + math.cos(3 * position) * (-self.env.unwrapped.gravity)
        velocity = np.clip(velocity, -self.env.unwrapped.max_speed, self.env.unwrapped.max_speed)
        position += velocity
        position = np.clip(position, self.env.unwrapped.min_position, self.env.unwrapped.max_position)
        
        if position == self.env.unwrapped.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.env.unwrapped.goal_position and velocity >= self.env.unwrapped.goal_velocity
        )
        
        reward = self.calculate_modified_reward((position, velocity), terminated)
        return reward, terminated


class ModifyTerminalStateRewardCartPole(ForwardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.highest_reward = 100.0
        self.domain_name = "CartPole"

    def calculate_modified_reward(self, state, terminated):
        x, x_dot, theta, theta_dot = state
        
        if terminated:
            return -100.0
        
        # Penalize angle heavily, and velocity/position moderately
        reward = 10.0 - (10.0 * (theta ** 2) + 0.1 * (x ** 2) + 0.5 * (theta_dot ** 2))
        return float(reward)

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        # Override reward with our custom logic
        reward = self.calculate_modified_reward(state, terminated)
        
        # Add noise as requested by the assignment
        reward += 0.5 * 2 * (np.random.rand() - 0.5)
        return state, reward, terminated, truncated, info

    def reward(self, action):
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        force = self.env.unwrapped.force_mag if action == 1 else -self.env.unwrapped.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.env.unwrapped.polemass_length * np.square(theta_dot) * sintheta) / self.env.unwrapped.total_mass
        thetaacc = (self.env.unwrapped.gravity * sintheta - costheta * temp) / (
            self.env.unwrapped.length * (4.0 / 3.0 - self.env.unwrapped.masspole * np.square(costheta) / self.env.unwrapped.total_mass)
        )
        xacc = temp - self.env.unwrapped.polemass_length * thetaacc * costheta / self.env.unwrapped.total_mass

        if self.env.unwrapped.kinematics_integrator == "euler":
            x = x + self.env.unwrapped.tau * x_dot
            x_dot = x_dot + self.env.unwrapped.tau * xacc
            theta = theta + self.env.unwrapped.tau * theta_dot
            theta_dot = theta_dot + self.env.unwrapped.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.env.unwrapped.tau * xacc
            x = x + self.env.unwrapped.tau * x_dot
            theta_dot = theta_dot + self.env.unwrapped.tau * thetaacc
            theta = theta + self.env.unwrapped.tau * theta_dot

        terminated = bool(
            x < -self.env.unwrapped.x_threshold
            or x > self.env.unwrapped.x_threshold
            or theta < -self.env.unwrapped.theta_threshold_radians
            or theta > self.env.unwrapped.theta_threshold_radians
        )
        
        reward = self.calculate_modified_reward((x, x_dot, theta, theta_dot), terminated)
        return reward, terminated
