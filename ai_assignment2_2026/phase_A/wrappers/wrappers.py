import gymnasium as gym
import numpy as np
import math


class ForwardWrapper(gym.Wrapper):
    def __getattr__(self, name):
        # fallback: forward attribute access to base env
        return getattr(self.env, name)


    
    def __setattr__(self, name, value):
        # keep wrapper's own attributes local
        if name in ["env", "truncated", "highest_reward"]:
            super().__setattr__(name, value)
        else:
            setattr(self.env.unwrapped, name, value)


class ModifyTerminalStateRewardMountainCar(gym.Wrapper):
    #TODO
    def __init__(self, env):
        super().__init__(env)
        #self.truncated = False
        #self.highest_reward = 10_000


    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        reward += 0.5 * 2* (np.random.rand() - 0.5) # add some noise to the reward to make it more interesting 
        #if truncated:
        #    self.truncated = True
        #    reward = self.highest_reward

        return state, reward, terminated, truncated, info

    def reward(self, action):
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        return -1.0 if not terminated else -10




class ModifyTerminalStateRewardCartPole(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        #self.truncated = False
        #self.highest_reward = 10_000
        #self.kinematics_integrator = ""



    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        reward += 0.5 * 2* (np.random.rand() - 0.5) # add some noise to the reward to make it more interesting 
        """ if truncated:
            self.truncated = True
            reward = self.highest_reward """

        return state, reward, terminated, truncated, info

    def reward(self, action):
        if self.truncated and action == -1:
            return self.highest_reward
            
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        '''if terminated:
            breakpoint()
        reward = - (np.abs(x) - self.x_threshold) - (np.abs(theta) -self.theta_threshold_radians)'''
        reward = - max(abs(x) / self.x_threshold, abs(theta) / self.theta_threshold_radians)
        return -reward if not terminated else 0
    