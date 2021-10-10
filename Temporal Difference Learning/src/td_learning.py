import gym
import numpy as np
import random
import matplotlib.pyplot as plt

from gym_gridworld import GridWorldEnv
from helpers import value_plot, policy_array_from_actions
from tqdm import trange


class TD_Agent:
    def __init__(self, env, discount_factor, learning_rate):
        self.g = discount_factor
        self.lr = learning_rate

        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])

        self.num_actions = env.action_space.n

        # V[x, y] is value for grid position x, y
        self.V = np.zeros([self.grid_height, self.grid_width], dtype=np.float32)
        # policy[x, y, z] is p of action z when in grid position x, y
        self.policy = np.ones([self.grid_height, self.grid_width, self.num_actions],
                              dtype=np.float32) / self.num_actions


        self.env = env

    def action(self, s):
        # This is quite slow, but whatever
        action = np.random.choice(np.arange(self.num_actions), p=self.policy[s[0], s[1]])

        return action

    def learn(self, n_timesteps=50000):
        s = env.reset()
        s_ = None

        for i in trange(n_timesteps, desc="TD Learning"):
            a = self.action(s)  # Apply the random policy
            s_, r, done, _ = env.step(a)  # Observe the next state and the reward

            self.update_TD(s, a, r, s_)  # Update the value function estimate using the latest transition

            s = s_
            # If episode terminated (i.e., the agent fell into a hole, discovered the goal state or max. number of steps were done)
            if done:
                s = env.reset()

    def update_TD(self, s, a, r, s_):
        self.V[s[0], s[1]] = (self.V[s[0], s[1]]
                              + self.lr *
                              (r + self.g * self.V[s_[0], s_[1]] - self.V[s[0], s[1]]))


if __name__ == "__main__":
    # Create Agent and environment
    env = GridWorldEnv()
    agentTD = TD_Agent(env, discount_factor=0.8, learning_rate=0.01)

    # Learn the Value function for 10000 steps.
    agentTD.learn(n_timesteps=10000)

    # Visualize V
    V = agentTD.V.copy()  # .reshape(4, 4)
    policy = policy_array_from_actions(agentTD)
    env_map = agentTD.env.map.copy()
    value_plot(V, policy, env_map)

