import numpy as np
import random
from helper import action_value_plot, test_agent

from gym_gridworld import GridWorldEnv

class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        # TODO: define a Q-function member variable self.Q
        # Remark: Use this kind of Q-value member variable for visualization tools to work, i.e. of shape [grid_height, grid_width, num_actions]
        # Q[y, x, z] is value of action z for grid position y, x
        self.Q = np.zeros([4, 4, 4], dtype=np.float32)

    def action(self, s, epsilon=0.0):
        # TODO: implement epsilon-greedy action selection
        randNumber = random.random()
        action = 0
        if randNumber <= epsilon:
            # choose random action
            action = np.random.choice(self.env.action_space.n)
            #print("action random : " + str(action))
        else:
            # choose best action according to Q
            action = np.argmax(self.Q[s[0], s[1]])
            #print("action max : " + str(action))
        return int(action)

class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        # TODO: implement training loop
        s = env.reset()
        a = self.action(self.env.agent_position, self.eps)

        for i in range(n_timesteps):
            s_, r, done, _ = env.step(a)  # Observe the next state and the reward
            a_ = self.action(s_, self.eps)
            self.update_Q(s, a, r, s_, a_)  # Update the value function estimate using the latest transition

            s = s_
            a = a_
            # If episode terminated (i.e., the agent fell into a hole, discovered the goal state or max. number of steps were done)
            if done:
                s = env.reset()

    def update_Q(self, s, a, r, s_, a_):
        # TODO: implement Q-value update rule

        #new_Q = self.Q[s[0], s[1], a] + self.lr * (r + self.g * self.Q[s_[0], s_[1], a_] - self.Q[s[0], s[1], a])
        #diff = new_Q - self.Q[s[0], s[1], a]
        #print("diff : " + str(diff))
        #self.Q[s[0], s[1], a] = new_Q
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (r + self.g * self.Q[s_[0], s_[1], a_] - self.Q[s[0], s[1], a])


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        # TODO: implement training loop
        s = env.reset()

        for i in range(n_timesteps):
            a = self.action(s, self.eps)
            s_, r, done, _ = env.step(a)
            self.update_Q(s, a, r, s_)
            s = s_
            if done:
                s = env.reset()


    def update_Q(self, s, a, r, s_):
        # TODO: implement Q-value update rule
        max_action_ind = np.argmax(self.Q[s_[0], s_[1]])
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (r + self.g * np.max(self.Q[s_[0], s_[1]]) - self.Q[s[0], s[1], a])


if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv("cliffwalking")

    discount_factor = 0.9
    learning_rate = 0.1
    epsilon = 0.4
    n_timesteps = 200000

    #Train SARSA agent
    sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    sarsa_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(sarsa_agent)
    # Uncomment to do a test run
    print("Testing SARSA agent...")
    test_agent(sarsa_agent, env, epsilon)

    # Train Q-Learning agent
    # qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    # qlearning_agent.learn(n_timesteps=n_timesteps)
    # action_value_plot(qlearning_agent)
    # # Uncomment to do a test run
    # print("Testing Q-Learning agent...")
    # test_agent(qlearning_agent, env, 0.0)
