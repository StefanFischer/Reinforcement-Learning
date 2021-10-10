import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helper import visualize_agent, episode_reward_plot


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        # TODO
        self.capacity = capacity
        self.array = []

    def put(self, obs, action, reward, next_obs, done):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        # TODO
        if len(self.array) == self.capacity:
            self.array.pop(0)

        self.array.append((obs, action, reward, next_obs, done))

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""
        # TODO
        samples = random.sample(self.array, batch_size)
        return samples

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        # TODO
        return len(self.array)


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()
        # TODO: Implement the network structure
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # TODO: Implement the forward funtion, which returns the output net(x)
        return self.model(x)


class DQN():
    """The DQN method."""

    def __init__(self, env, replay_size=10000, batch_size=32, gamma=0.99, sync_after=5, lr=0.001):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.        
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # TODO: Initialize DQN target network, load parameters from DQN network

        self.target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.target_net.load_state_dict(self.dqn_net.state_dict())

        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)

    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []

        obs = self.env.reset()
        for timestep in range(1, timesteps + 1):
            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)

            next_obs, reward, done, _ = env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_rewards.append(reward)

            if done:
                obs = env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_msbe_loss()

                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            # TODO: Sync the target network
            if timestep % self.sync_after == 0:
                self.target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep == timesteps:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
            if timestep % 2000 == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)

    def predict(self, state, epsilon=0.0):
        """Predict the best action based on state. With probability epsilon take random action
        
        Returns
        -------
        int
            The action to be taken.
        """

        # TODO: Implement epsilon-greedy action selection
        randNumber = random.random()

        if randNumber <= epsilon:
            # select random action with probability epsilon
            action = random.choice(range(self.env.action_space.n))
        else:
            # use network to get next action

            # wrap state into torch tensor
            state_tensor = torch.Tensor(state)

            # bring into batch format
            state_tensor = torch.reshape(state_tensor, (1, self.obs_dim))
            output_tensor = self.dqn_net.forward(state_tensor)
            max_idx = torch.argmax(output_tensor)
            action = max_idx.item()
        return action

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.
        
        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """
        # TODO: Implement MSBE calculation

        batch = self.replay_buffer.get(self.batch_size)

        mean = 0
        for obs, action, reward, next_obs, done in batch:
            # wrap state into torch tensor
            state_tensor = torch.Tensor(obs)
            next_state_tensor = torch.Tensor(next_obs)

            # bring into batch format
            state_tensor = torch.reshape(state_tensor, (1, self.obs_dim))
            next_state_tensor = torch.reshape(next_state_tensor, (1, self.obs_dim))
            Q = self.dqn_net.forward(state_tensor)
            Q = Q[0, action]
            Q_ = self.target_net.forward(next_state_tensor)
            if done:
                Q_ = torch.zeros(1)
            mean += (Q - (reward + self.gamma * (1 - done) * torch.max(Q_))) ** 2
        mean = mean / len(batch)
        return mean


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
    """Linearily decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
    # TODO: Implement epsilon decay function
    if timestep == 0:
        return epsilon_start
    elif timestep >= frames_decay:
        return epsilon_final

    epsilon_per_timestep = ((epsilon_final - epsilon_start) / frames_decay)
    epsilon = epsilon_start + epsilon_per_timestep * timestep
    return epsilon


if __name__ == '__main__':
    # Create gym environment
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    # Plot epsilon rate over time
    plt.plot([epsilon_by_timestep(i, epsilon_start=1.0, epsilon_final=0.02, frames_decay=10000) for i in range(50000)])
    plt.title("Epsilon Decay")
    plt.show()

    # Train the DQN agent
    dqn = DQN(env)
    dqn.learn(30000)

    # Visualize the agent
    visualize_agent(env, dqn)
