import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

from utils import episode_reward_plot, visualize_agent


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """

    return_lst = []
    ret = 0.0
    for reward in reversed(rewards):
        ret = reward + discount * ret
        return_lst.append(ret)
    return return_lst[::-1]


class TransitionMemory():
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):
        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.logp_lst = []
        self.return_lst = []

        self.gamma = gamma
        self.traj_start = 0

    def put(self, obs, action, reward, logp):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.logp_lst.append(logp)

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        assert len(self.obs_lst) == len(self.return_lst)
        return self.obs_lst, self.reward_lst, self.logp_lst, self.return_lst

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.logp_lst = []
        self.return_lst = []

        self.traj_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return and advantage or generalized advantage estimation.

        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
        """
        traj_returns = compute_returns(self.reward_lst[self.traj_start:], next_value, self.gamma)
        self.return_lst.extend(traj_returns)

        self.traj_start = len(self.obs_lst)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.net(obs)


class VPG():
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.

        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continous actions not implemented!')

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs = env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        num_episodes = 0

        for timestep in range(1, total_timesteps + 1):
            # Do one step
            action, logp = self.predict(obs, train_returns=True)
            next_obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)

            # Put into transition buffer
            self.memory.put(obs, action, reward, logp)

            # Update current obs
            obs = next_obs

            if done:
                obs = self.env.reset()

                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

                self.memory.finish_trajectory(0.0)

                num_episodes += 1

                if num_episodes == self.episodes_update:
                    # Get transitions inside memory
                    obs_lst, reward_lst, logp_lst, return_lst = self.memory.get()

                    # Calculate loss
                    loss = self.calc_actor_loss(logp_lst, return_lst)

                    # Backpropagate and optimize
                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

                    # Clear memory
                    self.memory.clear()
                    num_episodes = 0

            # Episode reward plot
            if timestep % 100000 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)
            if timestep == total_timesteps:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    def calc_actor_loss(self, logp_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""

        logp_tensor = torch.stack(logp_lst)
        return_tensor = torch.Tensor(return_lst)

        loss = -(logp_tensor * return_tensor).mean()
        return loss

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.

        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        obs = torch.Tensor(obs)

        # Actor forward
        logits = self.actor_net(obs)
        action_distribution = Categorical(logits=logits)
        action = action_distribution.sample()
        logp = action_distribution.log_prob(action)

        if train_returns:
            return action.item(), logp
        else:
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    vpg = VPG(env, episodes_update=20, gamma=0.99, lr=0.001)
    vpg.learn(1000000)

    # Visualize the agent
    visualize_agent(env, vpg)