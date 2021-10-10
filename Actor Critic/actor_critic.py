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
    ret = next_value
    for reward in reversed(rewards):
        ret = reward + discount * ret
        return_lst.append(ret)
    return return_lst[::-1]

def compute_advantages(returns, values):
    """ Compute episode advantages based on precomputed episode returns.

    Parameters
    ----------
    returns : list of float
        Episode returns calculated with compute_returns.
    values: list of float
        Critic outputs for the states visited during the episode

    Returns
    -------
    list of float
        Episode advantages.
    """
    # TODO
    A = [returns[i] - values[i] for i in range(len(returns))]
    return A

def compute_generalized_advantages(rewards, values, next_value, discount, lamb):
    """ Compute generalized advantages (GAE) of the episode.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    values: list of float
        Episode state values.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, critic output otherwise.
    discount : float
        Discount factor.
    lamb: float
        Lambda parameter of GAE.

    Returns
    -------
    list of float
        Generalized advanteges of the episode.
    """
    gae = 0
    advantages = []
    for reward, value in zip(reversed(rewards), reversed(values)):
        td_error = reward + discount * next_value - value.item()
        gae = td_error + discount * lamb * gae
        advantages.append(gae)
        next_value = value.item()
    return advantages[::-1]


class TransitionMemory():
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma, lamb, use_gae):
        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logp_lst = []
        self.return_lst = []
        self.adv_lst = []

        self.gamma = gamma
        self.lamb = lamb
        self.use_gae = use_gae

        self.traj_start = 0

    def put(self, obs, action, reward, logp, value):
        """Put a transition into the memory."""
        self.obs_lst.append(obs)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.value_lst.append(value)
        self.logp_lst.append(logp)

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        assert len(self.adv_lst) == len(self.obs_lst) == len(self.return_lst)
        return self.obs_lst, self.reward_lst, self.logp_lst, self.value_lst, self.return_lst, self.adv_lst

    def clear(self):
        """Reset the transition memory."""
        self.obs_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.logp_lst = []
        self.return_lst = []
        self.adv_lst = []

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

        if self.use_gae:
            traj_adv = compute_generalized_advantages(self.reward_lst[self.traj_start:], self.value_lst[self.traj_start:], next_value, self.gamma, self.lamb)
        else:
            traj_adv = compute_advantages(traj_returns, self.value_lst[self.traj_start:])
        self.adv_lst.extend(traj_adv)

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


class CriticNetwork(nn.Module):
    """Neural Network used to learn the state-value function."""

    def __init__(self, num_observations):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            # Q-value as output
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCritic():
    """The Actor-Critic approach."""

    def __init__(self, env, batch_size=200, gamma=0.99, lamb=0.99, lr=0.005, use_gae=False):
        """ Constructor.
        
        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        batch_size : int, optional
            Number of transitions to use for one opimization step.
        gamma : float, optional
            Discount factor.
        lamb : float, optional
            Lambda parameters of GAE.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.batch_size = batch_size
        self.env = env
        self.memory = TransitionMemory(gamma, lamb, use_gae)

        # TODO: Create networks and optimizers
        self.critc_net = CriticNetwork(self.obs_dim)
        self.optim_critic = optim.Adam(self.critc_net.parameters(), lr=lr)

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the actor-critic.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs = env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        batch_start = 0
        for timestep in range(1, total_timesteps + 1):
            # Do one step
            action, logp, value = self.predict(obs, train_returns=True)
            next_obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)

            # Put into transition buffer
            self.memory.put(obs, action, reward, logp, value)

            # Update current obs
            obs = next_obs

            if done:
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []
                # TODO: get new observation, finish trajectory

                obs = self.env.reset()

                self.memory.finish_trajectory(0.0)
            
            # Check if batch is full
            batch_finished = (timestep - batch_start) == self.batch_size

            if batch_finished:
                # TODO: Finish partial trajectory
                if not done:
                    self.memory.finish_trajectory(value.item())

                # Get transitions from memory
                obs_lst, reward_lst, logp_lst, value_lst, return_lst, adv_lst = self.memory.get()

                # TODO: Calculate losses
                actor_loss = self.calc_actor_loss(logp_lst, adv_lst)
                critic_loss = self.calc_critic_loss(value_lst, return_lst)

                # TODO: Backpropagate and optimize
                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step()

                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                # Clear memory
                self.memory.clear()
                batch_start = timestep

            # Episode reward plot
            if timestep % 100000 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)
            if timestep == total_timesteps:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    def calc_critic_loss(self, value_lst, return_lst):
        """Calculate critic loss for one batch of transitions."""
        # TODO
        value_tensor = torch.stack(value_lst)
        return_tensor = torch.Tensor(return_lst)
        loss = -(value_tensor - return_tensor).mean()
        return loss


    def calc_actor_loss(self, logp_lst, adv_lst):
        """Calculate actor "loss" for one batch of transitions."""
        # TODO
        logp_tensor = torch.stack(logp_lst)
        adv_tensor = torch.Tensor(adv_lst)

        loss = -(logp_tensor * adv_tensor).mean()
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

        # TODO: Calculate value based on obs
        value = self.critc_net(obs)
    
        if train_returns:
            return action.item(), logp, value
        else:
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v0"
    use_gae = True

    env = gym.make(env_id)
    AC = ActorCritic(env, batch_size=200, gamma=0.99, lamb=0.99, lr=0.005, use_gae=use_gae)
    AC.learn(1000000)

    # Visualize the agent
    visualize_agent(env, AC)