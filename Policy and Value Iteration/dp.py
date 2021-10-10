import numpy as np
from helper import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    # TODO: Write your implementation here
    """
        P : dict
        P captures the state transition probabilities and the reward function. For every state s and every possible action a, 
        P[s][a] contains a list of tuples (p, s', r, is_terminal) with:
        - p: the probability of s' being the next state given s, a
        - s': the next state
        - r: the reward gained from this event
        - is_terminal: if s' is a terminal state
    """
    for s_idx in range(mdp.num_states):
        sumOverActions = 0
        for a_idx in range(mdp.num_actions):
            sumOverNextStates = 0
            for next_state_tuple in mdp.P[s_idx][a_idx]:
                prob_for_next_state = next_state_tuple[0]
                next_state = next_state_tuple[1]
                reward = next_state_tuple[2]
                sumOverNextStates = sumOverNextStates + prob_for_next_state * (reward + discount * V[next_state])

            sumOverActions += policy[s_idx][a_idx] * sumOverNextStates
        V_new[s_idx] = sumOverActions

    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here
    delta = np.infty
    i = 0
    while delta > theta:
        i = i + 1
        delta = 0
        V_new = policy_evaluation_one_step(mdp, V, policy, discount=discount)
        for idx in range(len(V)):
            v = V[idx]
            v_new = V_new[idx]
            if delta < np.abs(v - v_new):
                delta = np.abs(v - v_new)
        V = V_new
    return V

def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))

    # TODO: Write your implementation here
    for s_idx in range(mdp.num_states):
        for a_idx in range(mdp.num_actions):
            next_state_tuple = mdp.P[s_idx][a_idx]
            print("next state tuple")
            print(next_state_tuple)
            prob_for_next_state = next_state_tuple[0]
            next_state = next_state_tuple[1]
            reward = next_state_tuple[2]
            policy[s_idx][a_idx] = prob_for_next_state * (reward + discount * V[next_state])
    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    # TODO: Write your implementation here

    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here

    # Get the greedy policy w.r.t the calculated value function
    policy = policy_improvement(mdp, V)
    
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)