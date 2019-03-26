'''
LEARNER CLASS (SuperClass of the TS algorithm and the UCB1 algorithm)

A learner object is defined by:
	- the number of arms that he can pull;
	- the current round;
	- the list of the collected rewards.

The learner interacts with the environment by selecting the arm to pull
at each round and observing the reward given by the environment.

'''

import numpy as np

class Learner:

    '''
    Initialization of the Learner:
        - number of arms
        - time
        - rewards obtained by each arm for each round
        - samples obtained by each arm
        - total collected rewards
    '''
    def __init__(self, n_arms, margins):
        self.n_arms = n_arms
        self.margins = margins
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.samples_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])


    '''
    Update of rewards:
        - add the obtained reward to the samples
        - add the obtained reward to the selected arm, considering the margin
        - add the obtained reward to the total rewards, considering the margin
    '''
    def update_observations(self, pulled_arm, reward):
        self.samples_per_arm[pulled_arm].append(reward)
        self.rewards_per_arm[pulled_arm].append(reward * self.margins[pulled_arm])
        self.collected_rewards = np.append(self.collected_rewards, reward * self.margins[pulled_arm])
