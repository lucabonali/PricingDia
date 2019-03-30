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

    def __init__(self, n_arms, margins):
        """
        Initialization of the Learner:
        :param n_arms: number of candidates
        :param margins: candidates margins
        :self t: time initialization
        :self rewards_per_arm: matrix where we collect the rewards for each arm
        :self samples_per_arm: matrix where we collect the samples of each arm
        :self collected_rewards: total rewards
        """
        self.n_arms = n_arms
        self.margins = margins
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.samples_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        Update of the rewards and samples matrices after a sample is observed. Update of the total rewards.
        :param pulled_arm: the selected arm associated to the candidate
        :param reward: the result of the sample
        """
        self.samples_per_arm[pulled_arm].append(reward)
        self.rewards_per_arm[pulled_arm].append(reward * self.margins[pulled_arm])
        self.collected_rewards = np.append(self.collected_rewards, reward * self.margins[pulled_arm])
