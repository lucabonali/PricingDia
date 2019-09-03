"""
Thomson Sampling learner
Also SuperClass of the Sliding Window Thomson Sampling learner
"""

from Learner import *
import numpy as np


class TS_Learner(Learner):

    def __init__(self, n_arms, margins, classes = []):
        """
        Initialization of the TS Learner
        :param n_arms: number of candidates
        :param margins: margins associated to each candidate
        :param classes: the classes that is learned
        :self beta_parameters = beta distribution parameters initialization for all the candidates
        """
        super().__init__(n_arms, margins)
        self.beta_parameters = np.ones((n_arms, 2))
        self.classes = classes

    def pull_arm(self):
        """
        Selection of the arm to pull at each round t.
        The reward is computed as: beta samples * candidates margins
        :return: the index of the candidate with max reward
        """
        samples_from_beta = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        idx = np.argmax(samples_from_beta * self.margins)
        return idx

    #return the best expected reward and the arm
    def get_best(self):
        samples_from_beta = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        idx = np.argmax(samples_from_beta * self.margins)
        best_reward = np.max(samples_from_beta * self.margins)
        return idx, best_reward


    def update(self, pulled_arm, reward):
        """
        Update of the beta distribution parameters and the rewards of the learner
        :param pulled_arm: the selected arm
        :param reward: the reward of the environment
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward
