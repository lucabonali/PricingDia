from Learner import *
import numpy as np
import numpy.linalg as linalg


class TS_Learner(Learner):

    def __init__(self, n_arms, margins):
        super().__init__(n_arms, margins)
        self.beta_parameters = np.ones((n_arms, 2))

    '''
    Selection of the arm pull at each round t:
    sampling of a value for each arm from the beta distribution and 
    selection of the arm associated to the maximum reward.
    Reward = beta samples * candidates margins 
    '''
    def pull_arm(self):
        samples_from_beta = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        # idx is the index of the candidate with max reward
        idx = np.argmax(samples_from_beta * self.margins)
        return idx

    '''
    Updating of the parameters of the beta distribution of the pulled arm
    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward
