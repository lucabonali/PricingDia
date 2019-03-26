from Learner import *
import numpy as np


class UCB1_Learner(Learner):

    '''
    Initialization of the UCB1:
        - number of arms
        - margins vector
        - bound of each arm
    '''
    def __init__(self, n_arms, margins):
        super().__init__(n_arms, margins)
        self.bounds = np.zeros(n_arms)

    '''
    Selection of the arm:
        We select the index of the arm with the maximum bound
        (in case of ties we choose randomly)
    '''
    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        margin_bounds = self.bounds * self.margins
        idxs = np.argwhere(margin_bounds == margin_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm


    '''
    Function that update the parameters of the pulled arm
        - pulled_arm: the selected arm
        - reward: the reward
    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # In order to avoid the division by 0,
        # because at the beginning we have all the arms with no samples
        if self.t <= self.n_arms:
            self.bounds[pulled_arm] = np.mean(self.samples_per_arm[pulled_arm])+np.sqrt(2*np.log(self.t)/0.00001)
        else:
            self.bounds[pulled_arm] = np.mean(self.samples_per_arm[pulled_arm])+ \
                                      np.sqrt(2*np.log(self.t)/(len(self.samples_per_arm[pulled_arm])-1))

