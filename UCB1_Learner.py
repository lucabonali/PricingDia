from Learner import *
import numpy as np


class UCB1_Learner(Learner):

    def __init__(self, n_arms, margins):
        """
        Initialization of the UCB1:
        :param n_arms: number of arms
        :param margins: margins vector
        :self.bounds = bound of each arm
        """
        super().__init__(n_arms, margins)
        self.bounds = np.zeros(n_arms)

    def pull_arm(self):
        """
        Selection of the arm:
        We select the index of the arm with the maximum bound
        (in case of ties we choose randomly)
        :return: the pulled arm
        """
        if self.t < self.n_arms:
            return self.t
        margin_bounds = self.bounds * self.margins
        idxs = np.argwhere(margin_bounds == margin_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        """
        Function that update the parameters of the pulled arm
        :param pulled_arm: the selected arm
        :param reward: the associated reward
        """
        self.update_observations(pulled_arm, reward)

        if self.t < self.n_arms:
            self.bounds[pulled_arm] = 10000
        else:
            self.bounds[pulled_arm] = np.mean(self.samples_per_arm[pulled_arm])+ \
                                      np.sqrt(2*np.log(self.t)/(len(self.samples_per_arm[pulled_arm])-1))
        self.t += 1
