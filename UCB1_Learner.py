from Learner import *
import numpy as np


class UCB1_Learner(Learner):

    def __init__(self, n_arms, margins, classes = []):
        """
        Initialization of the UCB1:
        :param n_arms: number of arms
        :param margins: margins vector
        :self.bounds = bound of each arm
        """
        super().__init__(n_arms, margins)
        self.bounds = np.ones(n_arms) * 1000
        self.classes = classes

    def pull_arm(self):
        """
        Selection of the arm:
        We select the index of the arm with the maximum bound
        (in case of ties we choose randomly)
        :return: the pulled arm
        """
        # if self.t < self.n_arms:
        #     return self.t
        margin_bounds = self.bounds * self.margins
        idxs = np.argwhere(margin_bounds == margin_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    #return the best expected reward and the arm
    def get_best(self, delta = 0.9):
        bound = np.sqrt(-(np.log(delta))/len(self.collected_rewards))

        margin_bounds = (self.bounds - bound) * self.margins
        idxs = np.argwhere(margin_bounds == margin_bounds.max()).reshape(-1)
        idx = np.random.choice(idxs)
        best_reward = margin_bounds.max()
        return idx, best_reward

    def update(self, pulled_arm, reward):
        """
        Function that update the parameters of the pulled arm
        :param pulled_arm: the selected arm
        :param reward: the associated reward
        """
        self.update_observations(pulled_arm, reward)

        mean = np.mean(self.samples_per_arm[pulled_arm])
        n_rounds_arm = len(self.samples_per_arm[pulled_arm])

        self.bounds[pulled_arm] = mean + np.sqrt(2 * np.log(self.t+1) / n_rounds_arm)

        self.t += 1
