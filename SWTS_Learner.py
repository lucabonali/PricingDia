"""
Sliding Window Thomson Sampling learner
It exploits a sliding window to use only the last observations for the updating of the beta distribution parameters
"""

from TS_Learner import *

class SWTS_Learner(TS_Learner):

    def __init__(self, n_arms, margins, window_size):
        """
        Initalization of the SWTS Learner
        :param n_arms: number of candidates
        :param margins: margins associated to each candidate
        :param window_size: size of the sliding window
        """
        super().__init__(n_arms, margins)
        self.window_size = window_size

    '''
    Update of the beta parameters considering
    '''
    def update(self, pulled_arm, reward):
        """
        Update of the beta distribution parameters and the rewards of the learner considering the sliding window
        :param pulled_arm: the selected arm
        :param reward: the reward of the environment
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)

        cum_rew = np.sum(self.samples_per_arm[pulled_arm][-self.window_size:])
        n_rounds_arm = len(self.samples_per_arm[pulled_arm][-self.window_size:])
        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0
