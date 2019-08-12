from UCB1_Learner import *


class SWUCB1_Learner(UCB1_Learner):

    def __init__(self, n_arms, margins, window_size):
        """
        Initialization of SWUCB1
        :param n_arms: number of arms
        :param margins: margins vector
        :param window_size: the size of the sliding window
        :self.sample_timestamp = for each arm: array of timestamps when the arm was pulled
        """
        super().__init__(n_arms, margins)
        self.window_size = window_size
        self.sample_timestamp = [[] for _ in range(n_arms)]
        # print(self.window_size)

    def update(self, pulled_arm, reward):
        """
        Update of the bound of the selected arm
        :param pulled_arm: the selected arm
        :param reward: the reward

        n_rounds_arm: the last window_size samples of the selected arm
        windowed_mean: the mean of the last window_size samples
        """
        self.update_timestamp(pulled_arm)
        self.update_observations(pulled_arm, reward)

        if self.t < self.n_arms:
            self.bounds[pulled_arm] = 10000
        else:
            n_rounds_arm = len(self.samples_per_arm[pulled_arm])
            windowed_mean = np.mean(self.samples_per_arm[pulled_arm])
            self.bounds[pulled_arm] = windowed_mean+np.sqrt(2*np.log(self.t)/n_rounds_arm)

    def update_timestamp(self, pulled_arm):
        """
        Update of the samples considering the sliding window.
        For the selected arm, remove the "older" samples:
        if the sample at the head of the array is older than the sliding window tail it is removed.
        Repeat until all the "older" samples are removed from the array
        :param pulled_arm: the selected arm/candidate
        """
        self.sample_timestamp[pulled_arm].append(self.t)

        while self.sample_timestamp[pulled_arm][0] < (self.t - self.window_size):
            self.sample_timestamp[pulled_arm] = self.sample_timestamp[pulled_arm][1:]
            self.samples_per_arm[pulled_arm] = self.samples_per_arm[pulled_arm][1:]

        self.t += 1
