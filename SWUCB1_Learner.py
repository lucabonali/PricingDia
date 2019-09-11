from UCB1_Learner import *


class SWUCB1_Learner(UCB1_Learner):

    def __init__(self, n_arms, margins, window_size, classes=[]):
        """
        Initialization of SWUCB1
        :param n_arms: number of arms
        :param margins: margins vector
        :param window_size: the size of the sliding window
        :param classes: classes that are learned
        :self.sample_timestamp = for each arm: array of timestamps when the arm was pulled
        """
        super().__init__(n_arms, margins)
        self.window_size = window_size
        self.sample_timestamp = [[] for _ in range(n_arms)]
        self.classes = classes
        # print(self.window_size)

    def pull_arm(self):
        """
        Selection of the arm:
        We select the index of the arm with the maximum bound
        (in case of ties we choose randomly)
        :return: the pulled arm
        """
        # if self.t < self.n_arms:
        #     return self.t

        # for i in range(0, self.n_arms):
        #     if len(self.samples_per_arm[i]) == 0:
        #         return i

        margin_bounds = self.bounds * self.margins
        idxs = np.argwhere(margin_bounds == margin_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update2(self, pulled_arm, reward):
        """
        Function that update the parameters of the pulled arm
        :param pulled_arm: the selected arm
        :param reward: the associated reward
        """
        self.update_observations(pulled_arm, reward)

        # if self.t < self.n_arms:
        #     self.bounds[pulled_arm] = 0
        # else:
        #     n_rounds_arm = len(self.samples_per_arm[pulled_arm][-self.window_size:])
        #     windowed_mean = np.mean(self.samples_per_arm[pulled_arm][-self.window_size:])
        #     self.bounds[pulled_arm] = windowed_mean + np.sqrt(2 * np.log(self.t + 1) / n_rounds_arm)

        n_rounds_arm = len(self.samples_per_arm[pulled_arm][-self.window_size:])
        windowed_mean = np.mean(self.samples_per_arm[pulled_arm][-self.window_size:])

        self.bounds[pulled_arm] = windowed_mean + np.sqrt(2 * np.log(self.t + 1) / n_rounds_arm)

        self.t += 1

    def update(self, pulled_arm, reward):
        """
        Update of the bound of the selected arm
        :param pulled_arm: the selected arm
        :param reward: the reward
        """

        # if self.t > self.window_size:
        #     print("ok")

        # 1. Move the window (discard old samples for all the arms)
        self.sample_timestamp[pulled_arm].append(self.t)
        for i in range(0, self.n_arms):
            if len(self.sample_timestamp[i]) > 0:
                while self.sample_timestamp[i][0] < (self.t - self.window_size):
                    self.sample_timestamp[i] = self.sample_timestamp[i][1:]
                    self.samples_per_arm[i] = self.samples_per_arm[i][1:]
                    # print(i, self.sample_timestamp[i])
                    # print(i, self.samples_per_arm[i])
                    if len(self.sample_timestamp[i]) == 0:
                        break

        # 2. Update the confidence bounds of all the arms since the window has been moved
        self.update_observations(pulled_arm, reward)

        for i in range(0, self.n_arms):
            if len(self.sample_timestamp[i]) == 0:
                self.bounds[i] = 1000
            else:
                n_rounds_arm = len(self.samples_per_arm[i])
                windowed_mean = np.mean(self.samples_per_arm[i])
                self.bounds[i] = windowed_mean+np.sqrt(2*np.log(self.t+1)/n_rounds_arm)

        self.t += 1
