from UCB1_Learner import *

#TODO: l'algoritmo sembra disimparare (capire come mai)
class SWUCB1_Learner(UCB1_Learner):

    '''
    Initialization of SWUCB1
        - UCB1 parameters
        - size of the sliding window
    '''
    def __init__(self, n_arms, margins, window_size):
        super().__init__(n_arms, margins)
        self.window_size = window_size
        self.sample_timestamp = [[] for _ in range(n_arms)]
        # print(self.window_size)

    '''
    Update of the bound of the selected arm
        - pulled_arm: the selected arm
        - reward: the reward
        - n_rounds_arm: the last window_size samples of the selected arm
        - windowed_mean: the mean of the last window_size samples
    '''
    def update(self, pulled_arm, reward):
        self.update_timestamp(pulled_arm)
        self.update_observations(pulled_arm, reward)

        # print("time",self.sample_timestamp)
        # print("arm:",self.samples_per_arm)

        n_rounds_arm = len(self.samples_per_arm[pulled_arm])
        windowed_mean = np.mean(self.samples_per_arm[pulled_arm])
        self.bounds[pulled_arm] = windowed_mean+np.sqrt(2*np.log(self.t)/n_rounds_arm)

    def update_timestamp(self, pulled_arm):
        """
        Update of the samples considering the sliding window.
        For the selected arm, remove the "older" samples:
        if the sample at the head of the array is older than the sliding window tail it is removed.
        Repeat until all the "older" samples are remove from the array
        :param pulled_arm: the selected arm/candidate
        """
        self.sample_timestamp[pulled_arm].append(self.t)

        while self.sample_timestamp[pulled_arm][0] < (self.t - self.window_size):
            self.sample_timestamp[pulled_arm] = self.sample_timestamp[pulled_arm][1:]
            self.samples_per_arm[pulled_arm] = self.samples_per_arm[pulled_arm][1:]

        self.t += 1
