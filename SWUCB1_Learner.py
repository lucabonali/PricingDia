from UCB1_Learner import *

#TODO: l'algoritmo sembra disimparare (rcapire come mai)
class SWUCB1_Learner(UCB1_Learner):

    '''
    Initialization of SWUCB1
        - UCB1 parameters
        - size of the sliding window
    '''
    def __init__(self, n_arms, margins, window_size):
        super().__init__(n_arms, margins)
        self.window_size = window_size

    '''
    Update of the bound of the selected arm
        - pulled_arm: the selected arm
        - reward: the reward
        - n_rounds_arm: the last window_size samples of the selected arm
        - windowed_mean: the mean of the last window_size samples
    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        n_rounds_arm = len(self.samples_per_arm[pulled_arm][-self.window_size:])
        windowed_mean = np.mean(self.samples_per_arm[pulled_arm][-self.window_size:])

        self.bounds[pulled_arm] = windowed_mean+np.sqrt(2*np.log(self.t)/n_rounds_arm)
