'''
The parameters of probability distributions of the arms change during time

5 phases

NON - STATIONARY ENVIROMENT CLASS
A Non - Stationary Environment is an Environment in which the arms reward functions
are dependent from the current time.
    - We need to specify, for each arm, a reward function which is dependent from the current time.
We will define this class as an extension of the Environment CLASS
'''

from Enviroment import *
import Data

class Non_Stationary_Environment(Environment):

    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities) #probabilities are the probabilty parameter value for each phase (n_phases x n_arms)
        self.t = 0
        self.horizon = horizon
        self.phase_sizes = Data.samples_per_phase
        self.n_phases = len(self.probabilities) #n_phases equals to the number of rows of probability matrix
        self.cumulative_samples_per_size = np.cumsum(self.phase_sizes)

    def get_current_phase(self, time):
        for i in range(self.n_phases):
            if(time < self.cumulative_samples_per_size[i]):
                return i
        return self.n_phases

    def round(self, pulled_arm):
        current_phase = self.get_current_phase(self.t)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        reward = np.random.binomial(1, p)
        return reward