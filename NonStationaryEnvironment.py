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

class Non_Stationary_Environment(Environment):

    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities) #probabilities are the probabilty parameter value for each phase (n_phases x n_arms)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        n_phases = len(self.probabilities) #n_phases equals to the number of rows of probability matrix
        phase_size = self.horizon / n_phases
        current_phase = int(self.t / phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        reward = np.random.binomial(1, p)
        return reward