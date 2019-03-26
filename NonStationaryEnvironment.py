'''
The parameters of probability distributions of the arms change during time

NON - STATIONARY ENVIROMENT CLASS
A Non - Stationary Environment is an Environment in which the arms reward functions
are dependent from the current time.
    - We need to specify, for each arm, a reward function which is dependent from the current time.
We will define this class as an extension of the Environment CLASS
'''

from Enviroment import *

class Non_Stationary_Environment(Environment):

    '''
    Initialization of the Non-stat. environment:
        - number of arms/candidates
        - probabilities associated to each phase
        - time initialization
        - time horizon
        - number of phases
        - cumulative_samples_per_size:
    '''
    def __init__(self, n_arms, probabilities, horizon, samples_per_phase):

        # probabilities are the probability parameter value for each phase (n_phases x n_arms)
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon
        self.phase_sizes = samples_per_phase

        # n_phases equals to the number of rows of probability matrix
        self.n_phases = len(self.probabilities)
        self.cumulative_samples_per_size = np.cumsum(self.phase_sizes)

    '''
    Get the current phase:
        - time: the current time instant
    '''
    def get_current_phase(self, time):
        for i in range(self.n_phases):
            if(time < self.cumulative_samples_per_size[i]):
                return i
        return self.n_phases

    '''
    Get the reward of the pulled arm from a Bernoulli distribution,
    with p = pulled arm probability in the current phase
    '''
    def round(self, pulled_arm):
        current_phase = self.get_current_phase(self.t)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        reward = np.random.binomial(1, p)
        return reward
