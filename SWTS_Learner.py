'''
SWTS-LEARNER CLASS
exploits a sliding window to use only recent observations to update the beta distribution parameters

We will define this class as an extension of the TS-Learner class
'''

from TS_Learner import *

class SWTS_Learner(TS_Learner):

    '''
    Initialization of the SWTS Learner:
        - number of arms/candidates
        - margins associated to the candidates
        - size of the window
    '''
    def __init__(self, n_arms, margins, window_size):
        super().__init__(n_arms, margins)
        self.window_size = window_size

    '''
    Update of the beta parameters considering
    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # vado a prendere gli ultimi window size rewards e li sommo
        cum_rew = np.sum(self.samples_per_arm[pulled_arm][-self.window_size:])
        n_rounds_arm = len(self.samples_per_arm[pulled_arm][-self.window_size:])
        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0
        # + 1.0 al posto di fare il max

        #La formula è uguale a quella delle slides, però qui non ho proprio la liding window
        #tengo gli ultimi valori
