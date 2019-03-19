'''
ENVIRONMENT CLASS

The environment is defined by:
	- a number of arms;
	- a probability distribution for each arm reward function.

The environment interacts with the learner by returning a stochastic
reward depending on the pulled arm.
'''

import numpy as np


class Environment:

    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        #n: numero di prove effettuate.
        #p: probabilità di successo della singola prova di Bernoulli
        return reward
