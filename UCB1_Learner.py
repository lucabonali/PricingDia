from Learner import *
import numpy as np

#TODO: metodi
class UCB1_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    '''
    Function that decide which arm pull at each round t,
    by sampling a value for each arm from a beta distribution and 
    then select the arm associated with the beta distribution that
    generate the sample with the maximun value
    '''
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        #armax return the position of the maximun value. Quindi in questo caso mi ritorna l'arm da pullare
        #cioè quella con il valore più grande
        return idx

    '''
    Function that update the parameters of the pulled arm
    '''
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward