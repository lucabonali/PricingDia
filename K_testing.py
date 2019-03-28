import scipy.stats as scs
import numpy as np
from math import *


'''
Sequential K-testing
    - H0: u1 = u2 -> try the new price
    - H1: u1 > u2 -> keep the old price

https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f

'''

class K_testing():
    def __init__(self, alpha, beta, delta, candidates):
        self.alpha = alpha  # 0,05
        self.beta = beta  # 0,8
        self.delta = delta  # 0,02
        self.candidates = candidates

    '''
    Sequential A/B testing:
    loop all the selected candidates and compare them with the temporary best one,
    if the new one is better then it becomes the temporary best one
    candidates: [(price, probability, margin),...]
    '''

    def sequential_testing(self):
        best_candidate = self.candidates[0]
        for i in range(1, len(self.candidates)):
            best_candidate = self.compare_candidates(best_candidate, self.candidates[i])
        return best_candidate[0]


    def get_min_number_of_samples(self, a_prob, b_prob):

        p = (a_prob + b_prob) / 2

        sigma = p * (1 - p)

        z_alpha = scs.norm(0, 1).ppf(1 - self.alpha)
        z_beta = scs.norm(0, 1).ppf(self.beta)

        return (sigma * (z_alpha + z_beta)**2) / self.delta ** 2

    def compare_candidates(self, a, b):

        a_prob = a[1]
        b_prob = b[1]

        a_margin = a[2]
        b_margin = b[2]

        num_of_samples = ceil(self.get_min_number_of_samples(a_prob, b_prob))

        a_samples = 0
        b_samples = 0

        # split of all the samples
        for i in range(num_of_samples):
            if np.random.binomial(1, 0.5) > 0.5:
                a_samples += 1
            else:
                b_samples += 1

        emp_mean_a = np.mean([np.random.binomial(1, a_prob) for _ in range(a_samples)])
        if emp_mean_a == 0:
            emp_mean_a = a_prob
        emp_mean_b = np.mean([np.random.binomial(1, b_prob) for _ in range(b_samples)])
        if emp_mean_b == 0:
            emp_mean_b = b_prob

        pooled_mean = (a_samples * emp_mean_a + b_samples * emp_mean_b) / num_of_samples

        # To verify if it is correct to insert the margins here
        z = (emp_mean_a * a_margin - emp_mean_b * b_margin) / np.sqrt(
            pooled_mean * (1 - pooled_mean) * (1 / a_samples + 1 / b_samples))

        z_alpha = scs.norm(0, 1).ppf(1 - self.alpha)

        # if z > z(1-alpha) we reject H0 (keep u1), otherwise we keep it
        better_one = a if z > z_alpha else b
        # print(z, z_alpha, better_one)

        return better_one

'''

#print(ceil(x.get_min_number_of_samples(0.2, 0.8)))

x.compare_candidates((1, 0.2, 1), (2, 0.2, 1)) # 2 is selected (z ~ 0)
x.compare_candidates((1, 0.8, 1), (2, 0.2, 1)) # 1 is selected (z > 0)
x.compare_candidates((1, 0.2, 1), (2, 0.8, 1)) # 2 should be selected (z < 0)

'''