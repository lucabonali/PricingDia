"""
Class that perform the sequential K-testing:
    - H0: u1 = u2 -> try the new price
    - H1: u1 > u2 -> keep the old price u1
"""

import scipy.stats as scs
import numpy as np
from math import *


class KTesting():
    def __init__(self, alpha, beta, delta, candidates):
        """
        Initialization of the tester for K-testing
        :param alpha: significance level
        :param beta: power level
        :param delta: alternative hypothesis relaxing coefficient
        :param candidates: list of candidates of the form (price, probability, margin)
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.candidates = candidates

    def sequential_testing(self):
        """
        Sequential A/B testing:
        Sequential comparison with the temporary best candidate and the next one in candidates list
        :return: best candidate among all the candidates
        """
        best_candidate = self.candidates[0]

        for i in range(1, len(self.candidates)):
            best_candidate = self.compare_candidates(best_candidate, self.candidates[i])

        return best_candidate[0]

    def get_min_number_of_samples(self, a_prob, b_prob):
        """
        Computation of the minimum number of samples for current A/B testing
        :param a_prob: probability of candidate a
        :param b_prob: probability of candidate b
        :return: min number if samples for a good testing
        """
        p = (a_prob + b_prob) / 2
        sigma = p * (1 - p)

        z_alpha = scs.norm(0, 1).ppf(1 - self.alpha)
        z_beta = scs.norm(0, 1).ppf(self.beta)

        return (sigma * (z_alpha + z_beta) ** 2) / self.delta ** 2

    def compare_candidates(self, a, b):
        """
        A/B testing between candidate a and candidate b:
            - Computation of the minimum number of samples
            - Split of the samples (50% for each candidate)
            - Computation of Z
            - Hypothesis testing: reject H0 (a is selected), not reject H0 (b is selected)
        :param a: old candidate
        :param b: new candidate
        :return: best candidate among a and b
        """
        a_prob = a[1]
        b_prob = b[1]
        a_margin = a[2]
        b_margin = b[2]

        # Computation of the number of samples
        num_of_samples = ceil(self.get_min_number_of_samples(a_prob, b_prob))

        # Split of the samples in 2 groups
        a_samples = 0
        b_samples = 0
        for i in range(num_of_samples):
            if np.random.binomial(1, 0.5) > 0.5:
                a_samples += 1
            else:
                b_samples += 1

        # If the means are 0, because the process is a Bernoulli, we use directly the probabilities or the candidate,
        # otherwise we'll have a division by 0 in the computation of Z
        emp_mean_a = np.mean([np.random.binomial(1, a_prob) for _ in range(a_samples)])
        if emp_mean_a == 0:
            emp_mean_a = a_prob
        emp_mean_b = np.mean([np.random.binomial(1, b_prob) for _ in range(b_samples)])
        if emp_mean_b == 0:
            emp_mean_b = b_prob

        pooled_mean = (a_samples * emp_mean_a + b_samples * emp_mean_b) / num_of_samples

        z = (emp_mean_a * a_margin - emp_mean_b * b_margin) / np.sqrt(
            pooled_mean * (1 - pooled_mean) * (1 / a_samples + 1 / b_samples))

        z_alpha = scs.norm(0, 1).ppf(1 - self.alpha)

        # candidate selection:
        # if z > z(1-alpha) we reject H0, otherwise we don't
        better_one = a if z > z_alpha else b

        return better_one
