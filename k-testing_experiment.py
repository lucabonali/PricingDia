"""
Module for the k-testing experiment:
    - alpha: significance level
    - beta: power level
    - delta: alternative hypothesis relaxing coefficient

NB: K-Testing is done considering only one phase
"""

import Data
from KTesting import *
from sys import stdout
import numpy as np

# Tester parameters
alpha = 0.005
beta = 0.85
delta = 0.05

candidates = Data.k_testing_candidates
n_experiment = 100

best_per_exp = []

# Computation of the best candidate for each phase
best_candidates = []
tester = KTesting(alpha, beta, delta, candidates[0])

for e in range(n_experiment):
    stdout.write("\rexperiment number: %d" %e)
    stdout.flush()
    #print("experiment number: {}".format(e))
    best_candidates.append(tester.sequential_testing())
    best_per_exp.append(tester.sequential_testing())

print(best_per_exp)

print("\nminimum: {}".format(np.min(np.array(best_per_exp))))
print("maximum: {}".format(np.max(np.array(best_per_exp))))
print("mean: {}".format(np.mean(np.array(best_per_exp))))
