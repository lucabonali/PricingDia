"""
Module for the k-testing experiment:
    - alpha: significance level
    - beta: power level
    - delta: alternative hypothesis relaxing coefficient

NB: K-Testing is done considering only one phase
"""

import Data
from KTesting import *

# Tester parameters
alpha = 0.005
beta = 0.85
delta = 0.05

candidates = Data.k_testing_candidates

# Computation of the best candidate for each phase
best_candidates = []
for p in range(len(Data.total_curve_values[0])):
    tester = KTesting(alpha, beta, delta, candidates[p])
    best_candidates.append(tester.sequential_testing())

print(best_candidates)
