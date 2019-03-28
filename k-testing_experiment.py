import Data
from K_testing import *

#TODO: chiedere al prof se il k-testing dobbiamo farlo per ogni fase (strano perchè noi le fasi non dovremmo saperle)
alpha = 0.005
beta = 0.85
delta = 0.05

candidates = Data.k_testing_candidates
best_candidates = []

for p in range(len(Data.total_curve_values[0])):
    tester = K_testing(alpha, beta, delta, candidates[p])

    best_candidates.append(tester.sequential_testing())

print(best_candidates)
