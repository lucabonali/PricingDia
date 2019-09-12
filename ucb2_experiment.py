import numpy as np
from UCB2 import ucb2
from Enviroment import *

# probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
# margins = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

probabilities = np.array([0.9, 0.8])
margins = np.array([10, 20])

opt = np.max[probabilities * margins]

n_exp = 100

n_arms = len(probabilities)

rewards_per_experimet = [[] for _ in n_exp]

for e in range(n_exp):
    learner = ucb2(n_arms= n_arms, margins= margins)
    env = Environment(n_arms= n_arms, probabilities= probabilities)

    for i in range(100):
        pulled_arm = learner.pull_arm()
        reward = env.round(pulled_arm)
        learner.update(pulled_arm, reward)

    rewards_per_experimet[e] = learner.collected_rewards
