import Learner
import math
import numpy as np

class ucb2(Learner):

    def __init__(self, n_arms, margins, classes = []):
        super().__init__(n_arms, margins)
        self.classes = classes
        self.emp_means = np.array([0 for _ in n_arms])
        self.n_samples_per_arm = np.array([0 for _ in n_arms])
        self.bounds = np.array([100000 for _ in n_arms])

    def pull_arm(self):
        upper_bounds = self.emp_means + self.bounds
        arms = np.argwhere(upper_bounds == upper_bounds.max())
        arm = np.random.choice(arms)

        return arm


    def update(self, pulled_arm, reward):

        super().update_observations(pulled_arm, reward)

        margin = reward * self.margins[pulled_arm]
        n_samples = self.n_samples_per_arm[pulled_arm]
        self.emp_means[pulled_arm] = ((self.emp_means * n_samples) + margin) / (n_samples + 1)
        self.n_samples_per_arm[pulled_arm] += 1
        self.bounds[pulled_arm] = math.sqrt(2*math.log(self.t+1 / self.n_samples_per_arm[pulled_arm]))
