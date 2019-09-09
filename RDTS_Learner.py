import numpy as np
import Data

class RDTS_Learner():

    def __init__(self, aggregations, class_probabilities, n_arms, margins):
        """
        :param ts_learners: the list of the learners
        :param class_probabilities: the probabilities that the classes occur
        """
        self.samples_per_arm = []
        self.aggregations = aggregations
        self.aggregate = True
        self.checking_samples = 5
        self.class_probabilities = class_probabilities
        self.runned_learner = None
        self.active_learners = [aggregations[0]]
        self.is_monday = False

        self.n_arms = n_arms
        self.margins = margins
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.samples_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])


    def check_aggregation(self):

        agg_rew = np.array([])
        for aggs in self.aggregations:
            sum_ = 0
            for e in range(self.checking_samples):

                pulled_class = np.random.choice(np.arange(len(self.class_probabilities)), 1, p=self.class_probabilities)[0]

                for learner in aggs:

                    if pulled_class in learner.classes:
                        pulled_arm, reward = learner.get_best()
                        sum_ += reward

            agg_rew = np.append(agg_rew, (sum_ / self.checking_samples))
            idx = np.argmax(agg_rew)
        self.aggregate = idx != len(self.aggregations) - 1
        self.update_aggregations(idx)

    def update_aggregations(self, best_agg_idx, aggregate):
        if best_agg_idx != 0 and len(self.aggregations) == 5:

            self.active_learners = self.aggregations[best_agg_idx]

            if best_agg_idx == 1:
                print("\tperforming disaggregation 1", end='')
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[1])
                self.aggregations.remove(self.aggregations[1])
            elif best_agg_idx == 2:
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[1])
                print("\tperforming disaggregation 1", end='')
            elif best_agg_idx == 3:
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                print("\tperforming disaggregation 1", end='')
            elif best_agg_idx == 4:
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                self.aggregations.remove(self.aggregations[0])
                print("\tperforming disaggregation 2\n", end='')
        elif best_agg_idx != 0:
            self.aggregations.remove(self.aggregations[0])
            print("\tperforming disaggregation 3")

    def pull_arm(self, pulled_class):
        """
        Selection of the arm to pull at each round t.
        The reward is computed as: beta samples * candidates margins
        :return: the index of the candidate with max reward
        """
        for learner in self.active_learners:
            if pulled_class in learner.classes:
                self.runned_learner = learner
                arm = learner.pull_arm()
                break

        return arm

    def update(self, pulled_arm, reward):
        """
        Update of the beta distribution parameters and the rewards of the learner
        :param pulled_arm: the selected arm
        :param reward: the reward of the environment
        """
        self.samples_per_arm[pulled_arm].append(reward)
        self.rewards_per_arm[pulled_arm].append(reward * self.margins[pulled_arm])
        self.collected_rewards = np.append(self.collected_rewards, reward * self.margins[pulled_arm])

        self.t += 1
        self.is_monday = self.t % Data.samples_per_week == 0
        self.runned_learner.update_observations(pulled_arm, reward)
        self.runned_learner.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.runned_learner.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward

        if self.is_monday and self.aggregate:
            self.check_aggregation(self.aggregations, self.t)

