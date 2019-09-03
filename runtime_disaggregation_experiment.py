#The alghoritm that start with ythe aggregate curve and every week check if it have to disaggregate

import Data
from NonStationaryEnvironment import *
from SWTS_Learner import *
from sys import stdout
import matplotlib.pyplot as plt

n_arms = Data.n_candidates

p_class0 = Data.get_class_probabilities(0)
p_class1 = Data.get_class_probabilities(1)
p_class2 = Data.get_class_probabilities(2)
p_agg = Data.get_class_probabilities(3)

#array of probabilities. Probabilities that a sample came from class i
class_probabilities = Data.weights

#boolean that tell us if the alghoritm prefere the aggregation
aggregate = True

#time at which the alghoritm choose to disaggregate
disaggregation_times = [Data.t_horizon]

#all the dissaggregation time of the experiments
disaggregation_times = []
margins = Data.margins

t_horizon = int(Data.t_horizon)
window_size = int(np.sqrt(t_horizon))

n_experiments = 100

checking_samples = 5

reward_per_experiment = []
cl0_reward_per_experiment = []
cl1_reward_per_experiment = []
cl2_reward_per_experiment = []

def check_aggregation(aggregations, time):

    agg_rew = np.array([])
    for aggs in aggregations:
        sum_ = 0
        for e in range(checking_samples):

            pulled_class = np.random.choice(np.arange(len(class_probabilities)), 1, p=class_probabilities)[0]

            for learner in aggs:

                if pulled_class in learner.classes:
                    pulled_arm, reward = learner.get_best()
                    sum_ += reward

        agg_rew = np.append(agg_rew, (sum_/checking_samples))
        idx = np.argmax(agg_rew)

    return idx, idx != len(aggregations) - 1, t



for e in range(n_experiments):
    stdout.write("\rexperiment number: %d" %e)
    stdout.flush()

    force_after = 600
    aggregate = True

    # creation of the learners
    agg_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [0, 1, 2])
    cl0_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [0])
    cl1_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [1])
    cl2_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [2])
    cl01_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [0, 1])
    cl02_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [0, 2])
    cl12_ts_learner = TS_Learner(n_arms=n_arms, margins=margins, classes = [1, 2])

    #the learners "active" at a certain time, at the beginnig the aggregate learner is actived
    active_learners = [agg_ts_learner]

    #possible aggregations
    aggregations = [[agg_ts_learner],
                    [cl01_ts_learner, cl2_ts_learner],
                    [cl02_ts_learner, cl1_ts_learner],
                    [cl12_ts_learner, cl0_ts_learner],
                    [cl0_ts_learner, cl1_ts_learner, cl2_ts_learner]]

    # ceration of the environment
    cl0_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class0,
                                           horizon=t_horizon, samples_per_phase=Data.samples_per_phase)

    cl1_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class1,
                                           horizon=t_horizon, samples_per_phase=Data.samples_per_phase)

    cl2_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class2,
                                           horizon=t_horizon, samples_per_phase=Data.samples_per_phase)

    learners = np.array([agg_ts_learner, cl0_ts_learner, cl1_ts_learner, cl2_ts_learner, cl01_ts_learner, cl02_ts_learner, cl12_ts_learner])
    environments = np.array([cl0_dis_env, cl1_dis_env, cl2_dis_env])

    collected_rewards = []

    for t in range(0, t_horizon):

        #if it's the firts day of the week, the alghoritm check if is better the aggregate cure or the disaggregate ones
        if ((t % Data.samples_per_week) == 0) and False: #and force_after == 0:
            best_agg_idx, aggregate, disaggregation_time = check_aggregation(aggregations, t)

            if best_agg_idx != 0 and len(aggregations) == 5:

                force_after = 600
                active_learners = aggregations[best_agg_idx]

                if best_agg_idx == 1:
                    print("\tperforming disaggregation 1", end='')
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[1])
                    aggregations.remove(aggregations[1])
                elif best_agg_idx == 2:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[1])
                    print("\tperforming disaggregation 1", end='')
                elif best_agg_idx == 3:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    print("\tperforming disaggregation 1", end='')
                elif best_agg_idx == 4:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    print("\tperforming disaggregation 2", end='')
            elif best_agg_idx != 0:
                aggregations.remove(aggregations[0])
                print("\tperforming disaggregation 3")


        if force_after > 0:
            force_after -=1

        pulled_class = np.random.choice(np.arange(len(class_probabilities)), 1, p=class_probabilities)[0]

        #pull the arm from the active learners
        for learner in active_learners:
            if pulled_class in learner.classes:
                pulled_arm = learner.pull_arm()
                break

        if pulled_arm == None:
            raise RuntimeError("No pulled arm")

        reward = environments[pulled_class].round(pulled_arm)

        for enviroment in environments:
            if enviroment != environments[pulled_class]:
                enviroment.inc_time()

        collected_rewards = np.append(collected_rewards, reward * margins[pulled_arm])

        # update the learners that learn the pulled class
        for learner in learners:
            if pulled_class in learner.classes:
                learner.update(pulled_arm, reward)


    # update the reward per experiments
    reward_per_experiment.append(collected_rewards)
    print("\n")

instantaneous_regret = np.zeros(t_horizon)

n_phases = len(p_agg)
phases_length = Data.samples_per_phase
rewards_per_phases = []
'''
for p in range(n_phases):
    rewards_per_phases.append(p_agg[p] * margins)
    cl0_rewards_per_phases.append(p_class0[p] * margins)
    cl1_rewards_per_phases.append(p_class1[p] * margins)
    cl2_rewards_per_phases.append(p_class2[p] * margins)
    '''

plt.figure(2)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(reward_per_experiment, axis=0), 'r')
plt.plot([131.13]*len(reward_per_experiment[0]), '--k')
plt.show()


