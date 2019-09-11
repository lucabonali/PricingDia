#The alghoritm that start with ythe aggregate curve and every week check if it have to disaggregate

import Data
from NonStationaryEnvironment import *
from UCB1_Learner import *
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
margins = Data.margins

#array that saves the pulled class each time
#pulled_classes_round = []

t_horizon = int(Data.t_horizon)
window_size = int(np.sqrt(t_horizon))

n_experiments = 100

checking_samples = 5

reward_per_experiment = []
cl0_reward_per_experiment = []
cl1_reward_per_experiment = []
cl2_reward_per_experiment = []

regrets_per_experiment = []


n_phases = len(p_agg)
phases_length = Data.samples_per_phase

agg_rewards_per_phase = []
cl0_rewards_per_phase = []
cl1_rewards_per_phase = []
cl2_rewards_per_phase = []
disag_rewards_per_phase = [cl0_rewards_per_phase, cl1_rewards_per_phase, cl2_rewards_per_phase]

#for each phase we calculate the true rewards of each cadidate
for p in range(n_phases):
    agg_rewards_per_phase.append(p_agg[p] * margins)
    cl0_rewards_per_phase.append(p_class0[p] * margins)
    cl1_rewards_per_phase.append(p_class1[p] * margins)
    cl2_rewards_per_phase.append(p_class2[p] * margins)
#print(cl0_rewards_per_phase)

#for each phase we calculate the optimum reward (wrt aggregate curve)
agg_opt_per_phases = np.array(agg_rewards_per_phase).max(axis=1)
'''
opt_candidate_per_phase = np.argmax(np.array(cl0_rewards_per_phase), axis=1)
opt_per_phases_cl0 = np.array(cl0_rewards_per_phase).max(axis=1) / [p_class0[x, val] for x, val in enumerate(opt_candidate_per_phase)]

opt_candidate_per_phase = np.argmax(np.array(cl1_rewards_per_phase), axis=1)
opt_per_phases_cl1 = np.array(cl1_rewards_per_phase).max(axis=1) / [p_class1[x, val] for x, val in enumerate(opt_candidate_per_phase)]

opt_candidate_per_phase = np.argmax(np.array(cl2_rewards_per_phase), axis=1)
opt_per_phases_cl2 = np.array(cl2_rewards_per_phase).max(axis=1) / [p_class2[x, val] for x, val in enumerate(opt_candidate_per_phase)]
'''

opt_per_phases_cl0 = np.array(cl0_rewards_per_phase).max(axis=1)
opt_per_phases_cl1 = np.array(cl1_rewards_per_phase).max(axis=1)
opt_per_phases_cl2 = np.array(cl2_rewards_per_phase).max(axis=1)

disag_opt_per_phase = [opt_per_phases_cl0, opt_per_phases_cl1, opt_per_phases_cl2]

agg_opt_per_round = np.zeros(t_horizon)
opt_per_round_cl0 = np.zeros(t_horizon)
opt_per_round_cl1 = np.zeros(t_horizon)
opt_per_round_cl2 = np.zeros(t_horizon)

#this array tell us at which samples a phase begins and when it stops
cumulative_samples = np.cumsum(phases_length)

for i in range(n_phases):
    #print("Phase: " + i.__str__())
    if i == 0:
        agg_opt_per_round[0 : cumulative_samples[i]] = agg_opt_per_phases[i]
        opt_per_round_cl0[0 : cumulative_samples[i]] = opt_per_phases_cl0[i]
        opt_per_round_cl1[0 : cumulative_samples[i]] = opt_per_phases_cl1[i]
        opt_per_round_cl2[0 : cumulative_samples[i]] = opt_per_phases_cl2[i]
    else:
        agg_opt_per_round[cumulative_samples[i-1] : cumulative_samples[i]] = agg_opt_per_phases[i]
        opt_per_round_cl0[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl0[i]
        opt_per_round_cl1[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl1[i]
        opt_per_round_cl2[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl2[i]

disag_opt_per_round = [opt_per_round_cl0, opt_per_round_cl1, opt_per_round_cl2]

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
    #print("\n")
    #print(agg_rew)
    #print(idx)
    return idx, idx != len(aggregations) - 1, t



for e in range(n_experiments):
    #stdout.write("\rexperiment number: %d" %e)
    #stdout.flush()
    print("\nrun experiment number: {}".format(e))
    print(" ")

    aggregate = True

    # creation of the learners
    agg_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [0, 1, 2])
    cl0_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [0])
    cl1_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [1])
    cl2_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [2])
    cl01_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [0, 1])
    cl02_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [0, 2])
    cl12_ts_learner = UCB1_Learner(n_arms=n_arms, margins=margins, classes = [1, 2])

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

    regrets = np.zeros(t_horizon)

    for t in range(0, t_horizon):

        #if it's the firts day of the week, the alghoritm check if is better the aggregate cure or the disaggregate ones
        if ((t % Data.samples_per_week) == 0) and aggregate: #and force_after == 0:
            best_agg_idx, aggregate, disaggregation_time = check_aggregation(aggregations, t)

            if best_agg_idx != 0 and len(aggregations) == 5:
                #print(best_agg_idx)
                active_learners = aggregations[best_agg_idx]
                print("\n\tActive learners {}".format(active_learners), end='')

                if best_agg_idx == 1:
                    print("\n\tperforming disaggregation 1", end='')
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[1])
                    aggregations.remove(aggregations[1])
                elif best_agg_idx == 2:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[1])
                    print("\n\tperforming disaggregation 1", end='')
                elif best_agg_idx == 3:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    print("\n\tperforming disaggregation 1", end='')
                elif best_agg_idx == 4:
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    aggregations.remove(aggregations[0])
                    print("\n\tperforming disaggregation 2\n", end='')
            elif best_agg_idx != 0:
                active_learners = aggregations[best_agg_idx]
                aggregations.remove(aggregations[0])
                print("\n\tActive learners {}".format(active_learners), end='')
                print("\n\tperforming disaggregation 3")

        pulled_class = np.random.choice(np.arange(len(class_probabilities)), 1, p=class_probabilities)[0]

        #pull the arm from the active learners
        for learner in active_learners:
            if pulled_class in learner.classes:
                pulled_arm = learner.pull_arm()
                break

        reward = environments[pulled_class].round(pulled_arm)

        for enviroment in environments:
            if enviroment != environments[pulled_class]:
                enviroment.inc_time()

        collected_rewards = np.append(collected_rewards, reward * margins[pulled_arm])

        # update the learners that learn the pulled class
        for learner in learners:
            if pulled_class in learner.classes:
                learner.update(pulled_arm, reward)

        #calculate the regret
        regret = disag_opt_per_round[pulled_class][t] - (reward * margins[pulled_arm])
        regrets[t] = regret
        '''
        print("\noptimum: {}".format(disag_opt_per_round[pulled_class][t]))
        print("reward: {}".format(reward))
        print("regret: {}".format(regret))
        print("pulled class: {}".format(pulled_class))
        print("pulled arm: {}".format(pulled_arm))
        '''


    # update the reward per experiment
    reward_per_experiment.append(collected_rewards)
    # update the regret per experiment
    regrets_per_experiment.append(regrets)
    #print("\n")

regrets_per_experiment = np.array(regrets_per_experiment)
print(regrets_per_experiment.shape)


plt.figure(2)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(reward_per_experiment, axis=0), 'r')
#plt.plot(opt_per_round_cl0)
#plt.plot(opt_per_round_cl1)
#plt.plot(opt_per_round_cl2)
plt.plot(agg_opt_per_round)
plt.legend(["Reward", "Aggregate optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("Time")
plt.plot(np.cumsum(np.mean(regrets_per_experiment, axis=0)))
plt.axvline(cumulative_samples[0])
plt.axvline(cumulative_samples[1])
plt.axvline(cumulative_samples[2])
plt.legend(["Regret"])
plt.show()

