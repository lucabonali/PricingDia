#In this file, the comparison between alghotritm wit aggregate curve and alghpritm with disaggregate curves

import Data
from NonStationaryEnvironment import *
from SWTS_Learner import *
import matplotlib.pyplot as plt
from sys import stdout

n_arms = Data.n_candidates

p_class0 = Data.get_class_probabilities(0)
p_class1 = Data.get_class_probabilities(1)
p_class2 = Data.get_class_probabilities(2)
p_agg = Data.get_class_probabilities(3)

disagg_reward = []

#array of probabilities. Probabilities that a sample came from class i
class_probabilities = Data.weights

#boolean that tell us if the alghoritm prefere the aggregation
aggregate = True

#time at which the alghoritm choose to disaggregate
#disaggregation_time = Data.t_horizon

#all the dissaggregation time of the experiments
#disaggregation_times = []
margins = Data.margins

t_horizon = int(Data.t_horizon)
#t_horizon = Data.t_horizon
window_size = int(np.sqrt(t_horizon))

n_experiments = 100

agg_ts_reward_per_experiment = []
cl0_reward_per_experiment = []
cl1_reward_per_experiment = []
cl2_reward_per_experiment = []
disagg_reward_per_experiment = []

'''
def check_aggregation(disag_learners, disagg_reward, agg_learner, time):
    """
    Check if it's better the aggregation curve or the disaggregate curves
    :param disag_learners: array of learners of esch class
    :param agg_learner: learner of the aggregate class
    :param time: time in which the check is done
    :return: aggregate, time
    if the aggregate is set to false return false and the time
    if the aggregate is set to true return true and the time horizion
    """
    if time==0:
        return True, Data.t_horizon

    #print("check for disaggregate at time: " + time.__str__())
    agg_ts_cumulative_reward = np.cumsum(agg_learner.collected_rewards)

    dis_cum_reward = np.cumsum(disagg_reward)



    if agg_ts_cumulative_reward[-1] < dis_cum_reward[-1]:
        #print("disaggregation time: " + disaggregation_time.__str__())
        return False, time
    else:
        return True, Data.t_horizon

'''



for e in range(n_experiments):
    stdout.write("\rexperiment number: %d" %e)
    stdout.flush()
    #print("\rexperiment number: {}".format(e))


    #creation of the learners
    agg_ts_learner = TS_Learner(n_arms=n_arms, margins=margins)
    cl0_ts_learner = TS_Learner(n_arms=n_arms, margins=margins)
    cl1_ts_learner = TS_Learner(n_arms=n_arms, margins=margins)
    cl2_ts_learner = TS_Learner(n_arms=n_arms, margins=margins)

    #creation of the environments
    env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg,
                                    horizon=t_horizon, samples_per_phase=Data.samples_per_phase)

    cl0_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class0,
                                        horizon=t_horizon, samples_per_phase = Data.samples_per_phase)

    cl1_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class1,
                                        horizon=t_horizon, samples_per_phase = Data.samples_per_phase)

    cl2_dis_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_class2,
                                        horizon=t_horizon, samples_per_phase = Data.samples_per_phase)


    disag_learners = np.array([cl0_ts_learner, cl1_ts_learner, cl2_ts_learner])
    disag_environments = np.array([cl0_dis_env, cl1_dis_env, cl2_dis_env])

    aggregate = True
    disagg_reward = []
    for t in range(0, t_horizon):
        '''
        #if it's the firts day of the week, the alghoritm check if is better the aggregate cure or the disaggregate ones
        if ((t % Data.samples_per_week) == 0) and aggregate:
            aggregate, disaggregation_time = check_aggregation(disag_learners, disagg_reward, agg_ts_learner, t)
            if not aggregate:
                disaggregation_times.append(disaggregation_time)
        '''

        #run the aggregate
        pulled_arm = agg_ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        agg_ts_learner.update(pulled_arm, reward)

        #run the disaggregate
        pulled_class = np.random.choice(np.arange(len(class_probabilities)), 1, p=class_probabilities)[0]
        pulled_arm = disag_learners[pulled_class].pull_arm()
        reward = disag_environments[pulled_class].round(pulled_arm)
        disag_learners[pulled_class].update(pulled_arm, reward)
        disagg_reward.append(disag_learners[pulled_class].get_last_reward())

        for i in range(len(disag_environments)):
            if i != pulled_class:
                disag_environments[i].inc_time()



    #update the reward per experiments
    agg_ts_reward_per_experiment.append(agg_ts_learner.collected_rewards)
    disagg_reward_per_experiment.append(np.array(disagg_reward))


#print("length of the disagg array: {}".format(np.array(disagg_reward_per_experiment).shape))
#print("length of the agg array: {}".format(np.array(agg_ts_reward_per_experiment).shape))

agg_ts_instantaneous_regret = np.zeros(t_horizon)
disagg_ts_instantaneous_regret = np.zeros(t_horizon)

instantaneous_regret = np.zeros(t_horizon)

n_phases = len(p_agg)
phases_length = Data.samples_per_phase
#print("number of phases: " + n_phases.__str__())
#print("phases lenght: " + phases_length.__str__())

rewards_per_phases = []
cl0_rewards_per_phases = []
cl1_rewards_per_phases = []
cl2_rewards_per_phases = []
#print(probabilities.shape)
for p in range(n_phases):
    rewards_per_phases.append(p_agg[p] * margins)
    cl0_rewards_per_phases.append(p_class0[p] * margins)
    cl1_rewards_per_phases.append(p_class1[p] * margins)
    cl2_rewards_per_phases.append(p_class2[p] * margins)


#print("\nreward per phases: " + rewards_per_phases.__str__())

opt_per_phases = np.array(rewards_per_phases).max(axis=1)
opt_per_phases_cl0 = np.array(cl0_rewards_per_phases).max(axis=1)
opt_per_phases_cl1 = np.array(cl1_rewards_per_phases).max(axis=1)
opt_per_phases_cl2 = np.array(cl2_rewards_per_phases).max(axis=1)
#print("\noptimum per phases: " + opt_per_phases.__str__())

opt_per_round = np.zeros(t_horizon)
opt_per_round_cl0 = np.zeros(t_horizon)
opt_per_round_cl1 = np.zeros(t_horizon)
opt_per_round_cl2 = np.zeros(t_horizon)

cumulative_samples = np.cumsum(phases_length)

#disaggregation_time = int(np.mean(np.array(disaggregation_times)))
#disaggregation_time = int(3 / 4 * t_horizon)
#print("mean of disaggregation time: {}".format(disaggregation_time))

for i in range(n_phases):
    #print("Phase: " + i.__str__())
    if i == 0:
        opt_per_round[0 : cumulative_samples[i]] = opt_per_phases[i]
        opt_per_round_cl0[0 : cumulative_samples[i]] = opt_per_phases_cl0[i]
        opt_per_round_cl1[0 : cumulative_samples[i]] = opt_per_phases_cl1[i]
        opt_per_round_cl2[0 : cumulative_samples[i]] = opt_per_phases_cl2[i]
    else:
        opt_per_round[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i]
        opt_per_round_cl0[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl0[i]
        opt_per_round_cl1[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl1[i]
        opt_per_round_cl2[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases_cl2[i]
    # print(opt_per_round[i*phases_lenght[i] : (i+1)*phases_lenght[i]])

    if i == 0:
        agg_ts_instantaneous_regret [0:cumulative_samples[i]] = opt_per_phases[i] - \
                                                            np.mean(agg_ts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        disagg_ts_instantaneous_regret [0:cumulative_samples[i]] = opt_per_phases[i] - \
                                                           np.mean(disagg_reward_per_experiment, axis=0)[
                                                          0 : cumulative_samples[i]]
        '''
        # print(opt_per_phases[i])
        if cumulative_samples[i] <= disaggregation_time:
            #print("\tsetting the aggr regr")
            instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - \
                                                            np.mean(agg_ts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        else:
            #print("\tsetting the aggr and disag regr")
            instantaneous_regret[0: disaggregation_time] = opt_per_phases[i] - \
                                                             np.mean(agg_ts_reward_per_experiment, axis=0)[
                                                             0: disaggregation_time]
            #print(cumulative_samples[i] - disaggregation_time)
            #print(len(np.mean(disagg_reward_per_experiment, axis=0)))
            instantaneous_regret[disaggregation_time : cumulative_samples[i]] = opt_per_phases[i] - \
                                                           np.mean(disagg_reward_per_experiment, axis=0)[
                                                           disaggregation_time : cumulative_samples[i]]
        '''
    else:
        agg_ts_instantaneous_regret[cumulative_samples[i-1]:cumulative_samples[i]] = opt_per_phases[i] - \
                                                               np.mean(agg_ts_reward_per_experiment, axis=0)[
                                                               cumulative_samples[i-1]: cumulative_samples[i]]
        disagg_ts_instantaneous_regret[cumulative_samples[i-1]:cumulative_samples[i]] = opt_per_phases[i] - \
                                                                  np.mean(disagg_reward_per_experiment, axis=0)[
                                                                  cumulative_samples[i-1]: cumulative_samples[i]]
        '''
        if cumulative_samples[i] <= disaggregation_time :
            #print("setting the aggr regr")
            instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                                                                                  np.mean(agg_ts_reward_per_experiment, axis=0)[
                                                                                  cumulative_samples[i-1] : cumulative_samples[i]]
        elif cumulative_samples[i-1] >= disaggregation_time:
            #print("setting the disaggr regr")
            instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                                                                                  np.mean(disagg_reward_per_experiment, axis=0)[
                                                                                  cumulative_samples[i-1] : cumulative_samples[i]]

        else:
            #print("setting the aggr and disagg regr")
            instantaneous_regret[cumulative_samples[i-1] : disaggregation_time] = opt_per_phases[i] -\
                                                                                  np.mean(agg_ts_reward_per_experiment, axis=0)[
                                                                                  cumulative_samples[i-1] : disaggregation_time]

            instantaneous_regret[disaggregation_time : cumulative_samples[i]] = opt_per_phases[i] - \
                                                                                np.mean(disagg_reward_per_experiment, axis=0)[
                                                                                disaggregation_time : cumulative_samples[i]]
        '''

#print("\n\n optimum per round: {}".format(opt_per_round))
'''
final_reward = np.append(np.mean(agg_ts_reward_per_experiment, axis=0)[0:disaggregation_time],
                         np.mean(disagg_reward_per_experiment, axis=0)[disaggregation_time:])
'''
plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
#plt.plot(final_reward, 'r')
plt.plot(opt_per_round, '--k')
plt.legend(["Mixed Alghoritm"])
#plt.plot(opt_per_round_cl0, '--k')
#plt.plot(opt_per_round_cl1, '--k')
#plt.plot(opt_per_round_cl2, '--k')
#plt.axvline(disaggregation_time)
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(instantaneous_regret), 'r')
plt.plot(np.cumsum(agg_ts_instantaneous_regret), 'b')
plt.plot(np.cumsum(disagg_ts_instantaneous_regret), 'g')
#plt.axvline(disaggregation_time)
plt.axvline(cumulative_samples[0])
plt.axvline(cumulative_samples[1])
plt.axvline(cumulative_samples[2])
plt.legend(["Mixed Alghoritm", "Aggregate", "Disaggregate"])
#plt.axvline(disaggregation_time)
plt.show()

plt.figure(2)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(agg_ts_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(disagg_reward_per_experiment, axis=0), 'b')
plt.plot(opt_per_round, '--k')
plt.legend(["Aggregate", "Disaggregate", "aggregate optimum"])
#plt.plot(opt_per_round_cl0, '--k')
#plt.plot(opt_per_round_cl1, '--k')
#plt.plot(opt_per_round_cl2, '--k')
plt.show()

plt.figure(3)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(agg_ts_reward_per_experiment, axis=0), 'r')
plt.plot(opt_per_round, '--k')
print(opt_per_round[0])
plt.legend(["Aggregate", "aggregate optimum"])
#plt.plot(opt_per_round_cl0, '--k')
#plt.plot(opt_per_round_cl1, '--k')
#plt.plot(opt_per_round_cl2, '--k')
#plt.axvline(disaggregation_time)
plt.show()


