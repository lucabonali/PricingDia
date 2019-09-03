import Data
from NonStationaryEnvironment import *
from SWTS_Learner import *
from SWUCB1_Learner import *
import matplotlib.pyplot as plt


n_arms = Data.n_candidates

p_class1 = Data.get_class_probabilities(0)
p_class2 = Data.get_class_probabilities(1)
p_class3 = Data.get_class_probabilities(2)
p_agg = Data.get_class_probabilities(3)

margins = Data.margins

t_horizon = Data.t_horizon
# window_size = int(np.sqrt(t_horizon))
# window_size = int(t_horizon / 10)
window_size = 500

n_experiments = 50

ts_reward_per_experiment = []
ucb1_reward_per_experiment = []
swts_reward_per_experiment = []
swucb1_reward_per_experiment = []


for e in range(n_experiments):
    print("experiment number: {}".format(e))

    # ts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
    # ts_learner = TS_Learner(n_arms=n_arms, margins=margins)
    #
    ucb1_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
    ucb1_learner = UCB1_Learner(n_arms=n_arms, margins=margins)
    #
    # swts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
    # swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size, margins=margins)

    swucb1_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
    swucb1_learner = SWUCB1_Learner(n_arms=n_arms, window_size=window_size, margins=margins)

    for t in range(0, t_horizon):

        # pulled_arm = ts_learner.pull_arm()
        # reward = ts_env.round(pulled_arm)
        # ts_learner.update(pulled_arm, reward)
        #
        pulled_arm = ucb1_learner.pull_arm()
        reward = ucb1_env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)
        #
        # pulled_arm = swts_learner.pull_arm()
        # reward = swts_env.round(pulled_arm)
        # swts_learner.update(pulled_arm, reward)

        pulled_arm = swucb1_learner.pull_arm()
        reward = swucb1_env.round(pulled_arm)
        swucb1_learner.update(pulled_arm, reward)

    # ts_reward_per_experiment.append(ts_learner.collected_rewards)
    ucb1_reward_per_experiment.append(ucb1_learner.collected_rewards)
    # swts_reward_per_experiment.append(swts_learner.collected_rewards)
    swucb1_reward_per_experiment.append(swucb1_learner.collected_rewards)

ts_instantaneous_regret = np.zeros(t_horizon)
ucb1_instantaneous_regret = np.zeros(t_horizon)
swts_instantaneous_regret = np.zeros(t_horizon)
swucb1_instantaneous_regret = np.zeros(t_horizon)


n_phases = len(p_agg)
phases_length = Data.samples_per_phase
# print("number of phases: " + n_phases.__str__())
# print("phases lenght: " + phases_length.__str__())

rewards_per_phases = []
# print(p_agg.shape)
for p in range(n_phases):
    rewards_per_phases.append(p_agg[p] * margins)

# print("reward per phases: " + rewards_per_phases.__str__())


opt_per_phases = np.array(rewards_per_phases).max(axis=1)
# print("optimum per phases: " + opt_per_phases.__str__())
opt_per_round = np.zeros(t_horizon)

cumulative_samples = np.cumsum(phases_length)

for i in range(n_phases):
    if i == 0:
        opt_per_round[0 : cumulative_samples[i]] = opt_per_phases[i]
    else:
        opt_per_round[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i]
    if i == 0:
        # ts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(ts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        ucb1_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(ucb1_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        # swts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(swts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        swucb1_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(swucb1_reward_per_experiment, axis=0)[0: cumulative_samples[i]]

    else:
        # ts_instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
        #                 np.mean(ts_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]
        #
        ucb1_instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                        np.mean(ucb1_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]
        #
        # swts_instantaneous_regret[cumulative_samples[i - 1]: cumulative_samples[i]] = opt_per_phases[i] - \
        #                 np.mean(swts_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

        swucb1_instantaneous_regret[cumulative_samples[i - 1]: cumulative_samples[i]] = opt_per_phases[i] - \
                        np.mean(swucb1_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]


# plt.figure(0)
# plt.ylabel("Reward TS and SWTS")
# plt.xlabel("t")
# plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
# plt.plot(opt_per_round, '--k')
# plt.legend(["TS", "SWTS", "Optimum"])
# plt.show()
#
#
# plt.figure(1)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
# plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
# plt.plot(np.cumsum(ucb1_instantaneous_regret), 'g')
# plt.plot(np.cumsum(swucb1_instantaneous_regret), 'y')
# plt.legend(["TS", "SWTS", "UCB1", "SWUCB1"])
# plt.show()
#
#
# plt.figure(2)
# plt.ylabel("Reward UCB1 and SWUCB1")
# plt.xlabel("t")
# plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(swucb1_reward_per_experiment, axis=0), 'b')
# plt.plot(opt_per_round, '--k')
# plt.legend(["UCB1", "SWUCB1", "Optimum"])
# plt.show()
#
#
# plt.figure(3)
# plt.ylabel("Reward TS")
# plt.xlabel("t")
# plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
# plt.plot(opt_per_round, '--k')
# plt.legend(["TS", "Optimum"])
# plt.show()
#
# plt.figure(4)
# plt.ylabel("Reward SWTS")
# plt.xlabel("t")
# plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'r')
# plt.plot(opt_per_round, '--k')
# plt.legend(["SWTS", "Optimum"])
# plt.show()
#

plt.figure(5)
plt.ylabel("Reward UCB1")
plt.xlabel("t")
plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'r')
plt.plot(opt_per_round, '--k')
plt.legend(["UCB1", "Optimum"])
plt.show()


plt.figure(6)
plt.ylabel("Reward SWUCB1")
plt.xlabel("t")
plt.plot(np.mean(swucb1_reward_per_experiment, axis=0), 'r')
plt.plot(opt_per_round, '--k')
plt.legend(["SWUCB1", "Optimum"])
plt.show()
#
# plt.figure(7)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
# plt.legend(["TS"])
# plt.show()
#
# plt.figure(8)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum(swts_instantaneous_regret), 'r')
# plt.legend(["SWTS"])
# plt.show()
#
plt.figure(9)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(ucb1_instantaneous_regret), 'r')
plt.legend(["UCB1"])
plt.show()

plt.figure(10)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(swucb1_instantaneous_regret), 'r')
plt.legend(["SWUCB1"])
plt.show()

# plt.figure(12)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum(ucb1_instantaneous_regret), 'r')
# plt.plot(np.cumsum(swucb1_instantaneous_regret), 'b')
# plt.legend(["UCB1", "SWUCB1"])
# plt.show()
#
