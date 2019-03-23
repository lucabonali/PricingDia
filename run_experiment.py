import Data
#from MainPlot import *
import matplotlib as plt
import numpy as np
from NonStationaryEnvironment import *
from TS_Learner import *
from SWTS_Learner import *
import matplotlib.pyplot as plt


n_arms = Data.n_arms

p_class1 = Data.get_class_probabilities(0)
p_class2 = Data.get_class_probabilities(1)
p_class3 = Data.get_class_probabilities(2)
p_agg = Data.get_class_probabilities(3)

margins = Data.margins

t_horizon = Data.t_horizon
window_size = int(np.sqrt(t_horizon))
print(window_size)

n_experiments = 500

ts_reward_per_experiment = []
swts_reward_per_experiment = []

a = np.array([1,2,3])
b = a

print(a*b)

for e in range(n_experiments):
    print("experiment number: {}".format(e))
    ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon)
    ts_learner = TS_Learner(n_arms=n_arms, margins=margins)

    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon)
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size, margins=margins)

    for t in range(0, t_horizon):
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    swts_reward_per_experiment.append(swts_learner.collected_rewards)

ts_instantaneous_regret = np.zeros(t_horizon)
swts_instantaneous_regret = np.zeros(t_horizon)
n_phases = len(p_agg)
phases_lenght = Data.samples_per_phase

rewards_per_phases = []
for p in range(n_phases):
    rewards_per_phases.append(p_agg[p] * margins)


opt_per_phases = np.array(rewards_per_phases).max(axis=1)
opt_per_round = np.zeros(t_horizon)

cumulative_samples = np.cumsum(phases_lenght)


for i in range(n_phases):
    if(i==0):
        opt_per_round[0 : cumulative_samples[i]] = opt_per_phases[i]
    else:
        opt_per_round[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i]
    #print(opt_per_round[i*phases_lenght[i] : (i+1)*phases_lenght[i]])
    if(i==0):
        print(opt_per_phases[i])
        ts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(ts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
        swts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(swts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
    else:
        ts_instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                                                                                   np.mean(ts_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

        swts_instantaneous_regret[cumulative_samples[i - 1]: cumulative_samples[i]] = opt_per_phases[i] - \
                                                                                    np.mean(swts_reward_per_experiment, axis=0)[cumulative_samples[i - 1]: cumulative_samples[i]]

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
plt.plot(opt_per_round, '--k')
plt.legend(["TS", "Optimum"])
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("regret")
plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
plt.legend(["TS"])
plt.show()



