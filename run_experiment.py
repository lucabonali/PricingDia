from NonStationaryEnvironment import *
from SWTS_Learner import *
from SWUCB1_Learner import *
import matplotlib.pyplot as plt
from sys import stdout


n_arms = Data.n_candidates - 4

p_class1 = [x[4:] for x in Data.get_class_probabilities(0)]
p_class2 = [x[4:] for x in Data.get_class_probabilities(1)]
p_class3 = [x[4:] for x in Data.get_class_probabilities(2)]
p_agg = [x[4:] for x in Data.get_class_probabilities(3)]

margins = Data.margins[4:]

K_vals = [0.1, 0.2, 0.5, 1, 5, 10, 20, 60, 1000]

swts_reward_per_k = np.array(len(K_vals))
swts_regret_per_k = np.array(len(K_vals))

swucb1_reward_per_k = []
swucb1_regret_per_k = []

run_ts = False
run_swts = False
run_ucb1 = False
run_swucb1 = True

t_horizon = Data.t_horizon

for K in K_vals:
    print("K: {}".format(K))
    thompson_window_size = int(np.sqrt(t_horizon)*K)
    ucb1_window_size = int(np.sqrt(t_horizon)*K)

    n_experiments = 10

    ts_reward_per_experiment = []
    ucb1_reward_per_experiment = []
    swts_reward_per_experiment = []
    swucb1_reward_per_experiment = []

    for e in range(n_experiments):
        stdout.write("\rexperiment number: %d" % e)
        stdout.flush()

        if run_ts:
            ts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
            ts_learner = TS_Learner(n_arms=n_arms, margins=margins)

        if run_ucb1:
            ucb1_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
            ucb1_learner = UCB1_Learner(n_arms=n_arms, margins=margins)

        if run_swts:
            swts_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
            swts_learner = SWTS_Learner(n_arms=n_arms, window_size=thompson_window_size, margins=margins)

        if run_swucb1:
            swucb1_env = NonStationaryEnvironment(n_arms=n_arms, probabilities=p_agg, horizon=t_horizon, samples_per_phase=Data.samples_per_phase)
            swucb1_learner = SWUCB1_Learner(n_arms=n_arms, window_size=ucb1_window_size, margins=margins)

        for t in range(0, t_horizon):
            if run_ts:
                pulled_arm = ts_learner.pull_arm()
                reward = ts_env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward)
            if run_ucb1:
                pulled_arm = ucb1_learner.pull_arm()
                reward = ucb1_env.round(pulled_arm)
                ucb1_learner.update(pulled_arm, reward)
            if run_swts:
                pulled_arm = swts_learner.pull_arm()
                reward = swts_env.round(pulled_arm)
                swts_learner.update(pulled_arm, reward)
            if run_swucb1:
                pulled_arm = swucb1_learner.pull_arm()
                reward = swucb1_env.round(pulled_arm)
                swucb1_learner.update(pulled_arm, reward)

        if run_ts:
            ts_reward_per_experiment.append(ts_learner.collected_rewards)
        if run_ucb1:
            ucb1_reward_per_experiment.append(ucb1_learner.collected_rewards)
        if run_swts:
            swts_reward_per_experiment.append(swts_learner.collected_rewards)
        if run_swucb1:
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
            if run_ts:
                ts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(ts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
            if run_ucb1:
                ucb1_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(ucb1_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
            if run_swts:
                swts_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(swts_reward_per_experiment, axis=0)[0 : cumulative_samples[i]]
            if run_swucb1:
                swucb1_instantaneous_regret[0: cumulative_samples[i]] = opt_per_phases[i] - np.mean(swucb1_reward_per_experiment, axis=0)[0: cumulative_samples[i]]

        else:
            if run_ts:
                ts_instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                            np.mean(ts_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

            if run_ucb1:
                ucb1_instantaneous_regret[cumulative_samples[i-1] : cumulative_samples[i]] = opt_per_phases[i] -\
                            np.mean(ucb1_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

            if run_swts:
                swts_instantaneous_regret[cumulative_samples[i - 1]: cumulative_samples[i]] = opt_per_phases[i] - \
                            np.mean(swts_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

            if run_swucb1:
                swucb1_instantaneous_regret[cumulative_samples[i - 1]: cumulative_samples[i]] = opt_per_phases[i] - \
                            np.mean(swucb1_reward_per_experiment, axis=0)[cumulative_samples[i-1] : cumulative_samples[i]]

    if run_ts and run_ucb1:
        plt.figure(0)
        plt.title("Reward TS and UCB1")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
        plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'b')
        plt.plot(opt_per_round, '--k')
        plt.legend(["TS", "UCB1", "Optimum"])
        plt.show()

    if run_swts and run_swucb1:
        plt.figure(1)
        plt.title("Reward SWTS and SWUCB1")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'r')
        plt.plot(np.mean(swucb1_reward_per_experiment, axis=0), 'b')
        plt.plot(opt_per_round, '--k')
        plt.legend(["SWTS = {}".format(thompson_window_size), "SWUCB1 = {}".format(ucb1_window_size), "Optimum"])
        plt.show()

    if run_ts:
        plt.figure(2)
        plt.title("Reward TS")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
        plt.plot(opt_per_round, '--k')
        plt.legend(["TS", "Optimum"])
        plt.show()

    if run_ucb1:
        plt.figure(2)
        plt.title("Reward UCB1")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'r')
        plt.plot(opt_per_round, '--k')
        plt.legend(["UCB1", "Optimum"])
        plt.show()

    if run_swts:
        plt.figure(3)
        plt.title("Reward SWTS")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'r')
        plt.plot(opt_per_round, '--k')
        plt.legend(["SWTS = {}".format(thompson_window_size), "Optimum"])
        plt.show()

    if run_swucb1:
        plt.figure(5)
        plt.title("Reward SWUCB1")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(swucb1_reward_per_experiment, axis=0), 'r')
        plt.plot(opt_per_round, '--k')
        plt.legend(["SWUCB1 = {}".format(ucb1_window_size), "Optimum"])
        plt.show()

    if run_ts and run_ucb1:
        plt.figure(6)
        plt.title("Regret TS and UCB1")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
        plt.plot(np.cumsum(ucb1_instantaneous_regret), 'b')
        plt.legend(["TS", "UCB1"])
        plt.show()

    if run_swts and run_swucb1:
        plt.figure(7)
        plt.title("Regret SWTS and SWUCB1")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(swts_instantaneous_regret), 'r')
        plt.plot(np.cumsum(swucb1_instantaneous_regret), 'b')
        plt.legend(["SWTS = {}".format(thompson_window_size), "SWUCB1 = {}".format(ucb1_window_size)])
        plt.show()

    if run_ts:
        plt.figure(8)
        plt.title("Regret TS")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
        plt.legend(["TS"])
        plt.show()

    if run_swts:
        plt.figure(9)
        plt.title("Regret SWTS")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(swts_instantaneous_regret), 'r')
        plt.legend(["SWTS = {}".format(thompson_window_size)])
        plt.show()

    if run_ucb1:
        plt.figure(10)
        plt.title("Regret UCB1")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(ucb1_instantaneous_regret), 'r')
        plt.legend(["UCB1"])
        plt.show()

    if run_swucb1:
        plt.figure(11)
        plt.title("Regret SWUCB1")
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(swucb1_instantaneous_regret), 'r')
        plt.legend(["SWUCB1 = {}".format(ucb1_window_size)])
        plt.show()

    if run_ucb1 and run_swucb1:
        plt.figure(12)
        plt.title("Reward UCB1 and SWUCB1 ")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(ucb1_reward_per_experiment, axis=0), 'r')
        plt.plot(np.mean(swucb1_reward_per_experiment, axis=0), 'b')
        plt.plot(opt_per_round, '--k')
        plt.legend(["UCB1", "SWUCB1 = {}".format(ucb1_window_size), "Optimum"])
        plt.show()

    if run_ts and run_swts:
        plt.figure(13)
        plt.title("Reward TS and SWTS")
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
        plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
        plt.plot(opt_per_round, '--k')
        plt.legend(["TS", "SWTS = {}".format(thompson_window_size), "Optimum"])
        plt.show()

    if run_ucb1 and run_swucb1:
        plt.figure(14)
        plt.title("Regret UCB1 and SWUCB1")
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(ucb1_instantaneous_regret), 'r')
        plt.plot(np.cumsum(swucb1_instantaneous_regret), 'b')
        plt.legend(["UCB1", "SWUCB1 = {}".format(ucb1_window_size)])
        plt.show()

    if run_ts and run_swts:
        plt.figure(15)
        plt.title("Regret TS and SWTS")
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
        plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
        plt.legend(["TS", "SWTS = {}".format(thompson_window_size)])
        plt.show()

    if run_swts:
        # swts_reward_per_k = np.append(swts_reward_per_k, np.mean(swts_reward_per_experiment))
        swts_regret_per_k = np.append(swts_regret_per_k, np.cumsum(swts_instantaneous_regret))

    if run_swucb1:
        # swucb1_reward_per_k.append(np.mean(swucb1_reward_per_experiment))
        swucb1_regret_per_k.append(np.cumsum(swucb1_instantaneous_regret))

if run_swts:
    plt.figure(16)
    plt.title("Regret SWTS")
    plt.xlabel("t")
    plt.ylabel("Regret")
    legend = []
    for i in range(len(K_vals)):
        plt.plot(swts_regret_per_k[i])
        legend.append("SWTS {}".format(int(np.sqrt(Data.t_horizon) * K_vals[i])))
    legend[len(K_vals) - 1] = "TS"
    plt.legend(legend)
    plt.show()

if run_swucb1:
    plt.figure(17)
    plt.title("Regret SWUCB1")
    plt.xlabel("t")
    plt.ylabel("Regret")
    legend = []
    for i in range(len(K_vals)):
        plt.plot(swucb1_regret_per_k[i])
        legend.append("SWUCB1 {}".format(int(np.sqrt(Data.t_horizon) * K_vals[i])))
    legend[len(K_vals)-1] = "UCB1"
    plt.legend(legend)
    plt.show()

