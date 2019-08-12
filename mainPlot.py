import matplotlib.pyplot as plt
import Data


"""
Module for the plotting of the graphs:
    - demands
    - rewards
"""


def plot_single_curves(phase, class_num):
    """
    Plot the demand curve of specific class and phase
    :param phase: the selected phase
    :param class_num: the selected class
    """
    legend = {0: "Class 1", 1: "Class 2", 2: "Class 3", 3: "Aggregated"}
    colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.ylim(0, 1.1)
    plt.title("Demand curve of {} of phase: {}".format(legend.get(class_num), phase+1))
    plt.plot(Data.total_curve_values[class_num][phase][1], Data.total_curve_values[class_num][phase][0],
             colors.get(class_num), linewidth=1, markersize=1)
    plt.legend([legend.get(class_num)])
    plt.show()


def plot_phase_curves(phase):
    """
    Plot the demand curves of the selected phase
    :param phase: the selected phase
    """
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.ylim(0, 1.1)
    plt.title('Demand curves of phase: {}'.format(phase+1))
    for i in range(len(Data.total_curve_values)):
        linewidth = 1
        if i == 3:
            linewidth = 3
        plt.plot(Data.total_curve_values[i][phase][1], Data.total_curve_values[i][phase][0], linewidth=linewidth,
                 markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


def plot_rewards(phase):
    """
    Plot the reward curves of the selected phase
    :param phase: the selected phase
    """
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Reward")
    plt.title('Reward curves of phase: {}'.format(phase+1))
    for i in range(len(Data.total_curve_values)):
        linewidth = 1
        if i == 3:
            linewidth = 3
        plt.plot(Data.total_curve_values[i][phase][1], Data.total_curve_values[i][phase][0]*Data.margins, linewidth=linewidth, markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


'''
Plotting of the demands and the rewards
'''
for p in range(len(Data.total_curve_values[0])):
    for c in range(len(Data.total_curve_values)):
        print(p,c)
        plot_single_curves(p, c)

for p in range(len(Data.total_curve_values[0])):
    plot_phase_curves(p)

for p in range(len(Data.total_curve_values[0])):
    plot_rewards(p)
