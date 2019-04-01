import matplotlib.pyplot as plt
import Data


"""
Module for the plotting of the graphs:
    - demands
    - rewards
"""


def plot_phase_curves(phase):
    """
    Plot the demand curves of the selected phase
    :param phase: the selected phase
    """
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.title('Demand curves of phase: %i' %phase)
    for i in range(len(Data.total_curve_values)):
        plt.plot(Data.total_curve_values[i][phase][1], Data.total_curve_values[i][phase][0], linewidth=1, markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


def plot_rewards(phase):
    """
    Plot the reward curves of the selected phase
    :param phase:
    """
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Reward")
    plt.title('Reward curves of phase: %i' % phase)
    for i in range(len(Data.total_curve_values)):
        plt.plot(Data.total_curve_values[i][phase][1], Data.total_curve_values[i][phase][0]*Data.margins, linewidth=1, markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


'''
Plotting of the demands and the rewards
'''
for p in range(len(Data.total_curve_values[0])):
    plot_phase_curves(p)

for p in range(len(Data.total_curve_values[0])):
    plot_rewards(p)
